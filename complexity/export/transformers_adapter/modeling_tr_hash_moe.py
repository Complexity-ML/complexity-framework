"""Self-contained Transformers implementation of TR-HASH MoE.

The module preserves the native checkpoint tensor names. Its universal
PyTorch expert path is the numerical reference; optimized serving remains the
job of an external serving runtime that implements this same routing law.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import GenerationMixin, PreTrainedModel
    from transformers.modeling_outputs import CausalLMOutputWithPast
except ModuleNotFoundError as exc:
    if exc.name != "transformers":
        raise
    raise ImportError(
        'The TR-HASH Transformers adapter requires transformers. '
        'Install with: pip install "complexity-framework[hf]"'
    ) from exc

from .composed_routing import (
    bind_value_channels,
    composed_head_routes,
    composed_mlp_routes,
    update_routing_history,
)
from .configuration_tr_hash_moe import TRHashConfig


def _expected_route_anchor(vocab_size, layer_index):
    tokens = torch.arange(vocab_size, device="cpu", dtype=torch.long).view(-1, 1)
    return composed_mlp_routes(tokens, layer_index).squeeze(-1)


def _validate_route_anchors(config, state_dict, *, require_all):
    for index in range(config.num_hidden_layers):
        key = f"layers.{index}.mlp.engine.route_table"
        if key not in state_dict:
            if require_all:
                raise RuntimeError(f"Missing canonical zero-history routing anchor: {key}")
            continue
        anchor = state_dict[key]
        expected = _expected_route_anchor(config.vocab_size, index)
        if (
            not isinstance(anchor, torch.Tensor)
            or anchor.dtype != torch.long
            or anchor.is_meta
            or not torch.equal(anchor.detach().cpu(), expected)
        ):
            raise RuntimeError(f"{key} must match the canonical zero-history routing anchor")

try:
    from transformers.cache_utils import Cache, DynamicCache
except ImportError:  # Older Transformers releases use tuple caches.

    class Cache:  # type: ignore[no-redef]
        pass

    DynamicCache = None  # type: ignore[assignment,misc]



if DynamicCache is not None:
    class TRHashCache(DynamicCache):
        """Dynamic KV cache whose routing IDs follow beam and crop operations."""
        def __init__(self, config=None):
            try:
                super().__init__(config=config)
            except TypeError:
                super().__init__()
            self.routing_token_ids = None
            self.routing_history = None

        def _select_routing(self, indices):
            for name in ("routing_token_ids", "routing_history"):
                value = getattr(self, name)
                if value is not None:
                    setattr(self, name, value.index_select(0, indices.to(value.device)))

        def reorder_cache(self, beam_idx):
            super().reorder_cache(beam_idx)
            self._select_routing(beam_idx)

        def batch_repeat_interleave(self, repeats):
            super().batch_repeat_interleave(repeats)
            for name in ("routing_token_ids", "routing_history"):
                value = getattr(self, name)
                if value is not None:
                    setattr(self, name, value.repeat_interleave(repeats, dim=0))

        def batch_select_indices(self, indices):
            super().batch_select_indices(indices)
            self._select_routing(indices)

        def crop(self, max_length):
            super().crop(max_length)
            if self.routing_token_ids is not None:
                self.routing_token_ids = self.routing_token_ids[:, :self.get_seq_length()]
                self.routing_history = update_routing_history(self.routing_token_ids)
else:
    TRHashCache = None

_GenerationBase = object if issubclass(PreTrainedModel, GenerationMixin) else GenerationMixin


class TRHashRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states


def _rotate_half(values: torch.Tensor) -> torch.Tensor:
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class TRHashRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_position_embeddings = max_position_embeddings

    def forward(
        self, position_ids: torch.LongTensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frequencies = torch.einsum("bi,j->bij", position_ids.float(), self.inv_freq.float())
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        return embeddings.cos().to(dtype=dtype), embeddings.sin().to(dtype=dtype)


class TRHashAttention(nn.Module):
    def __init__(self, config: TRHashConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_key_value_groups
        self.head_dim = config.head_dim
        self.dropout = config.attention_dropout

        kv_size = self.num_key_value_heads * self.head_dim
        self.k_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        if config.use_qk_norm:
            self.q_norm = TRHashRMSNorm(self.head_dim, 1e-6)
            self.k_norm = TRHashRMSNorm(self.head_dim, 1e-6)
        else:
            self.q_norm = None
            self.k_norm = None
        self.rotary_emb = TRHashRotaryEmbedding(
            self.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor,
        past_key_values: Cache | tuple[torch.Tensor, torch.Tensor] | None,
        use_cache: bool,
        cache_position: torch.LongTensor | None,
        token_ids: torch.LongTensor,
    ) -> tuple[torch.Tensor, Any]:
        batch_size, sequence_length, _ = hidden_states.shape
        key_states = self.k_proj(hidden_states)
        query_states = self.q_proj(hidden_states)
        value_states = self.v_proj(bind_value_channels(hidden_states, token_ids, self.layer_idx))
        query_states = query_states.view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if self.q_norm is not None:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)
        cos, sin = self.rotary_emb(position_ids, query_states.dtype)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        query_states = query_states * cos + _rotate_half(query_states) * sin
        key_states = key_states * cos + _rotate_half(key_states) * sin

        new_past: Any = None
        if isinstance(past_key_values, Cache):
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                {"cache_position": cache_position},
            )
            new_past = past_key_values if use_cache else None
        elif past_key_values is not None:
            key_states = torch.cat((past_key_values[0], key_states), dim=2)
            value_states = torch.cat((past_key_values[1], value_states), dim=2)
            new_past = (key_states, value_states) if use_cache else None
        elif use_cache:
            new_past = (key_states, value_states)

        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        query_length = query_states.shape[-2]
        key_length = key_states.shape[-2]
        is_causal = attention_mask is None and query_length == key_length
        attention_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
            scale=1.0 / math.sqrt(self.head_dim),
        )
        routes = composed_head_routes(token_ids, self.layer_idx, self.num_heads)
        selected = F.one_hot(routes, num_classes=self.num_heads).sum(dim=-2)
        attention_output = attention_output * selected.permute(0, 2, 1).unsqueeze(-1).to(attention_output.dtype) * math.sqrt(2)
        attention_output = attention_output.transpose(1, 2).reshape(batch_size, sequence_length, -1)
        return self.o_proj(attention_output), new_past


class TRHashExpertEngine(nn.Module):
    """Shared SwiGLU and four deterministic context-routed residual experts."""
    def __init__(self, config: TRHashConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_experts, self.top_k = 4, 2
        self.route_weights = (0.5, 0.5)
        self.shared_output_scale, self.routed_output_scale = 1.0, 2.0
        tokens = torch.arange(config.vocab_size, dtype=torch.long).view(-1, 1)
        self.register_buffer("route_table", composed_mlp_routes(tokens, layer_idx).squeeze(-1))
        self.register_buffer("fused_route_codes", torch.empty(0, dtype=torch.uint8))
        self.register_buffer("fused_expert_pairs", torch.empty(0, 2, dtype=torch.int32))
        self.expert_gate = nn.Parameter(torch.empty(4, config.hidden_size, config.expert_width))
        self.expert_up = nn.Parameter(torch.empty(4, config.hidden_size, config.expert_width))
        self.expert_down = nn.Parameter(torch.empty(4, config.expert_width, config.hidden_size))
        self.shared_gate = nn.Linear(config.hidden_size, config.shared_intermediate_size, bias=False)
        self.shared_up = nn.Linear(config.hidden_size, config.shared_intermediate_size, bias=False)
        self.shared_down = nn.Linear(config.shared_intermediate_size, config.hidden_size, bias=False)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        anchor = state_dict.get(prefix + "route_table")
        expected = _expected_route_anchor(self.route_table.shape[1], self.layer_idx)
        if anchor is not None and (
            not isinstance(anchor, torch.Tensor)
            or anchor.dtype != torch.long
            or anchor.is_meta
            or not torch.equal(anchor.detach().cpu(), expected)
        ):
            error_msgs.append(prefix + "route_table must match the canonical zero-history routing anchor")
            return
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    @property
    def experts(self):
        return self.expert_gate, self.expert_up, self.expert_down

    def forward(self, hidden_states, token_ids, previous_token_ids=None):
        batch_size, sequence_length, hidden_size = hidden_states.shape
        flat_states = hidden_states.reshape(-1, hidden_size)
        shared = self.shared_down(F.silu(self.shared_gate(flat_states)) * self.shared_up(flat_states))
        routes = composed_mlp_routes(token_ids, self.layer_idx, previous_token_ids).reshape(2, -1)
        route_weights = flat_states.new_tensor(self.route_weights).view(-1, 1)
        routed = torch.zeros_like(flat_states)
        for expert_index in range(4):
            token_weight = (routes.eq(expert_index).to(flat_states.dtype) * route_weights).sum(dim=0)
            active_states = flat_states * token_weight.ne(0).to(flat_states.dtype).unsqueeze(-1)
            intermediate = F.silu(active_states @ self.expert_gate[expert_index]) * (active_states @ self.expert_up[expert_index])
            expert_output = intermediate @ self.expert_down[expert_index]
            routed = routed + expert_output * token_weight.unsqueeze(-1)
        output = shared + 2.0 * routed
        return output.view(batch_size, sequence_length, hidden_size)


class TRHashMLP(nn.Module):
    def __init__(self, config: TRHashConfig, layer_idx: int) -> None:
        super().__init__()
        self.engine = TRHashExpertEngine(config, layer_idx)

    @property
    def experts(self):
        return self.engine.experts

    def forward(self, hidden_states, token_ids, previous_token_ids=None):
        return self.engine(hidden_states, token_ids, previous_token_ids)


class TRHashDecoderLayer(nn.Module):
    def __init__(self, config: TRHashConfig, layer_idx: int) -> None:
        super().__init__()
        self.input_layernorm = TRHashRMSNorm(config.hidden_size, config.norm_eps)
        self.self_attn = TRHashAttention(config, layer_idx)
        self.post_attention_layernorm = TRHashRMSNorm(config.hidden_size, config.norm_eps)
        self.mlp = TRHashMLP(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor,
        past_key_values: Any,
        use_cache: bool,
        cache_position: torch.LongTensor | None,
    ) -> tuple[torch.Tensor, Any]:
        previous_ids = None
        if isinstance(past_key_values, Cache):
            previous_ids = getattr(past_key_values, "routing_history", None)
        elif past_key_values is not None:
            if len(past_key_values) != 3:
                raise ValueError("TR-HASH cache entries require K, V and the last two routing IDs")
            previous_ids = past_key_values[2]
        residual = hidden_states
        attention_output, new_past = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask,
            position_ids,
            past_key_values,
            use_cache,
            cache_position,
            token_ids,
        )
        if use_cache and not isinstance(new_past, Cache):
            new_past = (*new_past, update_routing_history(token_ids, previous_ids))
        hidden_states = residual + attention_output
        return (
            hidden_states + self.mlp(self.post_attention_layernorm(hidden_states), token_ids, previous_ids),
            new_past,
        )


class TRHashForCausalLM(PreTrainedModel, _GenerationBase):
    config_class = TRHashConfig
    base_model_prefix = ""
    main_input_name = "input_ids"
    _supports_cache_class = True
    _supports_sdpa = True
    _no_split_modules = ["TRHashDecoderLayer"]

    def __init__(self, config: TRHashConfig) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            TRHashDecoderLayer(config, index) for index in range(config.num_hidden_layers)
        )
        self.norm = TRHashRMSNorm(config.hidden_size, config.norm_eps)
        # Besides initializing newly-created models, post_init records the
        # loading/tied-weight metadata required by both Transformers 4 and 5.
        # from_pretrained replaces every persisted tensor immediately after.
        self.post_init()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        _validate_route_anchors(self.config, state_dict, require_all=strict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @classmethod
    def _load_pretrained_model(cls, model, *args, **kwargs):
        # Transformers 4 and 5 use different argument names and loading paths.
        # Check actual checkpoint anchors before either path assigns any weights.
        parent = super()._load_pretrained_model
        arguments = inspect.signature(parent).bind(model, *args, **kwargs).arguments
        state_dict = arguments.get("state_dict")
        if state_dict is not None:
            anchors = state_dict
        else:
            files = arguments.get("checkpoint_files", arguments.get("resolved_archive_file"))
            if isinstance(files, (str, Path)):
                files = [files]
            anchors = {}
            for filename in files or []:
                path = Path(filename)
                if path.suffix == ".safetensors":
                    from safetensors import safe_open

                    with safe_open(path, framework="pt", device="cpu") as checkpoint:
                        for key in checkpoint.keys():
                            if key.endswith(".mlp.engine.route_table"):
                                anchors[key] = checkpoint.get_tensor(key)
                else:
                    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                    anchors.update({
                        key: value for key, value in checkpoint.items()
                        if key.endswith(".mlp.engine.route_table")
                    })
        _validate_route_anchors(model.config, anchors, require_all=True)
        return parent(model, *args, **kwargs)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, TRHashExpertEngine):
            nn.init.normal_(module.expert_gate, std=self.config.initializer_range)
            nn.init.normal_(module.expert_up, std=self.config.initializer_range)
            nn.init.normal_(
                module.expert_down,
                std=self.config.initializer_range / math.sqrt(2 * self.config.num_hidden_layers),
            )
        elif isinstance(module, TRHashRMSNorm):
            nn.init.ones_(module.weight)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.embed_tokens

    def set_output_embeddings(self, value):
        self.embed_tokens = value

    def tie_weights(self, *args, **kwargs):
        return None

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values=None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Bridge the generation contracts used by Transformers 4 and 5."""

        if isinstance(past_key_values, Cache):
            past_length = past_key_values.get_seq_length()
        elif past_key_values:
            past_length = past_key_values[0][0].shape[2]
        else:
            past_length = 0
        has_cached_values = past_length > 0
        full_input_ids = input_ids
        if has_cached_values:
            input_ids = input_ids[:, -1:]
        position_ids = kwargs.get("position_ids")
        if position_ids is None and attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask.eq(0), 0)
        if position_ids is not None:
            position_ids = position_ids[:, -input_ids.shape[1] :]
        model_inputs: dict[str, Any]
        if inputs_embeds is not None and not has_cached_values:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        routing_ids = kwargs.get("routing_ids")
        if routing_ids is not None:
            if routing_ids.ndim != 2 or routing_ids.shape[0] != input_ids.shape[0]:
                raise ValueError("routing_ids must match the generation batch")
            if has_cached_values:
                # Generation retains prompt kwargs. Only an explicitly extended
                # aligned sequence supplies routing IDs for newly decoded tokens.
                if (
                    routing_ids.shape == full_input_ids.shape
                    and routing_ids.shape[1] > past_length
                ):
                    routing_ids = routing_ids[:, -input_ids.shape[1]:]
                else:
                    routing_ids = input_ids
            elif inputs_embeds is None:
                prefix_length = routing_ids.shape[1]
                if prefix_length > input_ids.shape[1]:
                    raise ValueError("routing_ids exceed the generation input length")
                # Without KV caching, preserve the prompt's original routing
                # IDs and append the actual generated token IDs on every pass.
                routing_ids = torch.cat((routing_ids, input_ids[:, prefix_length:]), dim=1)
            model_inputs["routing_ids"] = routing_ids
        elif inputs_embeds is not None and not has_cached_values and input_ids.shape[:2] == inputs_embeds.shape[:2]:
            model_inputs["routing_ids"] = input_ids
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache", True),
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx: torch.LongTensor):
        if isinstance(past_key_values, Cache):
            past_key_values.reorder_cache(beam_idx)
            return past_key_values
        return tuple(
            tuple(state.index_select(0, beam_idx.to(state.device)) for state in layer)
            for layer in past_key_values
        )

    @staticmethod
    def _causal_padding_mask(
        attention_mask: torch.Tensor | None,
        batch_size: int,
        query_length: int,
        key_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        needs_offset_causal_mask = key_length != query_length and query_length > 1
        has_padding = attention_mask is not None and not bool(attention_mask.all())
        if not needs_offset_causal_mask and not has_padding:
            return None
        minimum = torch.finfo(dtype).min
        query_positions = torch.arange(
            key_length - query_length, key_length, device=device
        ).unsqueeze(-1)
        key_positions = torch.arange(key_length, device=device).unsqueeze(0)
        causal = key_positions > query_positions
        mask = causal.view(1, 1, query_length, key_length).expand(
            batch_size, 1, query_length, key_length
        )
        if attention_mask is None:
            padding = torch.zeros_like(mask)
        else:
            padding = attention_mask[:, None, None, :key_length].eq(0)
        return torch.zeros(mask.shape, dtype=dtype, device=device).masked_fill(
            mask | padding, minimum
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | tuple | None = None,
        inputs_embeds: torch.Tensor | None = None,
        routing_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast | tuple:
        cache_position = kwargs.pop("cache_position", None)
        if output_attentions:
            raise NotImplementedError("TR-HASH does not return attention weights")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("input_ids or inputs_embeds must be provided")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify only one of input_ids and inputs_embeds")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        output_hidden_states = (
            self.config.output_hidden_states
            if output_hidden_states is None
            else output_hidden_states
        )
        return_dict = self.config.return_dict if return_dict is None else return_dict

        hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        batch_size, query_length = hidden_states.shape[:2]
        token_ids = routing_ids if routing_ids is not None else input_ids
        if token_ids is None:
            raise ValueError("routing_ids must accompany inputs_embeds for TR-HASH")
        if token_ids.shape != (batch_size, query_length) or token_ids.dtype != torch.long or token_ids.device != hidden_states.device:
            raise ValueError("routing_ids must be same-device torch.long [batch, sequence]")
        if isinstance(past_key_values, Cache):
            past_length = past_key_values.get_seq_length()
            if not isinstance(past_key_values, TRHashCache):
                if past_length or not isinstance(past_key_values, DynamicCache):
                    raise ValueError("TR-HASH requires TRHashCache for a nonempty cache")
                past_key_values = TRHashCache(config=self.config)
            if past_length and past_key_values.routing_history is None:
                raise ValueError("TR-HASH cache is missing routing history")
        elif past_key_values:
            past_length = past_key_values[0][0].shape[2]
            if any(len(entry) != 3 for entry in past_key_values):
                raise ValueError("TR-HASH cache entries require K, V and routing history")
        else:
            past_length = 0
            if use_cache and TRHashCache is not None:
                past_key_values = TRHashCache(config=self.config)
        if cache_position is None:
            cache_position = torch.arange(
                past_length,
                past_length + query_length,
                device=hidden_states.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
        key_length = past_length + query_length
        causal_mask = self._causal_padding_mask(
            attention_mask,
            batch_size,
            query_length,
            key_length,
            hidden_states.dtype,
            hidden_states.device,
        )

        all_hidden_states = [hidden_states] if output_hidden_states else None
        tuple_cache = [] if use_cache and not isinstance(past_key_values, Cache) else None
        for layer_index, layer in enumerate(self.layers):
            layer_past = (
                past_key_values
                if isinstance(past_key_values, Cache)
                else (past_key_values[layer_index] if past_key_values else None)
            )
            hidden_states, new_past = layer(
                hidden_states,
                token_ids,
                causal_mask,
                position_ids,
                layer_past,
                use_cache,
                cache_position,
            )
            if tuple_cache is not None:
                tuple_cache.append(new_past)
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
        hidden_states = self.norm(hidden_states)
        if all_hidden_states is not None:
            all_hidden_states[-1] = hidden_states

        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            selected = hidden_states[:, -logits_to_keep:, :]
        elif isinstance(logits_to_keep, torch.Tensor):
            selected = hidden_states[:, logits_to_keep, :]
        else:
            selected = hidden_states
        logits = F.linear(selected, self.embed_tokens.weight)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous().float()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if use_cache and isinstance(past_key_values, Cache):
            prefix = past_key_values.routing_token_ids
            past_key_values.routing_history = update_routing_history(token_ids, past_key_values.routing_history)
            past_key_values.routing_token_ids = token_ids.clone() if prefix is None else torch.cat((prefix, token_ids), dim=1)
        returned_cache = (past_key_values if isinstance(past_key_values, Cache) else tuple_cache) if use_cache else None
        if not return_dict:
            values = (
                logits,
                returned_cache,
                tuple(all_hidden_states) if all_hidden_states else None,
            )
            return ((loss,) + values) if loss is not None else values
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=returned_cache,
            hidden_states=tuple(all_hidden_states) if all_hidden_states else None,
        )


__all__ = ["TRHashForCausalLM", "TRHashConfig", "TRHashCache"]
