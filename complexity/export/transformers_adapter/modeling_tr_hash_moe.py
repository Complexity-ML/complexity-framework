"""Self-contained Transformers implementation of TR-HASH MoE.

The module preserves the native checkpoint tensor names. Its universal
PyTorch expert path is the numerical reference; optimized serving remains the
job of TR-Hash-i64 and, once supported, vLLM.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_tr_hash_moe import TRHashConfig

try:
    from transformers.cache_utils import Cache, DynamicCache
except ImportError:  # Transformers 4.30-4.35: legacy tuple caches only.

    class Cache:  # type: ignore[no-redef]
        pass

    DynamicCache = None  # type: ignore[assignment,misc]


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
    ) -> tuple[torch.Tensor, Any]:
        batch_size, sequence_length, _ = hidden_states.shape
        # Match complexity-framework's fused K/Q/V projection order exactly.
        projection = F.linear(
            hidden_states,
            torch.cat(
                (self.k_proj.weight, self.q_proj.weight, self.v_proj.weight),
                dim=0,
            ),
        )
        kv_size = self.num_key_value_heads * self.head_dim
        key_states, query_states, value_states = projection.split(
            (kv_size, self.num_heads * self.head_dim, kv_size), dim=-1
        )
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
        attention_output = attention_output.transpose(1, 2).reshape(batch_size, sequence_length, -1)
        return self.o_proj(attention_output), new_past


class TRHashExpertEngine(nn.Module):
    """Shared SwiGLU plus fixed top-k experts with native tensor names."""

    def __init__(self, config: TRHashConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.vocab_size = config.vocab_size
        self.shared_output_scale = config.shared_output_scale
        self.routed_output_scale = config.routed_output_scale
        if self.top_k == 1:
            route_weights = (1.0,)
        else:
            primary = (
                1.0 / self.top_k
                if config.top_k_primary_weight is None
                else float(config.top_k_primary_weight)
            )
            route_weights = (
                primary,
                *((1.0 - primary) / (self.top_k - 1) for _ in range(self.top_k - 1)),
            )
        # Keep these as Python scalars. Transformers may construct the model
        # on the meta device while streaming a safetensors checkpoint; a
        # derived non-persistent tensor buffer can otherwise be materialized
        # as zeros because there is intentionally no checkpoint key for it.
        self.route_weights = tuple(float(value) for value in route_weights)
        # The checkpoint supplies these persisted routing artifacts. Zero
        # initialization prevents a second, subtly different hash builder from
        # ever being treated as authoritative.
        self.register_buffer(
            "route_table",
            torch.zeros(self.top_k, self.vocab_size, dtype=torch.long),
        )
        self.register_buffer(
            "fused_route_codes",
            torch.zeros(self.vocab_size, dtype=torch.uint8),
        )
        pair_count = self.num_experts * (self.num_experts - 1) // 2
        self.register_buffer(
            "fused_expert_pairs",
            torch.zeros(pair_count, 2, dtype=torch.int32),
        )
        self.expert_gate = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_size, config.expert_width)
        )
        self.expert_up = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_size, config.expert_width)
        )
        self.expert_down = nn.Parameter(
            torch.empty(config.num_experts, config.expert_width, config.hidden_size)
        )
        if config.shared_expert:
            self.shared_gate = nn.Linear(
                config.hidden_size, config.shared_intermediate_size, bias=False
            )
            self.shared_up = nn.Linear(
                config.hidden_size, config.shared_intermediate_size, bias=False
            )
            self.shared_down = nn.Linear(
                config.shared_intermediate_size, config.hidden_size, bias=False
            )
        else:
            self.shared_gate = self.shared_up = self.shared_down = None

    @property
    def experts(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expose the sparse expert block to serving/quantization adapters."""

        return self.expert_gate, self.expert_up, self.expert_down

    def forward(self, hidden_states: torch.Tensor, token_ids: torch.LongTensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        flat_states = hidden_states.reshape(-1, hidden_size)
        if self.shared_gate is None:
            shared = torch.zeros_like(flat_states)
        else:
            shared = self.shared_down(
                F.silu(self.shared_gate(flat_states)) * self.shared_up(flat_states)
            )
        routes = self.route_table[:, token_ids.clamp(0, self.vocab_size - 1)].reshape(
            self.top_k, -1
        )
        route_weights = flat_states.new_tensor(self.route_weights).view(-1, 1)
        routed = torch.zeros_like(flat_states)
        for expert_index in range(self.num_experts):
            token_weight = (routes.eq(expert_index).to(flat_states.dtype) * route_weights).sum(
                dim=0
            )
            active_states = flat_states * token_weight.ne(0).to(flat_states.dtype).unsqueeze(-1)
            intermediate = F.silu(active_states @ self.expert_gate[expert_index]) * (
                active_states @ self.expert_up[expert_index]
            )
            expert_output = intermediate @ self.expert_down[expert_index]
            routed.add_(expert_output * token_weight.unsqueeze(-1))
        output = self.shared_output_scale * shared + self.routed_output_scale * routed
        return output.view(batch_size, sequence_length, hidden_size)


class TRHashMLP(nn.Module):
    def __init__(self, config: TRHashConfig) -> None:
        super().__init__()
        self.engine = TRHashExpertEngine(config)

    @property
    def experts(self):
        return self.engine.experts

    def forward(self, hidden_states: torch.Tensor, token_ids: torch.LongTensor):
        return self.engine(hidden_states, token_ids)


class TRHashDecoderLayer(nn.Module):
    def __init__(self, config: TRHashConfig, layer_idx: int) -> None:
        super().__init__()
        self.input_layernorm = TRHashRMSNorm(config.hidden_size, config.norm_eps)
        self.self_attn = TRHashAttention(config, layer_idx)
        self.post_attention_layernorm = TRHashRMSNorm(config.hidden_size, config.norm_eps)
        self.mlp = TRHashMLP(config)

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
        residual = hidden_states
        attention_output, new_past = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask,
            position_ids,
            past_key_values,
            use_cache,
            cache_position,
        )
        hidden_states = residual + attention_output
        return (
            hidden_states + self.mlp(self.post_attention_layernorm(hidden_states), token_ids),
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
            has_cached_values = past_key_values.get_seq_length() > 0
        else:
            has_cached_values = past_key_values is not None
        if has_cached_values:
            input_ids = input_ids[:, -1:]
        position_ids = kwargs.get("position_ids")
        if position_ids is None and attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask.eq(0), 0)
        if position_ids is not None:
            position_ids = position_ids[:, -input_ids.shape[1] :]
        model_inputs: dict[str, Any]
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
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
        token_ids = input_ids
        if token_ids is None:
            token_ids = torch.zeros(
                batch_size, query_length, dtype=torch.long, device=hidden_states.device
            )

        if isinstance(past_key_values, Cache):
            past_length = past_key_values.get_seq_length()
        elif past_key_values:
            past_length = past_key_values[0][0].shape[2]
        else:
            past_length = 0
            if use_cache and past_key_values is None and DynamicCache is not None:
                past_key_values = DynamicCache(config=self.config)
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
        legacy_cache = [] if use_cache and not isinstance(past_key_values, Cache) else None
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
            if legacy_cache is not None:
                legacy_cache.append(new_past)
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

        returned_cache = past_key_values if isinstance(past_key_values, Cache) else legacy_cache
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


__all__ = ["TRHashForCausalLM", "TRHashConfig"]
