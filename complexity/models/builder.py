"""
Model Builder - constructs complete models from configuration.

This is the main entry point for creating models in the framework.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from ..config import ModelConfig, get_preset
from ..core.registry import NORMALIZATION_REGISTRY, register_model
from .block import TransformerBlock


def build_lexical_zipf_weights(
    counts: torch.Tensor,
    *,
    mode: str,
    alpha: float,
    floor: float,
    permutation_seed: int,
) -> torch.Tensor:
    """Build RMS-matched token weights from descending frequency ranks."""
    if counts.ndim != 1:
        raise ValueError("Zipf counts must be a 1D tensor")
    if mode not in {"ordered", "permuted"}:
        raise ValueError(f"Unsupported Zipf mode: {mode}")
    if alpha <= 0.0:
        raise ValueError("lexical_zipf_alpha must be positive")
    if not 0.0 <= floor <= 1.0:
        raise ValueError("lexical_zipf_floor must lie in [0, 1]")
    vocab_size = counts.numel()
    order = torch.argsort(counts.to(torch.float64), descending=True, stable=True)
    ranks = torch.empty(vocab_size, dtype=torch.float32)
    ranks[order] = torch.arange(1, vocab_size + 1, dtype=torch.float32)
    weights = (ranks / float(vocab_size)).pow(float(alpha)).clamp_min(float(floor))
    weights = weights / weights.square().mean().sqrt().clamp_min(1e-12)
    if mode == "permuted":
        generator = torch.Generator().manual_seed(int(permutation_seed))
        weights = weights[torch.randperm(vocab_size, generator=generator)]
    return weights


@register_model("complexity")
@register_model("decoder")
@register_model("causal_lm")
class ComplexityModel(nn.Module):
    """
    Complete Transformer model built from configuration.

    This model supports any architecture defined by ModelConfig:
    - Llama-style (GQA, SwiGLU, RMSNorm)
    - Mistral-style (sliding window)
    - GPT-style (MHA, GELU, LayerNorm)
    - Complexity custom (TR-Hash MoE)

    Usage:
        from complexity.config import ModelConfig
        from complexity.models import ComplexityModel

        # From config
        config = ModelConfig(hidden_size=768, num_hidden_layers=12)
        model = ComplexityModel(config)

        # From preset
        model = ComplexityModel.from_preset("llama-7b")

        # Forward pass
        logits = model(input_ids)

        # Generation/serving is delegated to vLLM or SGLang.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer layers
        context_layers = {0, config.num_hidden_layers // 2}
        self.layers = nn.ModuleList(
            [
                TransformerBlock(config, layer_idx=index)
                for index in range(config.num_hidden_layers)
            ]
        )
        if config.layer_mlp_controls is not None:
            for layer, control in zip(self.layers, config.layer_mlp_controls):
                layer.mlp.set_top_k_primary_weight(control["top_k_primary_weight"])
                layer.mlp.set_active_experts(control["active_num_experts"])
                layer.mlp.set_active_expert_width(control["active_expert_width"])
        if config.attention_type == "causal_fast_weight_conv":
            shared_context = getattr(self.layers[0].self_attn, "shared_context")
            for index, layer in enumerate(self.layers):
                setattr(layer.self_attn, "shared_context", shared_context)
                setattr(layer.self_attn, "context_enabled", index in context_layers)
        lexical_object_types = {
            "lexical_modulated",
            "lexical_object",
            "lexical_channel_modulated",
            "lexical_channel_object",
            "lexical_object_micro_expert",
        }
        if config.mlp_type in lexical_object_types and bool(
            getattr(config, "tie_lexical_object_embeddings", False)
        ):
            shared_token_scale = self.layers[0].mlp.token_scale
            self.lexical_token_scale = shared_token_scale
            for layer in self.layers:
                delattr(layer.mlp, "token_scale")

        zipf_mode = str(getattr(config, "lexical_zipf_mode", "uniform"))
        if zipf_mode != "uniform":
            zipf_path = getattr(config, "lexical_zipf_path", None)
            if not zipf_path:
                raise ValueError("lexical_zipf_path is required for non-uniform Zipf")
            artifact = torch.load(zipf_path, map_location="cpu", weights_only=True)
            counts = artifact["counts"] if isinstance(artifact, dict) else artifact
            if counts.numel() != config.vocab_size:
                raise ValueError(
                    f"Zipf counts have vocab {counts.numel()}, expected {config.vocab_size}"
                )
            self.register_buffer(
                "lexical_zipf_weights",
                build_lexical_zipf_weights(
                    counts,
                    mode=zipf_mode,
                    alpha=float(getattr(config, "lexical_zipf_alpha", 0.25)),
                    floor=float(getattr(config, "lexical_zipf_floor", 0.1)),
                    permutation_seed=int(
                        getattr(config, "lexical_zipf_permutation_seed", 1729)
                    ),
                ),
                persistent=True,
            )

        # Final normalization
        self.norm = NORMALIZATION_REGISTRY.build(
            config.norm_type,
            config.hidden_size,
            eps=config.norm_eps,
        )

        # Output projection (LM head)
        if config.tie_word_embeddings:
            self.lm_head = None  # Will use embed_tokens.weight
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Gradient checkpointing (disabled by default)
        self._gradient_checkpointing = False

        # Initialize weights (GPT-style: residual projections scaled by 1/√(2N))
        self.apply(self._init_weights)
        if config.mlp_type in lexical_object_types:
            if hasattr(self, "lexical_token_scale"):
                nn.init.zeros_(self.lexical_token_scale.weight)
            else:
                nn.init.zeros_(self.layers[0].mlp.token_scale.weight)
        self._init_residual_scaling()
        # μP: scale hidden→hidden Linears by 1/√(hidden_size / mup_base_width).
        # No-op at base width or when use_mup_init=False. Applied AFTER
        # _init_residual_scaling so the GPT-style 1/√(2N) factor on output
        # projections compounds with μP correctly: down/o_proj end up at
        # σ_base / (√(2N) · √width_mult).
        self._apply_mup_init_scaling()
        # Deterministic init pass: after the default GPT-style init runs,
        # walk the block list and re-initialise the three projections of any
        # DenseDeterministicMLP via a locally-seeded Generator. Preserves
        # both the residual scaling and the μP scaling (when enabled).
        self._init_dense_deterministic()

    def _init_weights(self, module: nn.Module):
        """
        GPT-style weight initialization.

        Standard layers: normal_(std=0.02)
        Residual projections (o_proj, down_proj_w): scaled by 1/√(2*num_layers)
        to prevent residual stream from growing with depth.
        Ref: Radford et al. (GPT-2), "Language Models are Unsupervised Multitask Learners"
        """
        std = self.config.initializer_range  # 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def _init_residual_scaling(self):
        """
        GPT-style residual scaling: init output projections with std=0.02/√(2N).

        Targets: o_proj (attention output) and down_proj/down_proj_w (MLP output).
        Re-initializes with normal_ then scales, overriding any prior init
        (e.g. kaiming_uniform_ in TokenRoutedMLP).
        Called once after self.apply(_init_weights).
        """
        n = self.config.num_hidden_layers
        residual_std = self.config.initializer_range / (2 * n) ** 0.5
        for layer in self.layers:
            # Attention output projection
            attn = layer.self_attn
            if hasattr(attn, 'o_proj') and isinstance(attn.o_proj, nn.Linear):
                nn.init.normal_(attn.o_proj.weight, mean=0.0, std=residual_std)
            output_proj = getattr(attn, 'output_proj', None)
            if isinstance(output_proj, nn.Linear):
                nn.init.normal_(output_proj.weight, mean=0.0, std=residual_std)
            # MLP down projection (nn.Linear or nn.Parameter) — routed experts and dense SwiGLU
            mlp = layer.mlp
            if hasattr(mlp, 'down_proj'):
                if isinstance(mlp.down_proj, nn.Linear):
                    nn.init.normal_(mlp.down_proj.weight, mean=0.0, std=residual_std)
                elif isinstance(mlp.down_proj, nn.Parameter):
                    nn.init.normal_(mlp.down_proj, mean=0.0, std=residual_std)
            if hasattr(mlp, 'down_proj_w') and isinstance(mlp.down_proj_w, nn.Parameter):
                nn.init.normal_(mlp.down_proj_w, mean=0.0, std=residual_std)
            # Shared expert down projection (TokenRoutedMLP with shared=True)
            if hasattr(mlp, 'shared_down') and isinstance(mlp.shared_down, nn.Linear):
                nn.init.normal_(mlp.shared_down.weight, mean=0.0, std=residual_std)
            # Canonical TRHashEngine is nested by the model-block adapter.
            tr_hash_engine = getattr(mlp, "engine", None)
            if tr_hash_engine is not None:
                expert_down = getattr(tr_hash_engine, "expert_down", None)
                if isinstance(expert_down, nn.Parameter):
                    nn.init.normal_(expert_down, mean=0.0, std=residual_std)
                shared_down = getattr(tr_hash_engine, "shared_down", None)
                if isinstance(shared_down, nn.Linear):
                    nn.init.normal_(shared_down.weight, mean=0.0, std=residual_std)
            if hasattr(mlp, 'object_down') and isinstance(mlp.object_down, nn.Linear):
                nn.init.normal_(mlp.object_down.weight, mean=0.0, std=residual_std)
            if hasattr(mlp, 'micro_down') and isinstance(mlp.micro_down, nn.Parameter):
                nn.init.normal_(mlp.micro_down, mean=0.0, std=residual_std)

    def _apply_mup_init_scaling(self):
        """
        μP init scaling: multiply every Linear weight that is *not* an
        embedding/output by 1/√width_mult, where
        width_mult = hidden_size / mup_base_width.

        Skipped when ``use_mup_init`` is False or width_mult ≤ 1.
        Embeddings keep their base std (μP convention: σ_emb is
        width-independent). The lm_head — when untied — *is* scaled here:
        the output side of μP relies on the ``use_mup_output_mult`` flag
        which divides the logits by width_mult at forward time, NOT on a
        differentiated init.
        """
        if not getattr(self.config, "use_mup_init", False):
            return
        width_mult = float(self.config.hidden_size) / float(self.config.mup_base_width)
        if width_mult <= 1.0:
            return
        scale = 1.0 / (width_mult ** 0.5)
        # Walk all Linears, skip embeddings (named with 'embed_tokens').
        # The lm_head, when not tied, is scaled like any other Linear.
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                module.weight.data.mul_(scale)

    def _init_dense_deterministic(self):
        """
        Deterministic, RNG-free initialisation pass.

        Activates when at least one block uses ``DenseDeterministicMLP``.
        In that case, every weight matrix that would otherwise consume
        PRNG state is re-initialised via ``deterministic_gaussian_init_``
        (a locally-seeded Generator), keyed on layer index + matrix
        role:

            - token embedding table,
            - attention Q, K, V, O in every block,
            - FFN gate, up, down in every block.

        Attention ``o_proj`` and FFN ``down_proj`` keep the GPT-style
        residual scaling std = σ / √(2N) so the residual-stream variance
        remains bounded with depth, matching the policy of
        ``_init_residual_scaling``.

        After this pass the model is a pure function of its config: two
        instantiations produce bit-identical weights, no seed consulted.

        Called *after* ``_init_weights`` and ``_init_residual_scaling``
        so the deterministic state is final.
        """
        try:
            from ..core.mlp.dense_deterministic import DenseDeterministicMLP
            from ..core.mlp.deterministic_init import deterministic_gaussian_init_
        except ImportError:
            return

        uses_deterministic = any(
            isinstance(getattr(layer, "mlp", None), DenseDeterministicMLP)
            for layer in self.layers
        )
        if not uses_deterministic:
            return

        std = float(getattr(self.config, "initializer_range", 0.02))
        n_layers = len(self.layers)
        # μP: hidden→hidden std scaled by 1/√width_mult. Embeddings keep
        # the base std regardless (μP convention).
        if getattr(self.config, "use_mup_init", False):
            width_mult = float(self.config.hidden_size) / float(self.config.mup_base_width)
            mup_scale = 1.0 / (width_mult ** 0.5) if width_mult > 1.0 else 1.0
        else:
            mup_scale = 1.0
        hidden_std = std * mup_scale
        res_std = hidden_std / (2 * n_layers) ** 0.5

        # Embedding table — base std (no μP scaling).
        emb = getattr(self, "embed_tokens", None)
        if isinstance(emb, nn.Embedding):
            deterministic_gaussian_init_(emb.weight, key=7_000_001, std=std)

        # Attention + FFN in every block. Each matrix gets a unique key so
        # its Gaussian sample is independent of every other matrix.
        for b, layer in enumerate(self.layers):
            base = b * 10 + 1_000_000

            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                # q_proj / k_proj / v_proj / o_proj have fixed offsets 1-4.
                # o_proj gets the residual-scaled std; the rest get the
                # μP-scaled hidden_std.
                known = {"q_proj": (1, hidden_std), "k_proj": (2, hidden_std),
                         "v_proj": (3, hidden_std), "o_proj": (4, res_std)}
                for name, (off, s) in known.items():
                    lin = getattr(attn, name, None)
                    if isinstance(lin, nn.Linear):
                        deterministic_gaussian_init_(lin.weight, key=base + off, std=s)
                        if lin.bias is not None:
                            nn.init.zeros_(lin.bias)
                # Any extra Linear inside attention (e.g. mu_to_q / mu_to_k /
                # mu_to_v for mu-guidance variants) gets a distinct key too.
                extra_off = 20
                for child_name, child in attn.named_children():
                    if child_name in known or not isinstance(child, nn.Linear):
                        continue
                    deterministic_gaussian_init_(child.weight, key=base + extra_off, std=hidden_std)
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)
                    extra_off += 1

            mlp = getattr(layer, "mlp", None)
            if isinstance(mlp, DenseDeterministicMLP):
                for name, off in (("gate_proj", 5), ("up_proj", 6),
                                  ("down_proj", 7)):
                    lin = getattr(mlp, name, None)
                    if not isinstance(lin, nn.Linear):
                        continue
                    s = res_std if name == "down_proj" else hidden_std
                    deterministic_gaussian_init_(lin.weight, key=base + off, std=s)
                    if lin.bias is not None:
                        nn.init.zeros_(lin.bias)

    def gradient_checkpointing_enable(self):
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self._gradient_checkpointing = False

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        self.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        if self.lm_head is not None:
            return self.lm_head
        return self.embed_tokens  # Tied weights

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Any]] = None,
        use_cache: bool = False,
        return_hidden_states: bool = False,
        return_logits: bool = True,
        inputs_embeds: Optional[torch.Tensor] = None,
        routing_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass through the model.

        Args:
            input_ids: [batch, seq_len] token IDs. Required unless
                ``inputs_embeds`` is supplied.
            attention_mask: Optional attention mask
            past_key_values: Optional list of KV caches per layer
            use_cache: Whether to return updated KV caches
            return_hidden_states: Whether to return all hidden states
            return_logits: Whether to compute LM logits. Training loops using
                fused cross-entropy can set this False and consume
                last_hidden_state directly.
            inputs_embeds: Optional externally prepared embeddings. This is the
                multimodal prefix path; it bypasses ``embed_tokens``.
            routing_ids: Token IDs used only for deterministic routing and
                lexical features when ``inputs_embeds`` is supplied.

        Returns:
            Dictionary with:
                - logits: [batch, seq_len, vocab_size]
                - past_key_values: Optional list of KV caches
                - hidden_states: Optional list of hidden states
        """
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds must be provided")
            hidden_states = self.embed_tokens(input_ids)
        else:
            if inputs_embeds.ndim != 3:
                raise ValueError("inputs_embeds must have shape [batch, seq, hidden]")
            if inputs_embeds.size(-1) != self.config.hidden_size:
                raise ValueError(
                    "inputs_embeds hidden size does not match the model configuration"
                )
            hidden_states = inputs_embeds

        batch_size, seq_len = hidden_states.shape[:2]
        token_ids = routing_ids if routing_ids is not None else input_ids
        if token_ids is None:
            token_ids = torch.zeros(
                batch_size,
                seq_len,
                dtype=torch.long,
                device=hidden_states.device,
            )
        if token_ids.shape != (batch_size, seq_len):
            raise ValueError("routing_ids must match the embedded sequence shape")

        # Store hidden states if requested
        all_hidden_states = [hidden_states] if return_hidden_states else None

        # Initialize per-layer cache/state list. Attention layers return KV
        # tuples; attention-free mixers return fixed-size state tensors.
        new_past_key_values = [] if use_cache else None

        # Process through layers.
        lexical_token_scale_values = (
            self.lexical_token_scale(token_ids)
            if hasattr(self, "lexical_token_scale")
            else None
        )
        lexical_zipf_values = (
            self.lexical_zipf_weights[token_ids]
            if hasattr(self, "lexical_zipf_weights")
            else None
        )
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None

            if self._gradient_checkpointing and self.training:
                hidden_states, new_kv, _ = activation_checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    past_kv,
                    use_cache,
                    token_ids,
                    None,  # velocity_state (unused, kept for compat)
                    None,  # sort_idx (computed internally by token_routed)
                    lexical_token_scale_values,
                    lexical_zipf_values,
                    use_reentrant=False,
                )
            else:
                hidden_states, new_kv, _ = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_kv,
                    use_cache=use_cache,
                    token_ids=token_ids,
                    lexical_token_scale_values=lexical_token_scale_values,
                    lexical_zipf_values=lexical_zipf_values,
                )

            if use_cache:
                new_past_key_values.append(new_kv)

            if return_hidden_states:
                all_hidden_states.append(hidden_states)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        # Fused-cross-entropy training paths consume last_hidden_state directly
        # and can skip the large vocab projection for memory/speed.
        if not return_logits:
            return {
                "logits": None,
                "past_key_values": new_past_key_values,
                "hidden_states": all_hidden_states,
                "last_hidden_state": hidden_states,
            }

        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            logits = F.linear(hidden_states, self.embed_tokens.weight)

        # μP output multiplier: divide logits by width_mult so the output
        # scale matches the proxy width regardless of model width.
        if getattr(self.config, "use_mup_output_mult", False):
            width_mult = float(self.config.hidden_size) / float(self.config.mup_base_width)
            if width_mult > 1.0:
                logits = logits / width_mult

        return {
            "logits": logits,
            "past_key_values": new_past_key_values,
            "hidden_states": all_hidden_states,
            "last_hidden_state": hidden_states,
        }

    def set_tokenizer(self, tokenizer):
        """Set tokenizer for callers that need tokenization metadata."""
        self.tokenizer = tokenizer

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        text: Optional[str] = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        return_text: bool = False,
    ) -> Union[torch.Tensor, str]:
        """
        Text generation is intentionally not implemented as a native PyTorch
        loop on ``ComplexityModel``.

        Complexity's PyTorch layer is for model definition, forward passes,
        training, and checkpoint export. Serving/generation must go through an
        external inference runtime such as vLLM or SGLang, using the OpenAI-
        compatible helpers in ``complexity.inference.external`` or the CLI.
        """
        raise RuntimeError(
            "ComplexityModel.generate is disabled by architecture contract: "
            "use vLLM or SGLang for generation/serving. Export the checkpoint "
            "and call a vLLM/SGLang OpenAI-compatible endpoint instead."
        )

    @classmethod
    def from_preset(cls, name: str) -> "ComplexityModel":
        """Create model from preset configuration."""
        config = get_preset(name)
        return cls(config)

    @classmethod
    def from_config(cls, config_path: str) -> "ComplexityModel":
        """Create model from config file."""
        config = ModelConfig.load(config_path)
        return cls(config)

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def save_pretrained(self, save_directory: Union[str, Path], safe_serialization: bool = True):
        """Save complete native weights and config, writing only on rank zero.

        Replicated models can be exported by rank zero alone. DTensor exports
        require every rank to participate in ``full_tensor``; a failed gather
        never falls back to saving an incomplete local shard.
        """
        import tempfile

        import torch.distributed as dist

        distributed = dist.is_initialized() and dist.get_world_size() > 1
        is_main = not distributed or dist.get_rank() == 0
        raw_state = self.state_dict()
        collective = any(hasattr(value, "full_tensor") for value in raw_state.values())
        if not is_main and not collective:
            return
        cpu_state, errors = {}, []
        for name, value in raw_state.items():
            try:
                if hasattr(value, "full_tensor"):
                    value = value.full_tensor()
                elif hasattr(value, "to_local"):
                    value = value.to_local()
                if is_main:
                    if not isinstance(value, torch.Tensor):
                        raise TypeError("native model exports require tensor-only state")
                    cpu_state[name] = value.detach().cpu().contiguous()
            except Exception as error:
                errors.append(f"{name}: {type(error).__name__}: {error}")
        if collective and distributed:
            rank_errors = [None] * dist.get_world_size()
            dist.all_gather_object(rank_errors, errors)
            errors = [error for group in rank_errors for error in group]
        if errors:
            raise RuntimeError("Cannot export complete model weights: " + "; ".join(errors))
        if not is_main:
            return

        if safe_serialization:
            from safetensors.torch import save_file
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        weights_path = save_directory / (
            "model.safetensors" if safe_serialization else "pytorch_model.bin"
        )
        config_path = save_directory / "config.json"
        pending, backups, published, recovery_files = [], {}, [], set()
        try:
            with tempfile.NamedTemporaryFile(dir=save_directory, suffix=".tmp", delete=False) as f:
                weight_temp = Path(f.name)
            pending.append(weight_temp)
            if safe_serialization:
                save_file(cpu_state, str(weight_temp))
            else:
                torch.save(cpu_state, weight_temp)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=save_directory, suffix=".tmp", delete=False
            ) as f:
                config_temp = Path(f.name)
                pending.append(config_temp)
                json.dump(self.export_config(), f, indent=2)
                f.write("\n")
            # Preserve both formats and config until the complete replacement
            # succeeds. A Python-level commit error restores the old bundle.
            for old in (config_path, save_directory / "model.safetensors", save_directory / "pytorch_model.bin"):
                if old.is_file():
                    with tempfile.NamedTemporaryFile(dir=save_directory, suffix=".backup", delete=False) as f:
                        backup = Path(f.name)
                    pending.append(backup)
                    old.replace(backup)
                    backups[old] = backup
            weight_temp.replace(weights_path)
            published.append(weights_path)
            config_temp.replace(config_path)
            published.append(config_path)
        except Exception:
            try:
                for path in published:
                    path.unlink(missing_ok=True)
                for original, backup in backups.items():
                    backup.replace(original)
            except OSError as error:
                recovery_files.update(path for path in backups.values() if path.exists())
                raise RuntimeError(
                    "Native export rollback failed; preserved original files at: "
                    + ", ".join(map(str, sorted(recovery_files)))
                ) from error
            raise
        finally:
            for path in pending:
                if path not in recovery_files:
                    path.unlink(missing_ok=True)
        print(f"Model saved to {save_directory}")

    def validate_export_controls(self, raw_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate saved controls against the allocated architecture, without mutation."""
        restored = ModelConfig.from_dict(raw_config)
        if restored.mlp_type not in {"tr_hash_engine", "tr_hash_moe"}:
            raise ValueError("Runtime MLP controls require a TR-Hash engine model")
        shape_fields = (
            "hidden_size", "vocab_size", "num_hidden_layers", "intermediate_size",
            "num_experts", "top_k", "routing_strategy", "route_hash_count",
        )
        if self.config.mlp_type not in {"tr_hash_engine", "tr_hash_moe"} or any(
            getattr(restored, name) != getattr(self.config, name) for name in shape_fields
        ):
            raise ValueError("Runtime MLP control architecture differs from the model")
        controls = restored.layer_mlp_controls or [
            {
                "top_k_primary_weight": 1.0 if restored.top_k == 1 else (
                    1 / restored.top_k if restored.top_k_primary_weight is None
                    else restored.top_k_primary_weight
                ),
                "active_num_experts": restored.active_num_experts if restored.active_num_experts is not None else restored.num_experts,
                "active_expert_width": restored.active_expert_width if restored.active_expert_width is not None else restored.intermediate_size // restored.num_experts,
            }
            for _ in self.layers
        ]
        # Validate the complete list before changing any live layer.
        ModelConfig.from_dict({**restored.to_dict(), "layer_mlp_controls": controls}, strict=True)
        return controls

    def apply_export_controls(self, raw_config: Dict[str, Any]) -> None:
        """Restore validated non-parameter MLP controls for this architecture."""
        controls = self.validate_export_controls(raw_config)
        for layer, control in zip(self.layers, controls):
            layer.mlp.set_top_k_primary_weight(control["top_k_primary_weight"])
            layer.mlp.set_active_experts(control["active_num_experts"])
            layer.mlp.set_active_expert_width(control["active_expert_width"])

    def export_config(self) -> Dict[str, Any]:
        """Describe the live model's execution controls without mutating its config."""
        config = self.config.to_dict()
        if self.config.mlp_type not in {"tr_hash_engine", "tr_hash_moe"}:
            return config
        controls = [
            {
                "top_k_primary_weight": layer.mlp.training_telemetry()["topk_w"],
                "active_num_experts": layer.mlp.engine.active_num_experts,
                "active_expert_width": layer.mlp.engine.active_expert_width,
            }
            for layer in self.layers
        ]
        if all(control == controls[0] for control in controls):
            control = controls[0]
            defaults = {
                "top_k_primary_weight": 1 / self.config.top_k,
                "active_num_experts": self.config.num_experts,
                "active_expert_width": self.config.intermediate_size // self.config.num_experts,
            }
            for name, value in control.items():
                configured = config.get(name)
                if value != (defaults[name] if configured is None else configured):
                    config[name] = value
            config.pop("layer_mlp_controls", None)
        else:
            config["layer_mlp_controls"] = controls
        return config

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Union[str, Path],
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "ComplexityModel":
        """
        Load model from pretrained directory.

        Args:
            pretrained_model_path: Path to saved model directory
            device: Device to load model on (default: cpu)
            dtype: Data type for model (default: float32)

        Returns:
            Loaded ComplexityModel instance
        """
        pretrained_model_path = Path(pretrained_model_path)

        # Load config
        config_path = pretrained_model_path / "config.json"
        if not config_path.exists():
            raise ValueError(f"Config not found at {config_path}")

        with open(config_path, "r") as f:
            config_dict = json.load(f)

        config = ModelConfig.from_dict(config_dict)

        # Create model
        model = cls(config)

        model._load_pretrained_weights(pretrained_model_path)

        # Move to device/dtype if specified
        if device is not None:
            model = model.to(device)
        if dtype is not None:
            model = model.to(dtype)

        return model

    def _load_pretrained_weights(self, path: Union[str, Path]) -> None:
        """Load native exports or historical model.pt weights with one policy.

        This is shared with the convenience API. No missing learned parameter
        is accepted; omitted deterministic buffers are reconstructed by config.
        """
        path = Path(path)
        if path.is_dir():
            candidates = [
                path / name for name in ("model.safetensors", "pytorch_model.bin", "model.pt")
            ]
            weights_path = next((candidate for candidate in candidates if candidate.is_file()), None)
            if weights_path is None:
                raise FileNotFoundError(
                    f"No model weights found in {path}: expected "
                    "model.safetensors, pytorch_model.bin or model.pt"
                )
        else:
            weights_path = path
            if not weights_path.is_file():
                raise FileNotFoundError(f"Model weights not found: {weights_path}")

        if weights_path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
            except ImportError as exc:
                raise ImportError(
                    "safetensors is required to load .safetensors files. "
                    "Install with: pip install safetensors"
                ) from exc
            state_dict = load_file(str(weights_path))
        else:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

        # strict=False: deterministic buffers (RoPE inv_freq, TokenRoutedMLP's
        # token_to_expert/pair_hash_* routing tables) are pure functions of
        # config and are already correct on the freshly constructed model —
        # some checkpoint export paths omit them. Any missing key that isn't
        # one of those is a real problem, so it's still surfaced loudly.
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        deterministic_buffer = re.compile(
            r"\.(inv_freq|token_to_expert|pair_hash_route_codes|pair_hash_expert_pairs)$"
        )
        unexplained_missing = [k for k in missing if not deterministic_buffer.search(k)]
        if unexplained_missing or unexpected:
            raise RuntimeError(
                "Error(s) in loading state_dict for ComplexityModel: "
                f"missing={unexplained_missing}, unexpected={unexpected}"
            )

    def __repr__(self) -> str:
        params = self.num_parameters() / 1e6
        return f"ComplexityModel({params:.1f}M params, {self.config.num_hidden_layers} layers)"
