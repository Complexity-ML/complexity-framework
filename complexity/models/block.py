"""
Transformer Block - the basic building unit.

A block consists of:
1. Attention (with pre-norm)
2. MLP/FFN (with pre-norm)
3. Residual connections
4. Optional Mu-Guidance (contextual mu flows between layers)
"""

import torch
import torch.nn as nn
from typing import Any, Optional, Tuple

from ..config import ModelConfig
from ..core.attention import AttentionConfig
from ..core.mlp import MLPConfig
from ..core.registry import ATTENTION_REGISTRY, MLP_REGISTRY, NORMALIZATION_REGISTRY


class MuGuidance(nn.Module):
    """
    Mu-Guidance — contextual latent state flowing between layers.

    Produces mu_contextual = clamp(mu) + mu_proj(h) which guides
    the next layer's K, Q, V projections in attention.

    This is the lightweight Mu path used by the standard Transformer block.
    """

    def __init__(
        self,
        hidden_size: int,
        mu_min: float = 0.0,
        mu_max: float = 2.0,
        clamp_contextual: bool = False,
        context_min: float = -2.0,
        context_max: float = 2.0,
        use_mu_norm: bool = False,
        alpha_init: float = 1.0,
        mu_init_value: Optional[float] = None,
    ):
        super().__init__()
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.clamp_contextual = clamp_contextual
        self.context_min = context_min
        self.context_max = context_max
        self.mu_alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        if mu_init_value is None:
            mu_init_value = (mu_min + mu_max) / 2
        self.mu = nn.Parameter(torch.full((hidden_size,), float(mu_init_value)))
        self.mu_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mu_norm = None
        if use_mu_norm:
            from ..core.normalization.norms import RMSNorm
            self.mu_norm = RMSNorm(hidden_size)
        nn.init.zeros_(self.mu_proj.weight)  # Start neutral

    @property
    def mu_clamped(self) -> torch.Tensor:
        return torch.clamp(self.mu, self.mu_min, self.mu_max)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Returns mu_contextual: [batch, seq_len, hidden_size]."""
        mu_delta = self.mu_proj(hidden_states)
        if self.mu_norm is not None:
            mu_delta = self.mu_norm(mu_delta)
        mu_contextual = self.mu_clamped + self.mu_alpha * mu_delta
        if self.clamp_contextual:
            mu_contextual = torch.clamp(mu_contextual, self.context_min, self.context_max)
        return mu_contextual


class TransformerBlock(nn.Module):
    """
    Single Transformer block with configurable components.

    Architecture (Pre-Norm):
        x = x + attention(norm1(x))
        x = x + mlp(norm2(x))
    """

    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Pre-attention normalization
        self.input_layernorm = NORMALIZATION_REGISTRY.build(
            config.norm_type,
            config.hidden_size,
            eps=config.norm_eps,
        )

        # Attention
        attn_config = AttentionConfig(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            attention_dropout=config.attention_dropout,
            use_qk_norm=config.use_qk_norm,
            sliding_window=config.sliding_window,
            use_sdpa=config.use_sdpa,
            rope_type=config.rope_type,
            use_mup_attn_scale=getattr(config, "use_mup_attn_scale", False),
            use_mu_guidance=config.effective_mu_guidance,
            causal_conv_kernel_size=getattr(config, "causal_conv_kernel_size", 4),
            causal_conv_dilation=2 ** (
                layer_idx % int(getattr(config, "causal_conv_dilation_cycle", 8))
            ),
            causal_state_rank=getattr(config, "causal_state_rank", 16),
            causal_context_gate_init=getattr(
                config, "causal_context_gate_init", 1.0
            ),
            causal_contextual_mix_init=getattr(
                config, "causal_contextual_mix_init", 0.0
            ),
            causal_context_fusion_size=getattr(
                config, "causal_context_fusion_size", 0
            ),
            causal_stable_delta=getattr(config, "causal_stable_delta", False),
            causal_delta_chunk_size=getattr(
                config, "causal_delta_chunk_size", 512
            ),
            causal_delta_timescales=getattr(
                config, "causal_delta_timescales", 1
            ),
            causal_delta_collision_normalized=getattr(
                config, "causal_delta_collision_normalized", False
            ),
            causal_delta_lexical_values=getattr(
                config, "causal_delta_lexical_values", False
            ),
            causal_delta_lexical_forge=getattr(
                config, "causal_delta_lexical_forge", False
            ),
            causal_delta_occurrence_address=getattr(
                config, "causal_delta_occurrence_address", False
            ),

            lexical_object_rank=config.lexical_object_rank,
            disable_lexical_wrv_residual=getattr(
                config, "disable_lexical_wrv_residual", False
            ),
            disable_lexical_wrv_norms=getattr(
                config, "disable_lexical_wrv_norms", False
            ),
            lexical_wrv_hybrid=getattr(config, "lexical_wrv_hybrid", False),
            lexical_wrv_gate_init=getattr(config, "lexical_wrv_gate_init", 0.0),
            lexical_gqa_rank=getattr(config, "lexical_gqa_rank", 16),
            lexical_gqa_gate_init=getattr(config, "lexical_gqa_gate_init", 0.0),
            vocab_size=config.vocab_size,
            layer_idx=layer_idx,
        )
        self.self_attn = ATTENTION_REGISTRY.build(config.attention_type, attn_config)

        # Post-attention normalization
        self.post_attention_layernorm = NORMALIZATION_REGISTRY.build(
            config.norm_type,
            config.hidden_size,
            eps=config.norm_eps,
        )

        # MLP
        mlp_config = MLPConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            num_experts=config.num_experts,
            vocab_size=config.vocab_size,
            routing_strategy=getattr(config, 'routing_strategy', 'zipf'),
            token_frequencies=config.token_frequencies,
            lsh_routing=getattr(config, 'lsh_routing', False),
            lsh_bits=getattr(config, 'lsh_bits', 0),
            lsh_from_layer=getattr(config, 'lsh_from_layer', 0),
            lsh_threshold_mode=getattr(config, 'lsh_threshold_mode', 'zero'),
            shared_expert=getattr(config, 'shared_expert', False),
            shared_intermediate_size=getattr(config, 'shared_intermediate_size', None),
            shared_expert_chunk_tokens=getattr(config, 'shared_expert_chunk_tokens', 0),
            use_shared_routed_gates=getattr(config, 'use_shared_routed_gates', False),
            shared_gate_init=getattr(config, 'shared_gate_init', 1.0),
            routed_gate_init=getattr(config, 'routed_gate_init', 1.0),
            top_k=getattr(config, 'top_k', 1),
            top_k_primary_weight=getattr(config, 'top_k_primary_weight', None),
            layer_idx=layer_idx,
            static_expert_capacity=getattr(config, 'static_expert_capacity', False),
            collect_moe_telemetry=getattr(config, 'collect_moe_telemetry', False),
            use_custom_kernels=getattr(config, 'use_custom_kernels', 'auto'),
            use_cggr=getattr(config, 'use_cggr', False),
            lexical_object_rank=getattr(config, 'lexical_object_rank', 16),
            lexical_object_gate_init=getattr(config, 'lexical_object_gate_init', 0.1),
            micro_num_experts=getattr(config, 'micro_num_experts', 4),
            micro_expert_width=getattr(config, 'micro_expert_width', 16),
            micro_expert_gate_init=getattr(config, 'micro_expert_gate_init', 0.1),
        )
        self.mlp = MLP_REGISTRY.build(config.mlp_type, mlp_config)

        # Mu-Guidance (optional — contextual mu flowing between layers)
        self.use_mu_guidance = config.effective_mu_guidance
        if self.use_mu_guidance:
            self.mu_guidance = MuGuidance(
                hidden_size=config.hidden_size,
                mu_min=getattr(config, 'mu_min', 0.0),
                mu_max=getattr(config, 'mu_max', 2.0),
                clamp_contextual=getattr(config, 'clamp_mu_contextual', False),
                context_min=getattr(config, 'mu_context_min', -2.0),
                context_max=getattr(config, 'mu_context_max', 2.0),
                use_mu_norm=getattr(config, 'use_mu_norm', False),
                alpha_init=getattr(config, 'mu_alpha_init', 1.0),
                mu_init_value=getattr(config, 'mu_init_value', None),
            )
        else:
            self.mu_guidance = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        use_cache: bool = False,
        token_ids: Optional[torch.Tensor] = None,
        velocity_state: Optional[torch.Tensor] = None,
        mu_prev: Optional[torch.Tensor] = None,
        sort_idx: Optional[torch.Tensor] = None,
        lexical_token_scale_values: Optional[torch.Tensor] = None,
        lexical_base_writes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Any], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through the transformer block.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            past_key_value: Optional KV cache
            use_cache: Whether to return updated KV cache
            token_ids: Optional token IDs for MoE routing
            velocity_state: Unused (kept for backward compat)
            mu_prev: Optional mu from previous layer (for mu-guided attention)
            sort_idx: Unused (sort_idx computed internally by token_routed)

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
            past_key_value: Optional updated KV cache
            velocity_state: None (kept for backward compat)
            mu_contextual: Optional mu for next layer guidance
        """
        residual = hidden_states

        # Self Attention
        hidden_states = self.input_layernorm(hidden_states)

        attn_kwargs = dict(
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        if self.config.attention_type in {
            "causal_fast_weight_conv",
            "lexical_gqa",
            "lexical_bias_gqa",
            "lexical_wrv",
        }:
            attn_kwargs["token_ids"] = token_ids
        if self.config.attention_type in {
            "causal_fast_weight_conv",
            "lexical_wrv",
        }:
            attn_kwargs["lexical_token_scale_values"] = lexical_token_scale_values
            attn_kwargs["lexical_base_writes"] = lexical_base_writes
        elif self.config.attention_type in {"lexical_gqa", "lexical_bias_gqa"}:
            attn_kwargs["lexical_scale"] = lexical_token_scale_values
        if mu_prev is not None:
            attn_kwargs["mu_prev"] = mu_prev
        hidden_states, new_kv = self.self_attn(hidden_states, **attn_kwargs)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(
            hidden_states,
            token_ids=token_ids,
            lexical_token_scale_values=lexical_token_scale_values,
        )
        hidden_states = residual + hidden_states

        # Mu-Guidance AFTER MLP — captures expert-specific information
        # so next layer's attention knows which expert processed each token
        mu_contextual = None
        if self.mu_guidance is not None:
            mu_contextual = self.mu_guidance(hidden_states)

        return hidden_states, new_kv, None, mu_contextual
