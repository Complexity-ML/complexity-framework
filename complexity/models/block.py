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
from typing import Optional, Tuple

from ..config import ModelConfig
from ..core.attention import AttentionConfig
from ..core.mlp import MLPConfig
from ..core.registry import ATTENTION_REGISTRY, MLP_REGISTRY, NORMALIZATION_REGISTRY


class MuGuidance(nn.Module):
    """
    Mu-Guidance — contextual latent state flowing between layers.

    Produces mu_contextual = clamp(mu) + mu_proj(h) which guides
    the next layer's K, Q, V projections in attention.

    Extracted from INLDynamics to keep Mu without the PiD controller.
    """

    def __init__(self, hidden_size: int, mu_min: float = 0.0, mu_max: float = 2.0):
        super().__init__()
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.mu = nn.Parameter(torch.full((hidden_size,), (mu_min + mu_max) / 2))
        self.mu_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.mu_proj.weight)  # Start neutral

    @property
    def mu_clamped(self) -> torch.Tensor:
        return torch.clamp(self.mu, self.mu_min, self.mu_max)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Returns mu_contextual: [batch, seq_len, hidden_size]."""
        return self.mu_clamped + self.mu_proj(hidden_states)


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
            token_frequencies=config.token_frequencies,
            shared_expert=getattr(config, 'shared_expert', False),
            shared_intermediate_size=getattr(config, 'shared_intermediate_size', None),
            routed_gate=getattr(config, 'routed_gate', False),
            routed_gate_init=getattr(config, 'routed_gate_init', 0.0),
            gpt2_residual_init=getattr(config, 'gpt2_residual_init', False),
            num_hidden_layers=getattr(config, 'num_hidden_layers', 1),
        )
        self.mlp = MLP_REGISTRY.build(config.mlp_type, mlp_config)

        # Mu-Guidance (optional — contextual mu flowing between layers)
        # Activated by use_mu_guidance OR use_mu_projection (unified under MuGuidance)
        self.use_mu_guidance = (
            getattr(config, 'use_mu_guidance', False) or getattr(config, 'use_mu_projection', False)
        ) and not getattr(config, 'disable_mu_guidance', False)
        if self.use_mu_guidance:
            self.mu_guidance = MuGuidance(hidden_size=config.hidden_size)
        else:
            self.mu_guidance = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        token_ids: Optional[torch.Tensor] = None,
        velocity_state: Optional[torch.Tensor] = None,
        mu_prev: Optional[torch.Tensor] = None,
        sort_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor], Optional[torch.Tensor]]:
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
        if mu_prev is not None:
            attn_kwargs["mu_prev"] = mu_prev
        hidden_states, new_kv = self.self_attn(hidden_states, **attn_kwargs)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, token_ids=token_ids)
        hidden_states = residual + hidden_states

        # Mu-Guidance AFTER MLP — captures expert-specific information
        # so next layer's attention knows which expert processed each token
        mu_contextual = None
        if self.mu_guidance is not None:
            mu_contextual = self.mu_guidance(hidden_states)

        return hidden_states, new_kv, None, mu_contextual
