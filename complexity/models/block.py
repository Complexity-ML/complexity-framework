"""
Transformer Block - the basic building unit.

A block consists of:
1. Attention (with pre-norm)
2. MLP/FFN (with pre-norm)
3. Residual connections
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from ..config import ModelConfig
from ..core.attention import AttentionConfig
from ..core.mlp import MLPConfig
from ..core.registry import ATTENTION_REGISTRY, MLP_REGISTRY, NORMALIZATION_REGISTRY
from ..core.dynamics import INLDynamics


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
        )
        self.mlp = MLP_REGISTRY.build(config.mlp_type, mlp_config)

        # INL Dynamics (optional - Complexity innovation for stability)
        self.use_inl_dynamics = config.use_inl_dynamics
        if self.use_inl_dynamics:
            self.dynamics = INLDynamics(
                hidden_size=config.hidden_size,
                beta_max=config.inl_beta_max,
                velocity_max=config.inl_velocity_max,
            )
        else:
            self.dynamics = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        token_ids: Optional[torch.Tensor] = None,
        velocity_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """
        Forward pass through the transformer block.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            past_key_value: Optional KV cache
            use_cache: Whether to return updated KV cache
            token_ids: Optional token IDs for MoE routing
            velocity_state: Optional velocity state for INL Dynamics

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
            past_key_value: Optional updated KV cache
            velocity_state: Optional updated velocity state
        """
        residual = hidden_states

        # Self Attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_kv = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

        # INL Dynamics (after attention, before residual)
        if self.dynamics is not None:
            hidden_states, velocity_state = self.dynamics(hidden_states, velocity_state)

        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, token_ids=token_ids)
        hidden_states = residual + hidden_states

        return hidden_states, new_kv, velocity_state
