"""
Grouped Query Attention (GQA) - Llama 2/3 style.

GQA uses fewer KV heads than Q heads, reducing memory and compute
while maintaining quality. When num_kv_heads=1, it becomes MQA.
When num_kv_heads=num_heads, it becomes standard MHA.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .base import AttentionBase, AttentionConfig
from ..registry import register_attention
from ..position.rotary import RotaryEmbedding, apply_rotary_pos_emb


HAS_SDPA = hasattr(F, "scaled_dot_product_attention")


@register_attention("gqa")
@register_attention("grouped_query")
class GroupedQueryAttention(AttentionBase):
    """
    Grouped Query Attention with modern optimizations.

    Features:
    - Grouped Query Attention (GQA) - configurable KV heads
    - Rotary Position Embeddings (RoPE)
    - Flash Attention via SDPA (PyTorch 2.0+)
    - QK Normalization (optional, stabilizes training)
    - Sliding Window Attention (optional, for efficiency)

    References:
        - GQA Paper: https://arxiv.org/abs/2305.13245
        - Llama 2: https://arxiv.org/abs/2307.09288
    """

    def __init__(self, config: AttentionConfig):
        super().__init__(config)

        # Initialize projections
        self._init_projections(bias=False)

        # QK Normalization (2024 innovation - stabilizes training)
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
        )

        self.attention_dropout = config.attention_dropout
        self.sliding_window = config.sliding_window
        self.use_sdpa = config.use_sdpa and HAS_SDPA

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for Grouped Query Attention.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size).
            attention_mask: Optional attention mask of shape (batch_size, 1, seq_len, seq_len).
            past_key_value: Optional cached (key, value) tuple for autoregressive generation.
            use_cache: Whether to return the new (key, value) cache for next step.

        Returns:
            Tuple of:
                - attn_output: Output tensor of shape (batch_size, seq_len, hidden_size).
                - past_key_value: Updated (key, value) cache if use_cache=True, else None.
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to [batch, heads, seq, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # QK Normalization
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Handle KV cache for generation
        kv_seq_len = seq_len
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[2]

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(kv_seq_len)
        cos = cos.to(q.device, dtype=q.dtype)
        sin = sin.to(q.device, dtype=q.dtype)

        # For cached generation, only rotate the new positions
        if past_key_value is not None:
            cos = cos[kv_seq_len - seq_len:]
            sin = sin[kv_seq_len - seq_len:]

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Update KV cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        new_past_key_value = (k, v) if use_cache else None

        # GQA: Repeat KV heads to match Q heads
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Use SDPA (Flash Attention) if available
        if self.use_sdpa:
            attn_output = self._sdpa_attention(q, k, v, attention_mask, seq_len, kv_seq_len)
        else:
            attn_output = self._standard_attention(q, k, v, attention_mask, seq_len, kv_seq_len)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, new_past_key_value

    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        seq_len: int,
        kv_seq_len: int,
    ) -> torch.Tensor:
        """Flash Attention via PyTorch SDPA."""
        # Build attention mask for SDPA
        if self.sliding_window is not None and seq_len > self.sliding_window:
            attn_mask = self._make_sliding_window_mask(seq_len, kv_seq_len, q.device, q.dtype)
        else:
            attn_mask = None

        dropout_p = self.attention_dropout if self.training else 0.0

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=(attn_mask is None),
        )

        return attn_output

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        seq_len: int,
        kv_seq_len: int,
    ) -> torch.Tensor:
        """Standard attention fallback (for PyTorch < 2.0)."""
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self.sliding_window is not None:
            mask = self._make_sliding_window_mask(seq_len, kv_seq_len, q.device, q.dtype)
            attn_weights = attn_weights + mask
        else:
            causal_mask = torch.triu(
                torch.ones(seq_len, kv_seq_len, device=q.device, dtype=torch.bool),
                diagonal=kv_seq_len - seq_len + 1,
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        if self.training and self.attention_dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout)

        return torch.matmul(attn_weights, v)

    def _make_sliding_window_mask(
        self,
        seq_len: int,
        kv_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create sliding window attention mask (Mistral-style)."""
        mask = torch.full((seq_len, kv_seq_len), float("-inf"), device=device, dtype=dtype)

        for i in range(seq_len):
            start = max(0, kv_seq_len - seq_len + i - self.sliding_window + 1)
            end = kv_seq_len - seq_len + i + 1
            mask[i, start:end] = 0.0

        return mask.unsqueeze(0).unsqueeze(0)


# Convenience aliases
@register_attention("mha")
@register_attention("multi_head")
class MultiHeadAttention(GroupedQueryAttention):
    """
    Standard Multi-Head Attention.

    This is GQA with num_kv_heads = num_heads.
    """

    def __init__(self, config: AttentionConfig):
        # Force num_kv_heads = num_heads for MHA
        config.num_key_value_heads = config.num_attention_heads
        super().__init__(config)


@register_attention("mqa")
@register_attention("multi_query")
class MultiQueryAttention(GroupedQueryAttention):
    """
    Multi-Query Attention.

    This is GQA with num_kv_heads = 1.
    Much more memory efficient but slightly lower quality.

    Reference: https://arxiv.org/abs/1911.02150
    """

    def __init__(self, config: AttentionConfig):
        # Force num_kv_heads = 1 for MQA
        config.num_key_value_heads = 1
        super().__init__(config)
