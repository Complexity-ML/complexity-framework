"""
Grouped Query Attention (GQA) - Llama 2/3 style.

GQA uses fewer KV heads than Q heads, reducing memory and compute
while maintaining quality. When num_kv_heads=1, it becomes MQA.
When num_kv_heads=num_heads, it becomes standard MHA.

Supports optional Mu-guided attention by adding learned projections from the
previous layer's Mu state into K, Q, and V.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from typing import Optional, Tuple

from .base import AttentionBase, AttentionConfig
from ..registry import register_attention
from ..position.rotary import RotaryEmbedding, PartialRoPE


HAS_SDPA = hasattr(F, "scaled_dot_product_attention")
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    SDPA_KERNEL_BACKENDS = [
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    ]
except Exception:
    SDPBackend = None
    sdpa_kernel = None
    SDPA_KERNEL_BACKENDS = None

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

        # KQV order keeps K and V adjacent for cache-oriented implementations.
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Mu-to-KQV projections are allocated only when Mu is enabled; dense
        # baselines should not count dead parameters that never run.
        self.use_mu_guidance = bool(getattr(config, "use_mu_guidance", False))
        if self.use_mu_guidance:
            self.mu_to_k = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
            self.mu_to_q = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.mu_to_v = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)

        # QK normalization stabilizes attention logits on small and large runs.
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            from ..normalization.norms import RMSNorm as _RMSNorm
            self.q_norm = _RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = _RMSNorm(self.head_dim, eps=1e-6)

        # Rotary embeddings (Partial RoPE if rope_fraction < 1.0)
        rope_fraction = getattr(config, "rope_fraction", 1.0)
        if rope_fraction < 1.0:
            self.rotary_emb = PartialRoPE(
                self.head_dim,
                max_seq_len=config.max_position_embeddings,
                theta=config.rope_theta,
                rope_fraction=rope_fraction,
            )
        else:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                max_seq_len=config.max_position_embeddings,
                theta=config.rope_theta,
            )

        self.attention_dropout = config.attention_dropout
        self.sliding_window = config.sliding_window
        self.use_sdpa = config.use_sdpa and HAS_SDPA

        # Attention logit scale. Default = 1/√d_head (standard).
        # μP variant = 1/d_head (Yang et al. 2022) for hyper-parameter
        # transfer across widths. Stored once for reuse in both SDPA
        # (passed via the `scale=` kwarg) and the standard fallback.
        if getattr(config, "use_mup_attn_scale", False):
            self.attn_scale = 1.0 / float(self.head_dim)
        else:
            self.attn_scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        mu_prev: Optional[torch.Tensor] = None,
        **kwargs,
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

        k_dim = self.num_kv_heads * self.head_dim
        q_dim = self.num_heads * self.head_dim
        v_dim = self.num_kv_heads * self.head_dim
        w_kqv = torch.cat([self.k_proj.weight, self.q_proj.weight, self.v_proj.weight], dim=0)
        kqv = F.linear(hidden_states, w_kqv)
        k, q, v = kqv.split([k_dim, q_dim, v_dim], dim=-1)
        if self.use_mu_guidance and mu_prev is not None:
            if mu_prev.shape != hidden_states.shape:
                raise ValueError(
                    "mu_prev must match hidden_states shape "
                    f"{tuple(hidden_states.shape)}, got {tuple(mu_prev.shape)}"
                )
            k = k + self.mu_to_k(mu_prev)
            q = q + self.mu_to_q(mu_prev)
            v = v + self.mu_to_v(mu_prev)

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

        q, k = self.rotary_emb.rotate(q, k, cos, sin)

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

        # is_causal=True only valid when q_len == kv_len (no KV cache)
        # With KV cache, q_len=1 and kv_len>1 → the single query token
        # must attend to all cached keys, not be masked by a causal mask.
        use_causal = (attn_mask is None) and (q.shape[2] == k.shape[2])
        sdpa_context = (
            sdpa_kernel(SDPA_KERNEL_BACKENDS)
            if sdpa_kernel is not None and SDPA_KERNEL_BACKENDS is not None
            else nullcontext()
        )
        with sdpa_context:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=use_causal,
                scale=self.attn_scale,
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
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale

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
