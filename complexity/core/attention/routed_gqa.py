"""
Routed GQA — Sort-and-Split Attention with shared K/V.

Innovation from Parameter Golf (Complexity-ML, 2026):
  Q and O projections are routed (E expert weight sets, sort-and-split).
  K and V are shared (single weight set) for compatible attention spaces.
  Full causal attention runs on all tokens after routing.

  Tokens → sort → Q experts → unsort → full attention → sort → O experts → unsort

Usage:
    config = AttentionConfig(hidden_size=512, num_attention_heads=8, num_key_value_heads=4)
    attn = RoutedGQA(config, num_experts=4)
    out, cache = attn(hidden_states, sort_idx=sort_idx)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .base import AttentionBase, AttentionConfig
from ..registry import register_attention
from ..position.rotary import RotaryEmbedding, PartialRoPE, apply_rotary_pos_emb
from ..triton_kernels import routed_proj as _routed_proj_impl


HAS_SDPA = hasattr(F, "scaled_dot_product_attention")


def _routed_proj(x: torch.Tensor, weight: torch.Tensor,
                 sort_idx: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Sort-and-split projection: fused Triton kernel on CUDA, bmm fallback on CPU.

    Requires N (batch × seq) divisible by num_experts for fullgraph compat.
    """
    return _routed_proj_impl(x, weight, sort_idx, num_experts)


@register_attention("routed_gqa")
@register_attention("sort_split_gqa")
class RoutedGQA(AttentionBase):
    """
    Routed Grouped Query Attention.

    Q and O projections use sort-and-split routing (E expert weight sets).
    K and V are shared — all tokens project into the same key/value space,
    ensuring compatible attention across expert groups.

    This gives each expert specialized "questions" (Q) and "interpretations" (O)
    while maintaining coherent attention through shared K/V.

    Benefits:
    - E× specialized Q/O weights for same compute
    - Compatible K/V space across all experts
    - fullgraph compatible (static shapes via argsort)
    - Works with GQA, MHA, MQA
    """

    def __init__(self, config: AttentionConfig, num_experts: int = 4):
        super().__init__(config)
        self.num_experts = num_experts

        kv_dim = self.num_kv_heads * self.head_dim
        q_dim = self.num_heads * self.head_dim

        # Routed Q/O: [E, in, out] — expert-specific
        self.q_proj_w = nn.Parameter(torch.empty(num_experts, self.hidden_size, q_dim))
        self.o_proj_w = nn.Parameter(torch.empty(num_experts, q_dim, self.hidden_size))
        nn.init.kaiming_uniform_(self.q_proj_w, a=5**0.5)
        nn.init.zeros_(self.o_proj_w)

        # Shared K/V — compatible attention space
        self.k_proj = nn.Linear(self.hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, kv_dim, bias=False)

        # QK Normalization
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

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

        self.use_sdpa = config.use_sdpa and HAS_SDPA

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        sort_idx: Optional[torch.Tensor] = None,
        mu_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for Routed GQA.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            sort_idx: [N] precomputed argsort for routing (reuse across layers)
            mu_prev: [batch, seq_len, hidden_size] mu guidance (unused for now,
                accepted for interface compatibility with GQA)
        """
        batch_size, seq_len, _ = hidden_states.shape
        # sort_idx must always be provided — no fallback for fullgraph compat

        # Routed Q, shared K/V
        q = _routed_proj(hidden_states, self.q_proj_w, sort_idx, self.num_experts)
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

        # Rotary embeddings
        kv_seq_len = seq_len
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[2]

        cos, sin = self.rotary_emb(kv_seq_len)
        cos = cos.to(q.device, dtype=q.dtype)
        sin = sin.to(q.device, dtype=q.dtype)

        if past_key_value is not None:
            cos = cos[kv_seq_len - seq_len:]
            sin = sin[kv_seq_len - seq_len:]

        q, k = self.rotary_emb.apply(q, k, cos, sin)

        # KV cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        new_past_key_value = (k, v) if use_cache else None

        # GQA: repeat KV heads
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Full causal attention (all tokens interact)
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, is_causal=(attention_mask is None),
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        # Routed output projection
        attn_output = _routed_proj(attn_output, self.o_proj_w, sort_idx, self.num_experts)

        return attn_output, new_past_key_value
