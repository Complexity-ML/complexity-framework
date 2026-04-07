"""
I64 Integer Attention — INT8 matmuls + LUT activations.

Train in float, deploy in INT8. All linear projections (QKV, mu-QKV, O)
are quantizable to INT8 via quantize(). Float-irreducible ops (RoPE,
softmax, QK dot product) remain in float.

Registered as "i64" / "integer" in the attention registry.

INL 2025 — ported from complexity-i64.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .base import AttentionBase, AttentionConfig
from ..registry import register_attention
from ..position.rotary import RotaryEmbedding, PartialRoPE, apply_rotary_pos_emb
from ..integer_ops import int8_linear, quantize_weight_int8


HAS_SDPA = hasattr(F, "scaled_dot_product_attention")


@register_attention("i64")
@register_attention("integer")
class I64Attention(AttentionBase):
    """
    Integer-native Mu-Guided Attention.

    INT8: QKV projection, mu projection, O projection (5 matmuls -> 3 fused INT8)
    Float: RoPE, QK dot product, softmax, attention x V (irreducible)

    Call quantize() after training to convert all linear ops to INT8.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__(config)

        # QKV projections (float for training, INT8 after quantize())
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Mu-guided projections (INL 2025)
        self.mu_to_q = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.mu_to_k = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.mu_to_v = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        for proj in [self.mu_to_q, self.mu_to_k, self.mu_to_v]:
            nn.init.normal_(proj.weight, std=0.01)

        # QK Normalization
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

        # RoPE (Partial if rope_fraction < 1.0)
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
        self.use_sdpa = config.use_sdpa and HAS_SDPA

    def _project_qkv(self, hidden, mu_prev):
        """QKV + mu projections — INT8 if quantized."""
        if hasattr(self, 'qkv_int8'):
            qkv = int8_linear(hidden, self.qkv_int8, self.qkv_scale)
            q = qkv[..., :self.q_size]
            k = qkv[..., self.q_size:self.q_size + self.kv_size]
            v = qkv[..., self.q_size + self.kv_size:]

            if mu_prev is not None and hasattr(self, 'mu_qkv_int8'):
                mu_qkv = int8_linear(mu_prev, self.mu_qkv_int8, self.mu_qkv_scale)
                q = q + mu_qkv[..., :self.q_size]
                k = k + mu_qkv[..., self.q_size:self.q_size + self.kv_size]
                v = v + mu_qkv[..., self.q_size + self.kv_size:]
        else:
            q = self.q_proj(hidden)
            k = self.k_proj(hidden)
            v = self.v_proj(hidden)
            if mu_prev is not None:
                q = q + self.mu_to_q(mu_prev)
                k = k + self.mu_to_k(mu_prev)
                v = v + self.mu_to_v(mu_prev)

        return q, k, v

    def _o_proj_forward(self, out):
        """O projection — INT8 if quantized."""
        if hasattr(self, 'o_int8'):
            return int8_linear(out, self.o_int8, self.o_scale)
        return self.o_proj(out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        mu_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape

        # QKV projection (INT8 or float)
        q, k, v = self._project_qkv(hidden_states, mu_prev)

        # Reshape to heads
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # QK Norm
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # RoPE (float — irreducible)
        kv_seq_len = seq_len
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[2]

        cos, sin = self.rotary_emb(kv_seq_len)
        cos = cos.to(q.device, dtype=q.dtype)
        sin = sin.to(q.device, dtype=q.dtype)

        if past_key_value is not None:
            cos = cos[kv_seq_len - seq_len:]
            sin = sin[kv_seq_len - seq_len:]

        q, k = self.rotary_emb.rotate(q, k, cos, sin)

        # KV cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        new_kv = (k, v) if use_cache else None

        # GQA expand
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Attention (float — softmax is irreducible)
        dropout_p = self.attention_dropout if self.training else 0.0

        if self.use_sdpa:
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask,
                dropout_p=dropout_p,
                is_causal=(attention_mask is None and past_key_value is None),
            )
        else:
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            else:
                causal_mask = torch.triu(
                    torch.ones(seq_len, kv_seq_len, device=q.device, dtype=torch.bool),
                    diagonal=kv_seq_len - seq_len + 1,
                )
                attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            if self.training and dropout_p > 0:
                attn_weights = F.dropout(attn_weights, p=dropout_p)
            attn_output = torch.matmul(attn_weights, v)

        # O projection (INT8 or float)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        out = self._o_proj_forward(attn_output)

        return out, new_kv

    def quantize(self):
        """Fuse and quantize QKV + mu + O to INT8."""
        # Fused QKV
        q_w = self.q_proj.weight.data
        k_w = self.k_proj.weight.data
        v_w = self.v_proj.weight.data
        qkv_w = torch.cat([q_w, k_w, v_w], dim=0)
        qkv_q, qkv_s = quantize_weight_int8(qkv_w)
        self.register_buffer("qkv_int8", qkv_q)
        self.register_buffer("qkv_scale", qkv_s)
        self.q_size = q_w.shape[0]
        self.kv_size = k_w.shape[0]

        # Fused mu_QKV
        mu_q_w = self.mu_to_q.weight.data
        mu_k_w = self.mu_to_k.weight.data
        mu_v_w = self.mu_to_v.weight.data
        mu_qkv_w = torch.cat([mu_q_w, mu_k_w, mu_v_w], dim=0)
        mu_qkv_q, mu_qkv_s = quantize_weight_int8(mu_qkv_w)
        self.register_buffer("mu_qkv_int8", mu_qkv_q)
        self.register_buffer("mu_qkv_scale", mu_qkv_s)

        # O projection
        o_q, o_s = quantize_weight_int8(self.o_proj.weight.data)
        self.register_buffer("o_int8", o_q)
        self.register_buffer("o_scale", o_s)

        # Free float weights
        del self.q_proj
        del self.k_proj
        del self.v_proj
        del self.mu_to_q
        del self.mu_to_k
        del self.mu_to_v
        del self.o_proj
