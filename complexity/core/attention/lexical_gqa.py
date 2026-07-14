"""GQA with a baseline-preserving lexical score channel."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..registry import register_attention
from .base import AttentionConfig
from .gqa import GroupedQueryAttention


@register_attention("lexical_gqa")
@register_attention("lexical_bias_gqa")
class LexicalBiasGQA(GroupedQueryAttention):
    """Add contextual-query/token-key lexical scores to unchanged GQA.

    The lexical channels are concatenated to Q and K while V stays contextual.
    With a zero lexical gate, their dot product is exactly zero and the operator
    reduces mathematically to the underlying GQA attention.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        self.lexical_rank = int(config.lexical_gqa_rank)
        if self.lexical_rank <= 0:
            raise ValueError("lexical_gqa_rank must be positive")
        self.lexical_object_rank = int(config.lexical_object_rank)
        self.lexical_q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.lexical_rank,
            bias=False,
        )
        self.lexical_k_proj = nn.Linear(
            self.lexical_object_rank,
            self.num_kv_heads * self.lexical_rank,
            bias=False,
        )
        self.lexical_gate = nn.Parameter(
            torch.full(
                (self.num_heads,),
                float(config.lexical_gqa_gate_init),
            )
        )

    @staticmethod
    def _rms_normalize(value: torch.Tensor) -> torch.Tensor:
        value_float = value.float()
        normalized = value_float * torch.rsqrt(
            value_float.square().mean(dim=-1, keepdim=True) + 1e-6
        )
        return normalized.to(value.dtype)

    def _token_code(self, token_ids: torch.Tensor) -> torch.Tensor:
        phase_dtype = (
            torch.float32 if token_ids.device.type == "mps" else torch.float64
        )
        dimensions = torch.arange(
            1,
            self.lexical_object_rank + 1,
            device=token_ids.device,
            dtype=phase_dtype,
        )
        phases = (
            (token_ids.to(phase_dtype)[..., None] + 1.0)
            * torch.pi
            * torch.sqrt(dimensions)
        )
        code = torch.where(
            (dimensions.to(torch.int64) % 2) == 0,
            torch.sin(phases),
            torch.cos(phases),
        )
        return code.to(self.lexical_k_proj.weight.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        mu_prev: Optional[torch.Tensor] = None,
        lexical_scale: Optional[torch.Tensor] = None,
        token_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        if lexical_scale is None:
            raise ValueError("lexical_gqa requires the tied lexical object scale")
        if lexical_scale.shape[:2] != hidden_states.shape[:2]:
            raise ValueError("lexical_scale must match hidden_states batch and sequence")
        if lexical_scale.shape[-1] != self.lexical_object_rank:
            raise ValueError(
                f"expected lexical rank {self.lexical_object_rank}, "
                f"got {lexical_scale.shape[-1]}"
            )
        if token_ids is None or token_ids.shape != hidden_states.shape[:2]:
            raise ValueError("lexical_gqa requires token_ids matching batch and sequence")

        batch_size, seq_len, _ = hidden_states.shape
        k_dim = self.num_kv_heads * self.head_dim
        q_dim = self.num_heads * self.head_dim
        v_dim = self.num_kv_heads * self.head_dim
        w_kqv = torch.cat(
            [self.k_proj.weight, self.q_proj.weight, self.v_proj.weight], dim=0
        )
        kqv = torch.nn.functional.linear(hidden_states, w_kqv)
        k, q, v = kqv.split([k_dim, q_dim, v_dim], dim=-1)
        if self.use_mu_guidance and mu_prev is not None:
            if mu_prev.shape != hidden_states.shape:
                raise ValueError("mu_prev must match hidden_states shape")
            k = k + self.mu_to_k(mu_prev)
            q = q + self.mu_to_q(mu_prev)
            v = v + self.mu_to_v(mu_prev)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        v = v.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        kv_seq_len = seq_len
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[2]
        cos, sin = self.rotary_emb(kv_seq_len)
        cos = cos.to(q.device, dtype=q.dtype)
        sin = sin.to(q.device, dtype=q.dtype)
        if past_key_value is not None:
            cos = cos[kv_seq_len - seq_len :]
            sin = sin[kv_seq_len - seq_len :]
        q, k = self.rotary_emb.rotate(q, k, cos, sin)

        lexical_q = self.lexical_q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.lexical_rank
        ).transpose(1, 2)
        lexical_source = lexical_scale + self._token_code(token_ids).to(
            lexical_scale.dtype
        )
        lexical_k = self.lexical_k_proj(lexical_source).view(
            batch_size, seq_len, self.num_kv_heads, self.lexical_rank
        ).transpose(1, 2)
        lexical_q = self._rms_normalize(lexical_q)
        lexical_k = self._rms_normalize(lexical_k)
        gate = torch.tanh(self.lexical_gate).view(1, self.num_heads, 1, 1)
        q = torch.cat([q, gate.to(q.dtype) * lexical_q], dim=-1)
        k = torch.cat([k, lexical_k], dim=-1)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        new_past_key_value = (k, v) if use_cache else None

        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)
        if self.use_sdpa:
            attn_output = self._sdpa_attention(
                q, k, v, attention_mask, seq_len, kv_seq_len
            )
        else:
            attn_output = self._standard_attention(
                q, k, v, attention_mask, seq_len, kv_seq_len
            )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        return self.o_proj(attn_output), new_past_key_value
