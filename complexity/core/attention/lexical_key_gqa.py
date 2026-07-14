"""GQA with an in-place learned lexical residual in the existing keys."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_attention
from .base import AttentionConfig
from .gqa import GroupedQueryAttention


@register_attention("lexical_key_gqa")
class LexicalKeyGQA(GroupedQueryAttention):
    """Inject tied lexical objects into K without widening attention heads."""

    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        self.lexical_object_rank = int(config.lexical_object_rank)
        if self.lexical_object_rank <= 0:
            raise ValueError("lexical_object_rank must be positive")
        if self.lexical_object_rank > self.head_dim:
            raise ValueError("lexical_object_rank cannot exceed the GQA head dimension")
        self.lexical_gate = nn.Parameter(
            torch.full(
                (self.num_kv_heads,),
                float(config.lexical_key_gate_init),
            )
        )

    @staticmethod
    def _rms_normalize(value: torch.Tensor) -> torch.Tensor:
        value_float = value.float()
        normalized = value_float * torch.rsqrt(
            value_float.square().mean(dim=-1, keepdim=True) + 1e-6
        )
        return normalized.to(value.dtype)

    def _modify_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lexical_scale = kwargs.get("lexical_scale")
        if lexical_scale is None:
            raise ValueError("lexical_key_gqa requires the tied lexical object scale")
        if lexical_scale.shape[:2] != (k.shape[0], k.shape[2]):
            raise ValueError("lexical_scale must match K batch and sequence")
        if lexical_scale.shape[-1] != self.lexical_object_rank:
            raise ValueError(
                f"expected lexical rank {self.lexical_object_rank}, "
                f"got {lexical_scale.shape[-1]}"
            )
        lexical_key = self._rms_normalize(lexical_scale)
        lexical_key = F.pad(
            lexical_key,
            (0, self.head_dim - self.lexical_object_rank),
        ).unsqueeze(1)
        lexical_key = lexical_key.expand(-1, self.num_kv_heads, -1, -1)
        gate = torch.tanh(self.lexical_gate).view(1, self.num_kv_heads, 1, 1)
        return q, k + gate.to(k.dtype) * lexical_key.to(k.dtype)


@register_attention("projected_lexical_key_gqa")
class ProjectedLexicalKeyGQA(GroupedQueryAttention):
    """Project learned lexical objects into K without widening GQA heads."""

    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        self.lexical_object_rank = int(config.lexical_object_rank)
        if self.lexical_object_rank <= 0:
            raise ValueError("lexical_object_rank must be positive")
        self.lexical_k_proj = nn.Linear(
            self.lexical_object_rank,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.lexical_gate = nn.Parameter(
            torch.full(
                (self.num_kv_heads,),
                float(config.lexical_key_gate_init),
            )
        )

    def _modify_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lexical_scale = kwargs.get("lexical_scale")
        if lexical_scale is None:
            raise ValueError(
                "projected_lexical_key_gqa requires the tied lexical object scale"
            )
        if lexical_scale.shape[:2] != (k.shape[0], k.shape[2]):
            raise ValueError("lexical_scale must match K batch and sequence")
        if lexical_scale.shape[-1] != self.lexical_object_rank:
            raise ValueError(
                f"expected lexical rank {self.lexical_object_rank}, "
                f"got {lexical_scale.shape[-1]}"
            )
        lexical_key = self.lexical_k_proj(lexical_scale).view(
            k.shape[0], k.shape[2], self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        gate = torch.tanh(self.lexical_gate).view(1, self.num_kv_heads, 1, 1)
        return q, k + gate.to(k.dtype) * lexical_key.to(k.dtype)
