"""
Position API - Factories pour créer des position encodings.
"""

from __future__ import annotations

import torch.nn as nn

from complexity.core import (
    POSITION_REGISTRY,
    RotaryEmbedding,
    StandardRoPE,
    YaRNRoPE,
    DynamicNTKRoPE,
    ALiBiPositionBias,
    LearnedPositionEmbedding,
    rotate_half,
    apply_rotary_pos_emb,
)


class Position:
    """
    Factory pour créer des position encodings.

    Usage:
        rope = Position.rope(dim=128, max_seq_len=8192)
        yarn = Position.yarn(dim=128, max_seq_len=8192, scale=4.0)
        alibi = Position.alibi(num_heads=32)
    """

    TYPES = {
        "rope": StandardRoPE,
        "yarn": YaRNRoPE,
        "ntk": DynamicNTKRoPE,
        "alibi": ALiBiPositionBias,
        "learned": LearnedPositionEmbedding,
    }

    @classmethod
    def create(cls, pos_type: str = "rope", **kwargs) -> nn.Module:
        """Crée un position encoding."""
        if pos_type in POSITION_REGISTRY._registry:
            return POSITION_REGISTRY.get(pos_type)(**kwargs)

        if pos_type not in cls.TYPES:
            raise ValueError(f"Unknown position type: {pos_type}. Use: {list(cls.TYPES.keys())}")

        return cls.TYPES[pos_type](**kwargs)

    @classmethod
    def rope(cls, dim: int, max_seq_len: int = 8192, base: float = 10000.0, **kwargs) -> nn.Module:
        """Standard RoPE."""
        return cls.create("rope", dim=dim, max_position_embeddings=max_seq_len, base=base, **kwargs)

    @classmethod
    def yarn(cls, dim: int, max_seq_len: int = 8192, scale: float = 4.0, **kwargs) -> nn.Module:
        """YaRN extended context."""
        return cls.create("yarn", dim=dim, max_position_embeddings=max_seq_len, scaling_factor=scale, **kwargs)

    @classmethod
    def ntk(cls, dim: int, max_seq_len: int = 8192, **kwargs) -> nn.Module:
        """Dynamic NTK-aware RoPE."""
        return cls.create("ntk", dim=dim, max_position_embeddings=max_seq_len, **kwargs)

    @classmethod
    def alibi(cls, num_heads: int, **kwargs) -> nn.Module:
        """ALiBi position bias."""
        return cls.create("alibi", num_heads=num_heads, **kwargs)

    @classmethod
    def learned(cls, max_seq_len: int, hidden_size: int, **kwargs) -> nn.Module:
        """Learned position embeddings."""
        return cls.create("learned", max_position_embeddings=max_seq_len, hidden_size=hidden_size, **kwargs)


# Aliases
RoPE = StandardRoPE
YaRN = YaRNRoPE
ALiBi = ALiBiPositionBias

__all__ = [
    "Position",
    "RoPE",
    "YaRN",
    "ALiBi",
    "StandardRoPE",
    "YaRNRoPE",
    "DynamicNTKRoPE",
    "ALiBiPositionBias",
    "LearnedPositionEmbedding",
    "RotaryEmbedding",
    "rotate_half",
    "apply_rotary_pos_emb",
]
