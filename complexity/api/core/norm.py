"""
Normalization API - Factories pour créer des normalization layers.
"""

from __future__ import annotations

import torch.nn as nn

from complexity.core import (
    NORMALIZATION_REGISTRY,
    RMSNorm,
    LayerNorm,
    IdentityNorm,
    build_norm,
)


class Norm:
    """
    Factory pour créer des normalization layers.

    Usage:
        norm = Norm.rms(hidden_size=4096)
        norm = Norm.layer(hidden_size=4096)
    """

    TYPES = {
        "rmsnorm": RMSNorm,
        "rms": RMSNorm,
        "layernorm": LayerNorm,
        "layer": LayerNorm,
        "identity": IdentityNorm,
        "none": IdentityNorm,
    }

    @classmethod
    def create(cls, norm_type: str = "rmsnorm", hidden_size: int = None, **kwargs) -> nn.Module:
        """Crée une normalization layer."""
        if norm_type in NORMALIZATION_REGISTRY._registry:
            return NORMALIZATION_REGISTRY.get(norm_type)(hidden_size, **kwargs)

        norm_type = norm_type.lower()
        if norm_type not in cls.TYPES:
            raise ValueError(f"Unknown norm type: {norm_type}. Use: {list(cls.TYPES.keys())}")

        return cls.TYPES[norm_type](hidden_size, **kwargs)

    @classmethod
    def rms(cls, hidden_size: int, eps: float = 1e-6, **kwargs) -> nn.Module:
        """RMSNorm (Llama style)."""
        return cls.create("rmsnorm", hidden_size=hidden_size, eps=eps, **kwargs)

    @classmethod
    def layer(cls, hidden_size: int, eps: float = 1e-5, **kwargs) -> nn.Module:
        """LayerNorm (GPT style)."""
        return cls.create("layernorm", hidden_size=hidden_size, eps=eps, **kwargs)


__all__ = [
    "Norm",
    "RMSNorm",
    "LayerNorm",
    "IdentityNorm",
    "build_norm",
]
