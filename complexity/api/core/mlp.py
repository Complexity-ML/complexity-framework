"""
MLP API - Factories pour créer des MLP layers.
"""

from __future__ import annotations

from typing import Type
import torch.nn as nn

from complexity.core import (
    MLP_REGISTRY,
    register_mlp,
    MLPBase,
    MLPConfig,
    StandardMLP,
    SwiGLUMLP,
    GeGLUMLP,
    TokenRoutedMLP,
    TokenRoutedMLPParallel,
)


class MLP:
    """
    Factory pour créer des MLP layers.

    Usage:
        mlp = MLP.create("swiglu", hidden_size=4096, intermediate_size=11008)
        mlp = MLP.swiglu(hidden_size=4096, intermediate_size=11008)
        mlp = MLP.geglu(hidden_size=4096, intermediate_size=11008)
        mlp = MLP.standard(hidden_size=4096, intermediate_size=16384)
        mlp = MLP.moe(hidden_size=4096, num_experts=8, top_k=2)
    """

    TYPES = {
        "standard": StandardMLP,
        "swiglu": SwiGLUMLP,
        "geglu": GeGLUMLP,
        "gated": SwiGLUMLP,  # alias
        "moe": TokenRoutedMLP,
        "moe_parallel": TokenRoutedMLPParallel,
    }

    @classmethod
    def create(cls, mlp_type: str = "swiglu", **kwargs) -> nn.Module:
        """
        Crée un MLP layer.

        Args:
            mlp_type: "standard", "swiglu", "geglu", "gated", "moe"
            **kwargs: hidden_size, intermediate_size, dropout, ...
        """
        if mlp_type in MLP_REGISTRY._registry:
            mlp_cls = MLP_REGISTRY.get(mlp_type)
            config = MLPConfig(**kwargs)
            return mlp_cls(config)

        if mlp_type not in cls.TYPES:
            raise ValueError(f"Unknown MLP type: {mlp_type}. Use: {list(cls.TYPES.keys())}")

        mlp_cls = cls.TYPES[mlp_type]
        config = MLPConfig(**kwargs)
        return mlp_cls(config)

    @classmethod
    def swiglu(cls, hidden_size: int, intermediate_size: int = None, **kwargs) -> nn.Module:
        """SwiGLU MLP (Llama style)."""
        intermediate_size = intermediate_size or int(hidden_size * 8 / 3)
        return cls.create("swiglu", hidden_size=hidden_size, intermediate_size=intermediate_size, **kwargs)

    @classmethod
    def geglu(cls, hidden_size: int, intermediate_size: int = None, **kwargs) -> nn.Module:
        """GeGLU MLP."""
        intermediate_size = intermediate_size or hidden_size * 4
        return cls.create("geglu", hidden_size=hidden_size, intermediate_size=intermediate_size, **kwargs)

    @classmethod
    def standard(cls, hidden_size: int, intermediate_size: int = None, **kwargs) -> nn.Module:
        """Standard MLP (GPT style)."""
        intermediate_size = intermediate_size or hidden_size * 4
        return cls.create("standard", hidden_size=hidden_size, intermediate_size=intermediate_size, **kwargs)

    @classmethod
    def moe(cls, hidden_size: int, num_experts: int = 8, top_k: int = 2, **kwargs) -> nn.Module:
        """Mixture of Experts."""
        return cls.create("moe", hidden_size=hidden_size, num_experts=num_experts, moe_top_k=top_k, **kwargs)

    @classmethod
    def register(cls, name: str, mlp_cls: Type):
        """Enregistre un nouveau type de MLP."""
        register_mlp(name)(mlp_cls)
        cls.TYPES[name] = mlp_cls


# Aliases
SwiGLU = SwiGLUMLP
GeGLU = GeGLUMLP

__all__ = [
    "MLP",
    "SwiGLU",
    "GeGLU",
    "SwiGLUMLP",
    "GeGLUMLP",
    "StandardMLP",
    "TokenRoutedMLP",
    "TokenRoutedMLPParallel",
    "MLPBase",
    "MLPConfig",
]
