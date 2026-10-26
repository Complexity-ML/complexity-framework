"""
MLP API - Factories pour créer des MLP layers.
"""

from __future__ import annotations

from typing import Type

import torch.nn as nn

from complexity.core import (
    MLP_REGISTRY,
    MLPBase,
    MLPConfig,
    register_mlp,
)
from complexity.core import (
    TRHashEngineMLP as CoreTRHashEngineMLP,
)


class TokenRoutedMLP(nn.Module):
    """API wrapper around the core TR-Hash engine MoE.

    The public API returns ``(output, aux_loss)`` for compatibility with other
    MoE layers. Core model code should use ``complexity.core.mlp.TRHashEngineMLP``
    directly. This wrapper uses that same TR-HASH implementation.

    ``token_ids`` is required for composed routing with its two causal lags.
    """

    def __init__(self, **kwargs):
        super().__init__()
        if "intermediate_size" not in kwargs:
            kwargs["intermediate_size"] = kwargs["hidden_size"]
        self.config = MLPConfig(**kwargs)
        self.mlp = CoreTRHashEngineMLP(self.config)

    def forward(self, hidden_states, token_ids=None, **kwargs):
        if token_ids is None:
            raise ValueError("TR-Hash MoE requires token_ids to route")
        out = self.mlp(hidden_states, token_ids=token_ids, **kwargs)
        aux_loss = out.new_zeros(())
        return out, aux_loss


class MLP:
    """
    Factory pour créer des MLP layers.

    TR-HASH utilise un routage composé causal, quatre experts et deux routes
    fixes de même poids, avec une branche partagée.

    Usage:
        mlp = MLP.create("moe", hidden_size=4096, num_experts=4)
        mlp = MLP.moe(hidden_size=4096, num_experts=4, top_k=2)
    """

    TYPES = {
        "moe": TokenRoutedMLP,
    }

    @classmethod
    def create(cls, mlp_type: str = "moe", **kwargs) -> nn.Module:
        """
        Crée un MLP layer.

        Args:
            mlp_type: "moe" (returns (output, aux_loss)), or any name
                registered directly on MLP_REGISTRY (e.g. "tr_hash_engine",
                returns just output).
            **kwargs: hidden_size, intermediate_size, dropout, ...
        """
        kwargs = dict(kwargs)
        if "intermediate_size" not in kwargs:
            hidden_size = kwargs.get("hidden_size")
            if hidden_size is None:
                raise ValueError("hidden_size is required when intermediate_size is omitted")
            kwargs["intermediate_size"] = hidden_size
        if mlp_type == "moe":
            return TokenRoutedMLP(**kwargs)
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
    def moe(cls, hidden_size: int, num_experts: int = 4, **kwargs) -> nn.Module:
        """TR-HASH à quatre experts et deux routes fixes."""
        return cls.create("moe", hidden_size=hidden_size, num_experts=num_experts, **kwargs)

    @classmethod
    def register(cls, name: str, mlp_cls: Type):
        """Enregistre un nouveau type de MLP."""
        register_mlp(name)(mlp_cls)
        cls.TYPES[name] = mlp_cls


__all__ = [
    "MLP",
    "TokenRoutedMLP",
    "MLPBase",
    "MLPConfig",
]
