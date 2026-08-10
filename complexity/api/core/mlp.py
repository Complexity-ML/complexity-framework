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
    TRHashEngineMLP as CoreTRHashEngineMLP,
)


class TokenRoutedMLP(nn.Module):
    """API wrapper around the core TR-Hash engine MoE.

    The public API returns ``(output, aux_loss)`` for compatibility with other
    MoE layers. Core model code should use ``complexity.core.mlp.TRHashEngineMLP``
    directly. Named ``TokenRoutedMLP`` for API stability — the underlying
    implementation is ``TRHashEngineMLP`` (the historical TokenRoutedMLP
    dispatch class was removed; see
    ``complexity.utils.token_routed_conversion`` for migrating old
    ``token_routed`` checkpoints).

    Unlike the removed implementation, ``token_ids`` is required: TR-Hash MoE
    always routes deterministically by token ID, there is no dense fallback.
    """

    def __init__(self, **kwargs):
        super().__init__()
        if "intermediate_size" not in kwargs:
            num_experts = kwargs.get("num_experts", 1)
            raw_width = int(kwargs["hidden_size"] * 8 / 3)
            # TR-Hash routed width must divide evenly across experts.
            kwargs["intermediate_size"] = -(-raw_width // num_experts) * num_experts
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

    Ce framework est recentré sur TR-Hash MoE (routing déterministe par ID de
    token / table de hachage) : les MLP denses autonomes (swiglu/geglu/standard)
    ont été retirés et seront réintroduits plus tard comme comparaisons
    explicites contre TR-Hash.

    Usage:
        mlp = MLP.create("moe", hidden_size=4096, num_experts=8)
        mlp = MLP.moe(hidden_size=4096, num_experts=8, top_k=2)
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
            num_experts = kwargs.get("num_experts", 1)
            raw_width = int(hidden_size * 8 / 3)
            # TR-Hash routed width must divide evenly across experts.
            kwargs["intermediate_size"] = -(-raw_width // num_experts) * num_experts
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
    def moe(cls, hidden_size: int, num_experts: int = 8, **kwargs) -> nn.Module:
        """Token-Routed MoE."""
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
