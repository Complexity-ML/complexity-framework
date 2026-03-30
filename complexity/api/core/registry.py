"""
Registry API - Pour enregistrer des composants custom.
"""

from __future__ import annotations

from complexity.core import (
    Registry,
    ATTENTION_REGISTRY,
    MLP_REGISTRY,
    NORMALIZATION_REGISTRY,
    POSITION_REGISTRY,
    MODEL_REGISTRY,
    register_attention,
    register_mlp,
    register_normalization,
    register_position,
    register_model,
)


def register(component_type: str, name: str):
    """
    Décorateur pour enregistrer un composant custom.

    Usage:
        @register("attention", "my_attention")
        class MyAttention(AttentionBase):
            ...

        @register("mlp", "my_mlp")
        class MyMLP(MLPBase):
            ...
    """
    REGISTRIES = {
        "attention": register_attention,
        "mlp": register_mlp,
        "norm": register_normalization,
        "position": register_position,
        "model": register_model,
    }
    if component_type not in REGISTRIES:
        raise ValueError(f"Unknown component type: {component_type}. Use: {list(REGISTRIES.keys())}")
    return REGISTRIES[component_type](name)


__all__ = [
    "Registry",
    "register",
    "ATTENTION_REGISTRY",
    "MLP_REGISTRY",
    "NORMALIZATION_REGISTRY",
    "POSITION_REGISTRY",
    "MODEL_REGISTRY",
    "register_attention",
    "register_mlp",
    "register_normalization",
    "register_position",
    "register_model",
]
