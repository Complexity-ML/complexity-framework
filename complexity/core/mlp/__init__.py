"""
MLP implementations for framework-complexity.

Available MLP types:
- standard / gelu: Standard FFN with GELU
- swiglu / silu / llama: SwiGLU (Llama-style)
- geglu: GeGLU variant
- token_routed / deterministic_moe / complexity: Token-Routed MoE (Complexity innovation)
- token_routed_parallel / batched_moe: Optimized batched version
- sparse_moe / learned_moe / mixtral: Standard MoE with learned routing
- sparse_moe_parallel: Optimized batched SparseMoE

Usage:
    from complexity.core.mlp import SwiGLUMLP, MLPConfig
    from complexity.core.registry import MLP_REGISTRY

    # Direct instantiation
    config = MLPConfig(hidden_size=768, intermediate_size=3072)
    mlp = SwiGLUMLP(config)

    # Via registry
    mlp = MLP_REGISTRY.build("swiglu", config)

    # Token-Routed MoE (deterministic, our innovation)
    config = MLPConfig(hidden_size=768, intermediate_size=3072, num_experts=4)
    moe = MLP_REGISTRY.build("token_routed", config)

    # SparseMoE (learned routing, standard approach)
    moe = MLP_REGISTRY.build("sparse_moe", config)
"""

from .base import MLPBase, MLPConfig
from .standard import StandardMLP, SwiGLUMLP, GeGLUMLP
from .token_routed import TokenRoutedMLP, TokenRoutedMLPParallel, Expert
from .sparse_moe import SparseMoE, SparseMoEParallel, SparseMoEConfig

__all__ = [
    "MLPBase",
    "MLPConfig",
    "StandardMLP",
    "SwiGLUMLP",
    "GeGLUMLP",
    "TokenRoutedMLP",
    "TokenRoutedMLPParallel",
    "Expert",
    "SparseMoE",
    "SparseMoEParallel",
    "SparseMoEConfig",
]
