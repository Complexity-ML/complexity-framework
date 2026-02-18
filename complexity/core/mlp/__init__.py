"""
MLP implementations for framework-complexity.

Available MLP types:
- standard / gelu: Standard FFN with GELU
- swiglu / silu / llama: SwiGLU (Llama-style)
- geglu: GeGLU variant
- token_routed / deterministic_moe / complexity: Token-Routed MoE (Complexity innovation)
- token_routed_parallel / batched_moe: Optimized batched version
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

"""

from .base import MLPBase, MLPConfig
from .standard import StandardMLP, SwiGLUMLP, GeGLUMLP
from .token_routed import TokenRoutedMLP, TokenRoutedMLPParallel, Expert

__all__ = [
    "MLPBase",
    "MLPConfig",
    "StandardMLP",
    "SwiGLUMLP",
    "GeGLUMLP",
    "TokenRoutedMLP",
    "TokenRoutedMLPParallel",
    "Expert",
]
