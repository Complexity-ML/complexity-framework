"""
Framework-Complexity Core Components
====================================

Modular building blocks for transformer architectures.

Modules:
- attention: MHA, GQA, MQA implementations
- mlp: TR-Hash MoE
- position: RoPE, YaRN, Dynamic NTK
- normalization: RMSNorm, LayerNorm
- registry: Component registration system

Usage:
    from complexity.core.registry import ATTENTION_REGISTRY, register_attention
    from complexity.core.attention import GroupedQueryAttention
    from complexity.core.mlp import TRHashEngineMLP
"""

# Registry (must be imported first)
from complexity.core.registry import (
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

# Import modules to register components
from complexity.core import attention
from complexity.core import mlp
from complexity.core import position
from complexity.core import normalization

# Re-export main classes
from complexity.core.attention import (
    AttentionBase,
    AttentionConfig,
    GroupedQueryAttention,
    MultiHeadAttention,
    MultiQueryAttention,
    RoutedGQA,
)

from complexity.core.mlp import (
    MLPBase,
    MLPConfig,
    TRHashEngineMLP,
)

from complexity.core.position import (
    RotaryEmbedding,
    StandardRoPE,
    YaRNRoPE,
    DynamicNTKRoPE,
    ALiBiPositionBias,
    LearnedPositionEmbedding,
    rotate_half,
    apply_rotary_pos_emb,
)

from complexity.core.normalization import (
    RMSNorm,
    LayerNorm,
    IdentityNorm,
    build_norm,
)


__all__ = [
    # Registry
    "Registry",
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
    # Attention
    "AttentionBase",
    "AttentionConfig",
    "GroupedQueryAttention",
    "MultiHeadAttention",
    "MultiQueryAttention",
    "RoutedGQA",
    # MLP
    "MLPBase",
    "MLPConfig",
    "TRHashEngineMLP",
    # Position
    "RotaryEmbedding",
    "StandardRoPE",
    "YaRNRoPE",
    "DynamicNTKRoPE",
    "ALiBiPositionBias",
    "LearnedPositionEmbedding",
    "rotate_half",
    "apply_rotary_pos_emb",
    # Normalization
    "RMSNorm",
    "LayerNorm",
    "IdentityNorm",
    "build_norm",
]
