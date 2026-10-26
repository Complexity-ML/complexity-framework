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
# Import modules to register components
from complexity.core import attention as attention
from complexity.core import mlp as mlp
from complexity.core import normalization as normalization
from complexity.core import position as position

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
from complexity.core.normalization import (
    IdentityNorm,
    LayerNorm,
    RMSNorm,
    build_norm,
)
from complexity.core.position import (
    ALiBiPositionBias,
    DynamicNTKRoPE,
    LearnedPositionEmbedding,
    RotaryEmbedding,
    StandardRoPE,
    YaRNRoPE,
    apply_rotary_pos_emb,
    rotate_half,
)
from complexity.core.registry import (
    ATTENTION_REGISTRY,
    MLP_REGISTRY,
    MODEL_REGISTRY,
    NORMALIZATION_REGISTRY,
    POSITION_REGISTRY,
    Registry,
    register_attention,
    register_mlp,
    register_model,
    register_normalization,
    register_position,
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
