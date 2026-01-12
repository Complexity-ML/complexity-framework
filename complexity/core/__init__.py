"""
Framework-Complexity Core Components
====================================

Modular building blocks for transformer architectures.

Modules:
- attention: MHA, GQA, MQA implementations
- mlp: Standard, SwiGLU, Token-Routed MoE
- position: RoPE, YaRN, Dynamic NTK
- normalization: RMSNorm, LayerNorm
- registry: Component registration system

Usage:
    from complexity.core.registry import ATTENTION_REGISTRY, register_attention
    from complexity.core.attention import GroupedQueryAttention
    from complexity.core.mlp import SwiGLUMLP
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
)

from complexity.core.mlp import (
    MLPBase,
    MLPConfig,
    StandardMLP,
    SwiGLUMLP,
    GeGLUMLP,
    TokenRoutedMLP,
    TokenRoutedMLPParallel,
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

# INL Dynamics (velocity tracking for stability)
from complexity.core.dynamics import (
    INLDynamics,
    INLDynamicsLite,
    DynamicsConfig,
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
    # MLP
    "MLPBase",
    "MLPConfig",
    "StandardMLP",
    "SwiGLUMLP",
    "GeGLUMLP",
    "TokenRoutedMLP",
    "TokenRoutedMLPParallel",
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
    # INL Dynamics
    "INLDynamics",
    "INLDynamicsLite",
    "DynamicsConfig",
]
