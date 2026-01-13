"""
Core Building Blocks - Pour construire des modèles maison.
==========================================================

Expose tous les composants du framework pour les builders.

Usage:
    from complexity.api.core import (
        # Attention
        Attention, GQA, MHA, MQA,
        # MLP
        MLP, SwiGLU, GeGLU, TokenRoutedMLP,
        # Position
        Position, RoPE, YaRN, ALiBi,
        # Normalization
        Norm, RMSNorm, LayerNorm,
        # Blocks
        Block, TransformerBlock,
        # Architectures INL (O(N))
        Architecture, Mamba, RWKV, RetNet,
        # CUDA / Triton Optimizations
        CUDA, Triton, FlashAttention, SlidingWindowAttention,
        # Registry
        Registry, register,
    )

    # Exemple CUDA/Triton
    flash_attn = CUDA.flash(hidden_size=4096, num_heads=32)
    sliding_attn = CUDA.sliding_window(hidden_size=4096, num_heads=32, window_size=4096)
    sparse_attn = CUDA.sparse(hidden_size=4096, num_heads=32)
    linear_attn = CUDA.linear(hidden_size=4096, num_heads=32)
"""

# Attention
from .attention import (
    Attention,
    GQA,
    MHA,
    MQA,
    GroupedQueryAttention,
    MultiHeadAttention,
    MultiQueryAttention,
    AttentionBase,
    AttentionConfig,
)

# MLP
from .mlp import (
    MLP,
    SwiGLU,
    GeGLU,
    SwiGLUMLP,
    GeGLUMLP,
    StandardMLP,
    TokenRoutedMLP,
    TokenRoutedMLPParallel,
    SparseMoE,
    SparseMoEParallel,
    SparseMoEConfig,
    MLPBase,
    MLPConfig,
)

# Position
from .position import (
    Position,
    RoPE,
    YaRN,
    ALiBi,
    StandardRoPE,
    YaRNRoPE,
    DynamicNTKRoPE,
    ALiBiPositionBias,
    LearnedPositionEmbedding,
    RotaryEmbedding,
    rotate_half,
    apply_rotary_pos_emb,
)

# Normalization
from .norm import (
    Norm,
    RMSNorm,
    LayerNorm,
    IdentityNorm,
    build_norm,
)

# Block & Model
from .block import (
    Block,
    TransformerBlock,
    ComplexityModel,
    ModelConfig,
)

# Architectures INL (O(N) alternatives)
from .architecture import (
    Architecture,
    Mamba,
    MambaBlock,
    MambaConfig,
    RWKV,
    RWKVBlock,
    RWKVConfig,
    RetNet,
    RetNetBlock,
    RetNetConfig,
    MoDBlock,
    MoDConfig,
)

# Registry
from .registry import (
    Registry,
    register,
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

# CUDA / Triton Optimizations
from .cuda import (
    CUDA,
    Triton,
    FlashAttention,
    SlidingWindowAttention,
    SparseAttention,
    LinearAttention,
    MultiScaleAttention,
)

# Helpers
from .helpers import (
    Helpers,
    Init,
    Mask,
    KVCache,
    SlidingWindowCache,
    Sampling,
    Tensors,
    Debug,
    Checkpointing,
)

# Efficient (small budget)
from .efficient import (
    Efficient,
    Quantize,
    QuantConfig,
    QuantizedLinear,
    MixedPrecision,
    MemoryEfficient,
    SmallModels,
)

# INL Dynamics (velocity tracking for stability)
from complexity.core.dynamics import (
    INLDynamics,
    INLDynamicsLite,
    DynamicsConfig,
)


__all__ = [
    # ========== Factories ==========
    "Attention",
    "MLP",
    "Position",
    "Norm",
    "Block",
    "Architecture",

    # ========== Attention ==========
    "GQA",
    "MHA",
    "MQA",
    "GroupedQueryAttention",
    "MultiHeadAttention",
    "MultiQueryAttention",
    "AttentionBase",
    "AttentionConfig",

    # ========== MLP ==========
    "SwiGLU",
    "GeGLU",
    "SwiGLUMLP",
    "GeGLUMLP",
    "StandardMLP",
    "TokenRoutedMLP",
    "TokenRoutedMLPParallel",
    "SparseMoE",
    "SparseMoEParallel",
    "SparseMoEConfig",
    "MLPBase",
    "MLPConfig",

    # ========== Position ==========
    "RoPE",
    "YaRN",
    "ALiBi",
    "StandardRoPE",
    "YaRNRoPE",
    "DynamicNTKRoPE",
    "ALiBiPositionBias",
    "LearnedPositionEmbedding",
    "RotaryEmbedding",
    "rotate_half",
    "apply_rotary_pos_emb",

    # ========== Normalization ==========
    "RMSNorm",
    "LayerNorm",
    "IdentityNorm",
    "build_norm",

    # ========== Block & Model ==========
    "TransformerBlock",
    "ComplexityModel",
    "ModelConfig",

    # ========== INL Architectures (O(N)) ==========
    "Mamba",
    "MambaBlock",
    "MambaConfig",
    "RWKV",
    "RWKVBlock",
    "RWKVConfig",
    "RetNet",
    "RetNetBlock",
    "RetNetConfig",
    "MoDBlock",
    "MoDConfig",

    # ========== Registry ==========
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

    # ========== CUDA / Triton ==========
    "CUDA",
    "Triton",
    "FlashAttention",
    "SlidingWindowAttention",
    "SparseAttention",
    "LinearAttention",
    "MultiScaleAttention",

    # ========== Helpers ==========
    "Helpers",
    "Init",
    "Mask",
    "KVCache",
    "SlidingWindowCache",
    "Sampling",
    "Tensors",
    "Debug",
    "Checkpointing",

    # ========== Efficient (Small Budget) ==========
    "Efficient",
    "Quantize",
    "QuantConfig",
    "QuantizedLinear",
    "MixedPrecision",
    "MemoryEfficient",
    "SmallModels",

    # ========== INL Dynamics ==========
    "INLDynamics",
    "INLDynamicsLite",
    "DynamicsConfig",
]
