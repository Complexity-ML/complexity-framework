"""
Position embedding implementations for framework-complexity.

Available position types:
- rope / standard: Standard Rotary Position Embedding
- yarn: YaRN for extended context
- dynamic_ntk: Dynamic NTK-aware scaling
- alibi: ALiBi (Attention with Linear Biases)
- learned: Classic learned position embeddings

Usage:
    from complexity.core.position import StandardRoPE, ALiBiPositionBias
    from complexity.core.registry import POSITION_REGISTRY

    # Direct instantiation
    rope = StandardRoPE(dim=64, max_seq_len=2048)
    alibi = ALiBiPositionBias(num_heads=32)

    # Via registry
    rope = POSITION_REGISTRY.build("yarn", dim=64, max_seq_len=8192)
"""

from .rotary import (
    RotaryEmbedding,
    StandardRoPE,
    YaRNRoPE,
    DynamicNTKRoPE,
    ALiBiPositionBias,
    LearnedPositionEmbedding,
    rotate_half,
    apply_rotary_pos_emb,
)

__all__ = [
    "RotaryEmbedding",
    "StandardRoPE",
    "YaRNRoPE",
    "DynamicNTKRoPE",
    "ALiBiPositionBias",
    "LearnedPositionEmbedding",
    "rotate_half",
    "apply_rotary_pos_emb",
]
