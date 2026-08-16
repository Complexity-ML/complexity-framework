"""
Normalization layers for framework-complexity.

Available types:
- rmsnorm / rms / llama: RMS Normalization (efficient, modern)
- layernorm / ln / gpt: Standard Layer Normalization
- identity / none: No normalization (passthrough)

Usage:
    from complexity.core.normalization import RMSNorm, LayerNorm
    from complexity.core.registry import NORMALIZATION_REGISTRY

    # Direct
    norm = RMSNorm(hidden_size=768)

    # Via registry
    norm = NORMALIZATION_REGISTRY.build("rmsnorm", hidden_size=768)
"""

from .norms import RMSNorm, LayerNorm, IdentityNorm, build_norm

__all__ = [
    "RMSNorm",
    "LayerNorm",
    "IdentityNorm",
    "build_norm",
]
