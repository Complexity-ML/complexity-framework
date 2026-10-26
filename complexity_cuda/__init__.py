"""CUDA/Triton primitives used by the canonical Complexity architecture.

Construct TR-HASH models through ``complexity`` or ``complexity.tr_hash``.
This package supplies kernels; it does not define another routing architecture.
"""

# Check for Triton availability
try:
    import triton as triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# CGGR primitives. Expert assignments are supplied by the canonical engine.
from .triton_token_routed import (
    sort_tokens_by_expert,
    fused_swiglu_triton,
    fused_rmsnorm,
)

# Fused QK Norm + Flash Attention
from .fused_attention import (
    fused_qk_rmsnorm,
    flash_attention_triton,
    fused_qknorm_flash_attention,
    FusedQKNormAttention,
)

# Fused RMSNorm + MLP Projections
from .fused_mlp import (
    fused_rmsnorm_gate_up,
    fused_swiglu_down,
    fused_mlp,
    FusedSwiGLUMLP,
    FusedMLP,
)

# Persistent CGGR
from .persistent_cggr import (
    sort_tokens_by_expert_fast,
    persistent_swiglu_cggr,
)

# INT8/FP8 Quantization
from .quantization import (
    QuantType,
    dynamic_quantize_int8,
    int8_gemm,
    fused_quantize_gemm,
    QuantizedLinear,
    quantize_model,
)

# Fused Residual + RMSNorm
from .fused_residual import (
    fused_residual_rmsnorm,
    fused_residual_rmsnorm_dropout,
    fused_add_rmsnorm_inplace,
    FusedResidualRMSNorm,
)

__all__ = [
    # Triton flag
    "HAS_TRITON",

    # === CGGR primitives ===
    "sort_tokens_by_expert",
    "fused_swiglu_triton",
    "fused_rmsnorm",

    # === Fused Attention ===
    "fused_qk_rmsnorm",
    "flash_attention_triton",
    "fused_qknorm_flash_attention",
    "FusedQKNormAttention",

    # === Fused MLP ===
    "fused_rmsnorm_gate_up",
    "fused_swiglu_down",
    "fused_mlp",
    "FusedSwiGLUMLP",
    "FusedMLP",

    # === Persistent CGGR ===
    "sort_tokens_by_expert_fast",
    "persistent_swiglu_cggr",

    # === Quantization ===
    "QuantType",
    "dynamic_quantize_int8",
    "int8_gemm",
    "fused_quantize_gemm",
    "QuantizedLinear",
    "quantize_model",

    # === Fused Residual ===
    "fused_residual_rmsnorm",
    "fused_residual_rmsnorm_dropout",
    "fused_add_rmsnorm_inplace",
    "FusedResidualRMSNorm",
]


def get_optimization_info() -> dict:
    """Get information about available optimizations."""
    return {
        "triton_available": HAS_TRITON,
        "optimizations": {
            "fused_qk_attention": {
                "description": "Fused QK Normalization + Flash Attention",
                "speedup": "15-20%",
                "available": HAS_TRITON,
            },
            "fused_mlp": {
                "description": "Fused RMSNorm + Gate/Up/Down Projections",
                "speedup": "20-30%",
                "available": HAS_TRITON,
            },
            "persistent_cggr": {
                "description": "Persistent kernels for Token-Routed MLP",
                "speedup": "10-15%",
                "available": HAS_TRITON,
            },
            "int8_quantization": {
                "description": "INT8 quantization with fused GEMM",
                "speedup": "40-50% throughput",
                "memory_reduction": "50%",
                "available": HAS_TRITON,
            },
            "fused_residual": {
                "description": "Fused Residual + RMSNorm",
                "speedup": "5-10%",
                "available": HAS_TRITON,
            },
        }
    }
