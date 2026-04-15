"""
Fused activation helpers with Liger Triton backend + pure-PyTorch fallback.

Used by:
  - TokenRoutedMLP (shared SwiGLU + per-expert SwiGLU)
  - any future SwiGLU/GEGLU MLP

``fused_silu_mul(gate, up)`` computes ``silu(gate) * up`` in a single Triton
kernel (forward + backward) on CUDA via Liger, or falls back to the naive
PyTorch ``F.silu(gate) * up`` on MPS / CPU.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _liger_silu_mul_available() -> bool:
    """Cached availability check."""
    if not hasattr(_liger_silu_mul_available, "_cache"):
        try:
            from liger_kernel.ops.swiglu import LigerSiLUMulFunction  # type: ignore[import-not-found] # noqa: F401
            _liger_silu_mul_available._cache = True
        except Exception:
            _liger_silu_mul_available._cache = False
    return _liger_silu_mul_available._cache


def fused_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Compute ``silu(gate) * up`` fused.

    On CUDA with ``liger-kernel`` installed, dispatches to
    ``LigerSiLUMulFunction`` which fuses forward and backward into single
    Triton kernels (saves ~2× memory traffic vs the naive
    ``F.silu(gate) * up`` path).

    Falls back to the naive path on MPS/CPU or when Liger is missing.
    Gradient semantics are identical.
    """
    if gate.is_cuda and _liger_silu_mul_available():
        from liger_kernel.ops.swiglu import LigerSiLUMulFunction  # type: ignore[import-not-found]
        return LigerSiLUMulFunction.apply(gate, up)
    return F.silu(gate) * up
