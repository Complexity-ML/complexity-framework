"""
Fused Triton kernels for training acceleration.

1. Fused RMSNorm — single kernel instead of 3 (variance, rsqrt, scale)
2. Fused SiLU × gate + up — SwiGLU activation in one kernel
3. Fused Residual + RMSNorm — combine residual add + normalization

Each kernel saves 1-3 kernel launches and reduces memory round-trips.
Typical speedup: 1.3-2x per operation.

Complexity-ML — 2026
"""

import torch
import torch.nn as nn
import warnings

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# These kernels hard-cast their output to tl.float16. Running them on a
# bfloat16 (or fp32) tensor silently corrupts the dtype, breaking mixed-
# precision training. We gate on fp16 and warn once for other dtypes so
# users notice instead of diverging silently.
_FUSED_DTYPE_WARNED = False

def _fused_kernel_compatible(x: torch.Tensor, kernel_name: str) -> bool:
    global _FUSED_DTYPE_WARNED
    if x.dtype == torch.float16:
        return True
    if not _FUSED_DTYPE_WARNED:
        warnings.warn(
            f"complexity.gpu.fused_kernels: {kernel_name} received dtype={x.dtype}. "
            f"These Triton kernels hard-cast to fp16 — falling back to the PyTorch "
            f"implementation to preserve dtype. Use fp16 inputs (or the non-fused "
            f"RMSNorm / F.silu*up) to avoid this fallback.",
            RuntimeWarning, stacklevel=3,
        )
        _FUSED_DTYPE_WARNED = True
    return False


# =============================================================================
# 1. Fused RMSNorm
# =============================================================================

if HAS_TRITON:
    @triton.jit
    def _rmsnorm_fwd_kernel(
        X, W, Y,
        stride_x, stride_y,
        N,
        eps: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Fused RMSNorm: y = x * rsqrt(mean(x²) + eps) * weight"""
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N

        x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)

        # RMS = sqrt(mean(x²))
        x_sq = x * x
        mean_sq = tl.sum(x_sq, axis=0) / N
        rrms = 1.0 / tl.sqrt(mean_sq + eps)

        y = (x * rrms * w).to(tl.float16)  # or bfloat16
        tl.store(Y + row * stride_y + cols, y, mask=mask)


    @triton.jit
    def _residual_rmsnorm_fwd_kernel(
        X, RESIDUAL, W, Y, RESIDUAL_OUT,
        stride_x, stride_y,
        N,
        eps: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Fused Residual + RMSNorm: residual = x + residual; y = rmsnorm(residual)"""
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N

        x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        res = tl.load(RESIDUAL + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)

        # Residual add
        hidden = x + res
        tl.store(RESIDUAL_OUT + row * stride_x + cols, hidden.to(tl.float16), mask=mask)

        # RMSNorm
        x_sq = hidden * hidden
        mean_sq = tl.sum(x_sq, axis=0) / N
        rrms = 1.0 / tl.sqrt(mean_sq + eps)

        y = (hidden * rrms * w).to(tl.float16)
        tl.store(Y + row * stride_y + cols, y, mask=mask)


    @triton.jit
    def _swiglu_fwd_kernel(
        GATE, UP, OUT,
        stride,
        N,
        BLOCK_N: tl.constexpr,
    ):
        """Fused SiLU(gate) × up — SwiGLU activation in one kernel."""
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N

        gate = tl.load(GATE + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(UP + row * stride + cols, mask=mask, other=0.0).to(tl.float32)

        # SiLU(gate) = gate * sigmoid(gate)
        sigmoid_gate = 1.0 / (1.0 + tl.exp(-gate))
        silu_gate = gate * sigmoid_gate

        out = (silu_gate * up).to(tl.float16)
        tl.store(OUT + row * stride + cols, out, mask=mask)


def _next_power_of_2(n):
    p = 1
    while p < n:
        p *= 2
    return p


class FusedRMSNorm(nn.Module):
    """
    RMSNorm with Triton-fused forward pass.

    3 CUDA kernels → 1 Triton kernel.
    Falls back to PyTorch if Triton unavailable.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.hidden_size = hidden_size

    def _triton_forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_2d = x.view(-1, self.hidden_size)
        M = x_2d.shape[0]
        y = torch.empty_like(x_2d)

        BLOCK_N = _next_power_of_2(self.hidden_size)
        _rmsnorm_fwd_kernel[(M,)](
            x_2d, self.weight, y,
            x_2d.stride(0), y.stride(0),
            self.hidden_size,
            eps=self.eps,
            BLOCK_N=BLOCK_N,
        )
        return y.view(shape)

    def _pytorch_forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (HAS_TRITON and x.is_cuda and not torch.is_grad_enabled()
                and _fused_kernel_compatible(x, "FusedRMSNorm")):
            return self._triton_forward(x)
        return self._pytorch_forward(x)


def fused_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Fused SiLU(gate) × up in a single Triton kernel.

    Replaces: F.silu(gate) * up (2 kernels + 1 intermediate tensor)
    With: 1 kernel, no intermediate.
    """
    if not HAS_TRITON or not gate.is_cuda or not _fused_kernel_compatible(gate, "fused_swiglu"):
        return torch.nn.functional.silu(gate) * up

    shape = gate.shape
    gate_2d = gate.view(-1, shape[-1])
    up_2d = up.view(-1, shape[-1])
    out = torch.empty_like(gate_2d)
    M = gate_2d.shape[0]
    N = gate_2d.shape[1]

    BLOCK_N = _next_power_of_2(N)
    _swiglu_fwd_kernel[(M,)](
        gate_2d, up_2d, out,
        gate_2d.stride(0),
        N,
        BLOCK_N=BLOCK_N,
    )
    return out.view(shape)


def fused_residual_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple:
    """
    Fused residual + RMSNorm in a single kernel.

    Replaces:
        residual = x + residual  (1 kernel)
        y = rmsnorm(residual)    (3 kernels)
    With: 1 kernel.

    Returns: (normed_output, updated_residual)
    """
    if not HAS_TRITON or not x.is_cuda or not _fused_kernel_compatible(x, "fused_residual_rmsnorm"):
        residual = x + residual
        variance = residual.to(torch.float32).pow(2).mean(-1, keepdim=True)
        normed = residual * torch.rsqrt(variance + eps)
        return weight * normed, residual

    shape = x.shape
    N = shape[-1]
    x_2d = x.view(-1, N)
    res_2d = residual.view(-1, N)
    M = x_2d.shape[0]

    y = torch.empty_like(x_2d)
    res_out = torch.empty_like(x_2d)

    BLOCK_N = _next_power_of_2(N)
    _residual_rmsnorm_fwd_kernel[(M,)](
        x_2d, res_2d, weight, y, res_out,
        x_2d.stride(0), y.stride(0),
        N,
        eps=eps,
        BLOCK_N=BLOCK_N,
    )
    return y.view(shape), res_out.view(shape)
