"""
Fused Residual + RMSNorm Kernels

Optimizations:
1. Single kernel for residual addition + RMSNorm
2. Eliminates memory round-trip between residual add and norm
3. Fused dropout (optional)
4. In-place computation where possible

Performance: ~5-10% faster than separate operations

Author: Pacific Prime
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# =============================================================================
# FUSED RESIDUAL + RMSNORM KERNELS
# =============================================================================

if HAS_TRITON:
    @triton.jit
    def _fused_residual_rmsnorm_kernel(
        # Inputs
        x_ptr,          # Input to add [batch*seq, hidden]
        residual_ptr,   # Residual connection [batch*seq, hidden]
        weight_ptr,     # RMSNorm weight [hidden]
        # Outputs
        out_ptr,        # Normalized output [batch*seq, hidden]
        residual_out_ptr,  # Updated residual (x + residual) [batch*seq, hidden]
        # Dimensions
        batch_seq,
        hidden_size,
        # Strides
        stride_x_row, stride_x_col,
        stride_r_row, stride_r_col,
        stride_o_row, stride_o_col,
        stride_ro_row, stride_ro_col,
        # Epsilon
        eps,
        # Options
        STORE_RESIDUAL: tl.constexpr,  # Whether to store updated residual
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused Residual Add + RMSNorm kernel.

        Computes:
            hidden = x + residual
            out = hidden / sqrt(mean(hidden^2) + eps) * weight

        If STORE_RESIDUAL, also stores the hidden state for next layer's residual.
        """
        row_idx = tl.program_id(0)
        if row_idx >= batch_seq:
            return

        # First pass: compute residual sum and sum of squares
        sum_sq = 0.0

        for col in range(0, hidden_size, BLOCK_SIZE):
            col_offs = col + tl.arange(0, BLOCK_SIZE)
            mask = col_offs < hidden_size

            # Load x and residual
            x = tl.load(
                x_ptr + row_idx * stride_x_row + col_offs * stride_x_col,
                mask=mask,
                other=0.0
            )
            residual = tl.load(
                residual_ptr + row_idx * stride_r_row + col_offs * stride_r_col,
                mask=mask,
                other=0.0
            )

            # Add residual
            hidden = x + residual

            # Accumulate sum of squares
            sum_sq += tl.sum(hidden * hidden, axis=0)

            # Store updated residual if needed
            if STORE_RESIDUAL:
                tl.store(
                    residual_out_ptr + row_idx * stride_ro_row + col_offs * stride_ro_col,
                    hidden,
                    mask=mask
                )

        # Compute inverse RMS
        inv_rms = 1.0 / tl.sqrt(sum_sq / hidden_size + eps)

        # Second pass: normalize and apply weight
        for col in range(0, hidden_size, BLOCK_SIZE):
            col_offs = col + tl.arange(0, BLOCK_SIZE)
            mask = col_offs < hidden_size

            # Load x and residual again (or from stored residual)
            if STORE_RESIDUAL:
                hidden = tl.load(
                    residual_out_ptr + row_idx * stride_ro_row + col_offs * stride_ro_col,
                    mask=mask,
                    other=0.0
                )
            else:
                x = tl.load(
                    x_ptr + row_idx * stride_x_row + col_offs * stride_x_col,
                    mask=mask,
                    other=0.0
                )
                residual = tl.load(
                    residual_ptr + row_idx * stride_r_row + col_offs * stride_r_col,
                    mask=mask,
                    other=0.0
                )
                hidden = x + residual

            # Load weight
            weight = tl.load(weight_ptr + col_offs, mask=mask, other=1.0)

            # Normalize
            out = hidden * inv_rms * weight

            # Store output
            tl.store(
                out_ptr + row_idx * stride_o_row + col_offs * stride_o_col,
                out,
                mask=mask
            )


    @triton.jit
    def _fused_residual_rmsnorm_dropout_kernel(
        # Inputs
        x_ptr,
        residual_ptr,
        weight_ptr,
        # Outputs
        out_ptr,
        residual_out_ptr,
        # Random seed for dropout
        seed,
        # Dimensions
        batch_seq,
        hidden_size,
        # Strides
        stride_x_row, stride_x_col,
        stride_r_row, stride_r_col,
        stride_o_row, stride_o_col,
        stride_ro_row, stride_ro_col,
        # Parameters
        eps,
        dropout_p,
        # Options
        TRAINING: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused Residual + Dropout + RMSNorm kernel.

        Computes:
            x = dropout(x, p)  # Only in training
            hidden = x + residual
            out = rmsnorm(hidden)
        """
        row_idx = tl.program_id(0)
        if row_idx >= batch_seq:
            return

        # Dropout scale
        scale = 1.0 / (1.0 - dropout_p) if TRAINING and dropout_p > 0 else 1.0

        # First pass: apply dropout, add residual, compute sum of squares
        sum_sq = 0.0

        for col in range(0, hidden_size, BLOCK_SIZE):
            col_offs = col + tl.arange(0, BLOCK_SIZE)
            mask = col_offs < hidden_size

            x = tl.load(
                x_ptr + row_idx * stride_x_row + col_offs * stride_x_col,
                mask=mask,
                other=0.0
            )

            # Apply dropout
            if TRAINING and dropout_p > 0:
                # Generate random mask
                random_offset = row_idx * hidden_size + col_offs
                random_vals = tl.rand(seed, random_offset)
                dropout_mask = random_vals > dropout_p
                x = tl.where(dropout_mask, x * scale, 0.0)

            residual = tl.load(
                residual_ptr + row_idx * stride_r_row + col_offs * stride_r_col,
                mask=mask,
                other=0.0
            )

            hidden = x + residual
            sum_sq += tl.sum(hidden * hidden, axis=0)

            # Store updated residual
            tl.store(
                residual_out_ptr + row_idx * stride_ro_row + col_offs * stride_ro_col,
                hidden,
                mask=mask
            )

        inv_rms = 1.0 / tl.sqrt(sum_sq / hidden_size + eps)

        # Second pass: normalize
        for col in range(0, hidden_size, BLOCK_SIZE):
            col_offs = col + tl.arange(0, BLOCK_SIZE)
            mask = col_offs < hidden_size

            hidden = tl.load(
                residual_out_ptr + row_idx * stride_ro_row + col_offs * stride_ro_col,
                mask=mask,
                other=0.0
            )
            weight = tl.load(weight_ptr + col_offs, mask=mask, other=1.0)

            out = hidden * inv_rms * weight

            tl.store(
                out_ptr + row_idx * stride_o_row + col_offs * stride_o_col,
                out,
                mask=mask
            )


    @triton.jit
    def _fused_add_rmsnorm_inplace_kernel(
        # Input/Output (in-place)
        hidden_ptr,     # [batch*seq, hidden] - will be modified
        # Additional input
        mlp_out_ptr,    # [batch*seq, hidden] - MLP output to add
        weight_ptr,     # [hidden]
        # Output
        out_ptr,        # [batch*seq, hidden] - normalized output
        # Dimensions
        batch_seq,
        hidden_size,
        # Strides
        stride_h_row, stride_h_col,
        stride_m_row, stride_m_col,
        stride_o_row, stride_o_col,
        # Epsilon
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        In-place residual add + RMSNorm.

        Modifies hidden in place: hidden = hidden + mlp_out
        Then outputs: out = rmsnorm(hidden)

        Useful for transformer layers where hidden state is passed through.
        """
        row_idx = tl.program_id(0)
        if row_idx >= batch_seq:
            return

        # First pass: add and compute sum of squares
        sum_sq = 0.0

        for col in range(0, hidden_size, BLOCK_SIZE):
            col_offs = col + tl.arange(0, BLOCK_SIZE)
            mask = col_offs < hidden_size

            hidden = tl.load(
                hidden_ptr + row_idx * stride_h_row + col_offs * stride_h_col,
                mask=mask,
                other=0.0
            )
            mlp_out = tl.load(
                mlp_out_ptr + row_idx * stride_m_row + col_offs * stride_m_col,
                mask=mask,
                other=0.0
            )

            # In-place add
            hidden = hidden + mlp_out
            sum_sq += tl.sum(hidden * hidden, axis=0)

            # Store back
            tl.store(
                hidden_ptr + row_idx * stride_h_row + col_offs * stride_h_col,
                hidden,
                mask=mask
            )

        inv_rms = 1.0 / tl.sqrt(sum_sq / hidden_size + eps)

        # Second pass: normalize
        for col in range(0, hidden_size, BLOCK_SIZE):
            col_offs = col + tl.arange(0, BLOCK_SIZE)
            mask = col_offs < hidden_size

            hidden = tl.load(
                hidden_ptr + row_idx * stride_h_row + col_offs * stride_h_col,
                mask=mask,
                other=0.0
            )
            weight = tl.load(weight_ptr + col_offs, mask=mask, other=1.0)

            out = hidden * inv_rms * weight

            tl.store(
                out_ptr + row_idx * stride_o_row + col_offs * stride_o_col,
                out,
                mask=mask
            )


# =============================================================================
# PYTHON WRAPPERS
# =============================================================================

def fused_residual_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    store_residual: bool = True,
) -> tuple:
    """
    Fused Residual Add + RMSNorm.

    Args:
        x: Input tensor [batch, seq, hidden] or [batch*seq, hidden]
        residual: Residual tensor (same shape as x)
        weight: RMSNorm weight [hidden]
        eps: Epsilon for numerical stability
        store_residual: Whether to return updated residual

    Returns:
        out: Normalized output
        residual_out: Updated residual (x + residual) if store_residual
    """
    original_shape = x.shape
    if x.dim() == 3:
        batch_seq = x.shape[0] * x.shape[1]
        hidden_size = x.shape[2]
        x = x.view(batch_seq, hidden_size)
        residual = residual.view(batch_seq, hidden_size)
    else:
        batch_seq, hidden_size = x.shape

    if not HAS_TRITON or not x.is_cuda:
        # PyTorch fallback
        hidden = x + residual
        rms = torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + eps)
        out = hidden * rms * weight
        if len(original_shape) == 3:
            out = out.view(original_shape)
            hidden = hidden.view(original_shape)
        return (out, hidden) if store_residual else (out,)

    x = x.contiguous()
    residual = residual.contiguous()

    out = torch.empty_like(x)
    residual_out = torch.empty_like(x) if store_residual else x  # Dummy if not storing

    BLOCK_SIZE = min(1024, triton.next_power_of_2(hidden_size))

    _fused_residual_rmsnorm_kernel[(batch_seq,)](
        x, residual, weight,
        out, residual_out,
        batch_seq, hidden_size,
        x.stride(0), x.stride(1),
        residual.stride(0), residual.stride(1),
        out.stride(0), out.stride(1),
        residual_out.stride(0), residual_out.stride(1),
        eps,
        STORE_RESIDUAL=store_residual,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    if len(original_shape) == 3:
        out = out.view(original_shape)
        if store_residual:
            residual_out = residual_out.view(original_shape)

    return (out, residual_out) if store_residual else (out,)


def fused_residual_rmsnorm_dropout(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    dropout_p: float = 0.0,
    eps: float = 1e-6,
    training: bool = True,
) -> tuple:
    """
    Fused Residual + Dropout + RMSNorm.

    Args:
        x: Input tensor [batch, seq, hidden]
        residual: Residual tensor
        weight: RMSNorm weight
        dropout_p: Dropout probability
        eps: Epsilon
        training: Whether in training mode

    Returns:
        out: Normalized output
        residual_out: Updated residual
    """
    original_shape = x.shape
    if x.dim() == 3:
        batch_seq = x.shape[0] * x.shape[1]
        hidden_size = x.shape[2]
        x = x.view(batch_seq, hidden_size)
        residual = residual.view(batch_seq, hidden_size)
    else:
        batch_seq, hidden_size = x.shape

    if not HAS_TRITON or not x.is_cuda:
        # PyTorch fallback
        if training and dropout_p > 0:
            x = F.dropout(x, p=dropout_p, training=True)
        hidden = x + residual
        rms = torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + eps)
        out = hidden * rms * weight
        if len(original_shape) == 3:
            out = out.view(original_shape)
            hidden = hidden.view(original_shape)
        return out, hidden

    x = x.contiguous()
    residual = residual.contiguous()

    out = torch.empty_like(x)
    residual_out = torch.empty_like(x)

    # Random seed for dropout
    seed = torch.randint(0, 2**31, (1,), device=x.device).item()

    BLOCK_SIZE = min(1024, triton.next_power_of_2(hidden_size))

    _fused_residual_rmsnorm_dropout_kernel[(batch_seq,)](
        x, residual, weight,
        out, residual_out,
        seed,
        batch_seq, hidden_size,
        x.stride(0), x.stride(1),
        residual.stride(0), residual.stride(1),
        out.stride(0), out.stride(1),
        residual_out.stride(0), residual_out.stride(1),
        eps, dropout_p,
        TRAINING=training,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    if len(original_shape) == 3:
        out = out.view(original_shape)
        residual_out = residual_out.view(original_shape)

    return out, residual_out


def fused_add_rmsnorm_inplace(
    hidden: torch.Tensor,
    mlp_out: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    In-place residual add + RMSNorm.

    Modifies hidden in-place!

    Args:
        hidden: Hidden state (modified in-place) [batch, seq, hidden]
        mlp_out: MLP output to add
        weight: RMSNorm weight

    Returns:
        out: Normalized output (hidden is also modified)
    """
    original_shape = hidden.shape
    if hidden.dim() == 3:
        batch_seq = hidden.shape[0] * hidden.shape[1]
        hidden_size = hidden.shape[2]
        hidden = hidden.view(batch_seq, hidden_size)
        mlp_out = mlp_out.view(batch_seq, hidden_size)
    else:
        batch_seq, hidden_size = hidden.shape

    if not HAS_TRITON or not hidden.is_cuda:
        # PyTorch fallback
        hidden.add_(mlp_out)
        rms = torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + eps)
        out = hidden * rms * weight
        if len(original_shape) == 3:
            out = out.view(original_shape)
        return out

    mlp_out = mlp_out.contiguous()
    out = torch.empty_like(hidden)

    BLOCK_SIZE = min(1024, triton.next_power_of_2(hidden_size))

    _fused_add_rmsnorm_inplace_kernel[(batch_seq,)](
        hidden, mlp_out, weight, out,
        batch_seq, hidden_size,
        hidden.stride(0), hidden.stride(1),
        mlp_out.stride(0), mlp_out.stride(1),
        out.stride(0), out.stride(1),
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    if len(original_shape) == 3:
        out = out.view(original_shape)

    return out


# =============================================================================
# FUSED LAYER NORM MODULE
# =============================================================================

class FusedResidualRMSNorm(nn.Module):
    """
    Fused Residual + RMSNorm layer.

    Combines residual addition with normalization in a single kernel.
    ~5-10% faster than separate operations.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.dropout_p = dropout_p

        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple:
        """
        Forward pass.

        Args:
            x: Input [batch, seq, hidden]
            residual: Residual connection

        Returns:
            out: Normalized output
            new_residual: Updated residual (x + residual)
        """
        if self.dropout_p > 0 and self.training:
            return fused_residual_rmsnorm_dropout(
                x, residual, self.weight,
                dropout_p=self.dropout_p,
                eps=self.eps,
                training=self.training,
            )
        else:
            return fused_residual_rmsnorm(
                x, residual, self.weight,
                eps=self.eps,
                store_residual=True,
            )


# =============================================================================
# BENCHMARK
# =============================================================================

def benchmark_fused_residual(
    batch_size: int = 4,
    seq_len: int = 512,
    hidden_size: int = 768,
    n_iter: int = 100,
):
    """Benchmark fused vs separate residual + RMSNorm."""
    import time

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = "cuda"
    dtype = torch.float16

    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    residual = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    weight = torch.ones(hidden_size, device=device, dtype=dtype)

    def separate_ops(x, residual, weight):
        hidden = x + residual
        rms = torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + 1e-6)
        return hidden * rms * weight, hidden

    # Warmup
    for _ in range(10):
        _ = fused_residual_rmsnorm(x, residual, weight)
        _ = separate_ops(x, residual, weight)
    torch.cuda.synchronize()

    # Benchmark fused
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = fused_residual_rmsnorm(x, residual, weight)
    torch.cuda.synchronize()
    fused_time = (time.perf_counter() - start) / n_iter * 1000

    # Benchmark separate
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = separate_ops(x, residual, weight)
    torch.cuda.synchronize()
    separate_time = (time.perf_counter() - start) / n_iter * 1000

    print(f"\nFused Residual + RMSNorm Benchmark")
    print(f"  batch={batch_size}, seq={seq_len}, hidden={hidden_size}")
    print(f"=" * 50)
    print(f"  Separate:  {separate_time:.3f} ms")
    print(f"  Fused:     {fused_time:.3f} ms")
    print(f"  Speedup:   {separate_time / fused_time:.2f}x")
    print(f"=" * 50)


if __name__ == "__main__":
    benchmark_fused_residual()
