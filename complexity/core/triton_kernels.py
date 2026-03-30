"""
Fused Triton kernel for Sort-and-Split routed projection.

Replaces the PyTorch sequence: gather(sort) → view → bmm → zeros → scatter(unsort)
with a single fused GPU kernel, eliminating intermediate tensor allocations.

Key insight: in sorted order, tokens are contiguous per expert (chunk = N/E).
Each Triton program block handles a tile of tokens that all belong to the SAME
expert, so we load one expert's weight matrix and do a standard tiled matmul.

INL / Complexity-ML — 2026
"""

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _fused_routed_proj_kernel(
        # Pointers
        x_ptr,          # [N, D_in]  input (original order)
        w_ptr,          # [E, D_in, D_out]  expert weights
        sort_idx_ptr,   # [N]  argsort permutation
        out_ptr,        # [N, D_out]  output (original order)
        # Strides
        stride_x_n, stride_x_d,
        stride_w_e, stride_w_din, stride_w_dout,
        stride_out_n, stride_out_d,
        # Dimensions
        N,
        D_in,
        D_out,
        chunk,          # N // E
        # Block sizes
        BLOCK_M: tl.constexpr,  # tokens per block
        BLOCK_K: tl.constexpr,  # D_in tile
        BLOCK_N: tl.constexpr,  # D_out tile
    ):
        """Tiled matmul with fused gather/scatter.

        Each block handles BLOCK_M contiguous sorted tokens × BLOCK_N output features.
        Since tokens are sorted by expert, all tokens in a block share the same expert.
        """
        pid_m = tl.program_id(0)  # which token block (sorted order)
        pid_n = tl.program_id(1)  # which output feature block

        # Sorted token indices
        m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = m_offs < N

        # Expert ID — all tokens in this block share the same expert
        # (because they're contiguous in sorted order and chunk = N/E)
        block_start = pid_m * BLOCK_M
        expert_id = block_start // chunk
        # Clamp to valid expert range
        expert_id = min(expert_id, (N // chunk) - 1)

        # Original (unsorted) token indices — for gather/scatter
        orig_idx = tl.load(sort_idx_ptr + m_offs, mask=m_mask, other=0)

        # Output feature offsets
        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_offs < D_out

        # Tiled matmul: accumulate over D_in
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, D_in, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offs < D_in

            # Load x[orig_idx, k] — gather from original positions
            x_ptrs = x_ptr + orig_idx[:, None] * stride_x_n + k_offs[None, :] * stride_x_d
            x_vals = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

            # Load w[expert_id, k, n] — single expert for entire block
            w_ptrs = w_ptr + expert_id * stride_w_e + k_offs[:, None] * stride_w_din + n_offs[None, :] * stride_w_dout
            w_vals = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

            acc += tl.dot(x_vals, w_vals)

        # Store to out[orig_idx, n] — scatter to original positions
        out_ptrs = out_ptr + orig_idx[:, None] * stride_out_n + n_offs[None, :] * stride_out_d
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


def fused_routed_proj(
    x: torch.Tensor,
    weight: torch.Tensor,
    sort_idx: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Fused sort + matmul + unsort in a single Triton kernel.

    Eliminates intermediate allocations (sorted_x, sorted_out, zeros).

    Args:
        x: [B, S, D_in] or [N, D_in] input
        weight: [E, D_in, D_out] expert weight matrices
        sort_idx: [N] argsort permutation
        num_experts: number of experts E

    Returns:
        [B, S, D_out] or [N, D_out] output in original token order
    """
    orig_shape = x.shape
    if x.dim() == 3:
        B, S, D_in = x.shape
        x = x.reshape(B * S, D_in)
    else:
        D_in = x.shape[-1]

    N = x.shape[0]
    D_out = weight.shape[2]
    chunk = N // num_experts

    out = torch.empty(N, D_out, device=x.device, dtype=x.dtype)

    # Block sizes tuned for sm_90 (H100/H200)
    BLOCK_M = min(64, chunk)
    BLOCK_N = min(64, D_out)
    BLOCK_K = min(32, D_in)

    grid = (triton.cdiv(N, BLOCK_M), triton.cdiv(D_out, BLOCK_N))

    _fused_routed_proj_kernel[grid](
        x, weight, sort_idx, out,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1), weight.stride(2),
        out.stride(0), out.stride(1),
        N, D_in, D_out, chunk,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
    )

    if len(orig_shape) == 3:
        return out.reshape(orig_shape[0], orig_shape[1], D_out)
    return out


def routed_proj(x, weight, sort_idx, num_experts):
    """Drop-in replacement for _routed_proj with Triton fallback.

    Uses fused Triton kernel on CUDA, falls back to PyTorch bmm on CPU.
    """
    if HAS_TRITON and x.is_cuda:
        return fused_routed_proj(x, weight, sort_idx, num_experts)

    # Fallback: PyTorch bmm
    if x.dim() == 3:
        bsz, seqlen, dim = x.shape
        N = bsz * seqlen
        flat_x = x.reshape(N, dim)
    else:
        N, dim = x.shape
        flat_x = x
        bsz = None

    chunk = N // num_experts
    sorted_x = flat_x[sort_idx]
    sorted_out = torch.bmm(
        sorted_x.view(num_experts, chunk, dim), weight
    ).reshape(N, -1)
    out = torch.zeros(N, sorted_out.shape[-1], device=x.device, dtype=sorted_out.dtype)
    out[sort_idx] = sorted_out

    if bsz is not None:
        return out.reshape(bsz, seqlen, -1)
    return out
