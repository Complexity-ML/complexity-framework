"""
Triton-accelerated Token-Routed MLP with CGGR

CGGR = Coalesced Grouped Gemm with Ragged tensors

Key optimization for Token-Routed MLP:
1. Sort tokens by expert (token ID -> expert mapping is deterministic)
2. Grouped GEMM: Single kernel for all experts
3. Coalesced memory access (5-6x faster than bmm)

Performance:
- Standard loop: O(num_experts) iterations
- Batched bmm: 3.3x speedup
- CGGR Triton: 5-6x speedup

Author: Boris Peyriguere
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

import logging
_logger = logging.getLogger(__name__)

# Try to import Triton
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    _logger.warning("Triton not available — Token-Routed MLP will use PyTorch fallback")


def _is_rocm() -> bool:
    """Return True when PyTorch is running on AMD ROCm. Used to pick autotune
    configs without num_stages > 2 (deep pipelining hurts on CDNA — too few
    waves available, register pressure spills)."""
    try:
        return torch.cuda.is_available() and torch.version.hip is not None
    except Exception:
        return False


def _to_local(t: torch.Tensor) -> torch.Tensor:
    """Convert DTensor to local tensor (FSDP v2 compat)."""
    if hasattr(t, 'to_local'):
        return t.to_local()
    return t


# =============================================================================
# CGGR UTILITIES
# =============================================================================

def sort_tokens_by_expert(
    tokens: torch.Tensor,
    expert_ids: torch.Tensor,
    num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sort tokens by expert ID for coalesced access.

    For Token-Routed MLP, expert_ids are computed deterministically
    from token IDs, so this is stable and predictable.

    Returns:
        sorted_tokens: Tokens reordered by expert
        sorted_indices: Original indices (for scatter back)
        expert_offsets: Start index for each expert [num_experts + 1]
        expert_counts: Number of tokens per expert [num_experts]
    """
    sorted_expert_ids, sorted_indices = torch.sort(expert_ids)
    sorted_tokens = tokens[sorted_indices]

    expert_counts = torch.bincount(expert_ids, minlength=num_experts)
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.long, device=tokens.device)
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)

    return sorted_tokens, sorted_indices, expert_offsets, expert_counts


def grouped_gemm_pytorch(
    sorted_tokens: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_counts: torch.Tensor
) -> torch.Tensor:
    """
    Grouped GEMM fallback (PyTorch).
    """
    num_experts = expert_weights.shape[0]
    out_dim = expert_weights.shape[2]
    total_tokens = sorted_tokens.shape[0]

    output = torch.zeros(total_tokens, out_dim, device=sorted_tokens.device, dtype=sorted_tokens.dtype)

    for exp_id in range(num_experts):
        start = expert_offsets[exp_id].item()
        end = expert_offsets[exp_id + 1].item()

        if end > start:
            output[start:end] = sorted_tokens[start:end] @ expert_weights[exp_id]

    return output


if HAS_TRITON:
    # =========================================================================
    # CGGR TRITON KERNELS
    # =========================================================================

    @triton.jit
    def _pair_coverage_hash_kernel(
        token_ids_ptr,
        route_codes_ptr,
        expert_pairs_ptr,
        output_ptr,
        total_tokens: tl.constexpr,
        vocab_size: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Decode a deterministic top-2 route directly from token identity.

        The large per-layer ``int64[2, vocab]`` lookup is replaced by one
        compact ``uint8[vocab]`` compiled hash plus a tiny pair table. Its
        output is laid out as two concatenated int32 assignment streams
        consumed by one CGGR partition.
        """

        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_tokens
        token_ids = tl.load(token_ids_ptr + offsets, mask=mask, other=0)
        token_ids = tl.maximum(0, tl.minimum(token_ids, vocab_size - 1))
        route_codes = tl.load(
            route_codes_ptr + token_ids, mask=mask, other=0
        )
        pair_indices = route_codes & 0x7

        expert_a = tl.load(
            expert_pairs_ptr + pair_indices * 2,
            mask=mask,
            other=0,
        )
        expert_b = tl.load(
            expert_pairs_ptr + pair_indices * 2 + 1,
            mask=mask,
            other=0,
        )
        swap = (route_codes & 0x8) != 0
        primary = tl.where(swap, expert_b, expert_a)
        secondary = tl.where(swap, expert_a, expert_b)
        tl.store(output_ptr + offsets, primary, mask=mask)
        tl.store(output_ptr + total_tokens + offsets, secondary, mask=mask)

    @triton.jit
    def _pair_hash_block_counts_kernel(
        token_ids_ptr,
        route_codes_ptr,
        expert_pairs_ptr,
        block_counts_ptr,
        total_tokens: tl.constexpr,
        vocab_size: tl.constexpr,
        num_experts: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Count each block's two hash assignments without global atomics."""

        block_id = tl.program_id(0)
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_tokens
        token_ids = tl.load(token_ids_ptr + offsets, mask=mask, other=0)
        token_ids = tl.maximum(0, tl.minimum(token_ids, vocab_size - 1))
        codes = tl.load(route_codes_ptr + token_ids, mask=mask, other=0)
        pair_indices = codes & 0x7
        expert_a = tl.load(
            expert_pairs_ptr + pair_indices * 2, mask=mask, other=0
        )
        expert_b = tl.load(
            expert_pairs_ptr + pair_indices * 2 + 1, mask=mask, other=0
        )

        for expert_idx in range(num_experts):
            count = tl.sum(
                (mask & (expert_a == expert_idx)).to(tl.int32)
                + (mask & (expert_b == expert_idx)).to(tl.int32),
                axis=0,
            )
            tl.store(
                block_counts_ptr + block_id * num_experts + expert_idx,
                count,
            )

    @triton.jit
    def _pair_hash_scatter_kernel(
        token_ids_ptr,
        route_codes_ptr,
        expert_pairs_ptr,
        block_offsets_ptr,
        expert_offsets_ptr,
        sorted_indices_ptr,
        inverse_indices_ptr,
        total_tokens: tl.constexpr,
        vocab_size: tl.constexpr,
        num_experts: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Scatter both routes into expert-contiguous order.

        Prefixes are computed per input block, so the scatter needs no global
        atomics and remains deterministic across launches.
        """

        block_id = tl.program_id(0)
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_tokens
        token_ids = tl.load(token_ids_ptr + offsets, mask=mask, other=0)
        token_ids = tl.maximum(0, tl.minimum(token_ids, vocab_size - 1))
        codes = tl.load(route_codes_ptr + token_ids, mask=mask, other=0)
        pair_indices = codes & 0x7
        expert_a = tl.load(
            expert_pairs_ptr + pair_indices * 2, mask=mask, other=0
        )
        expert_b = tl.load(
            expert_pairs_ptr + pair_indices * 2 + 1, mask=mask, other=0
        )
        swap = (codes & 0x8) != 0
        primary = tl.where(swap, expert_b, expert_a)
        secondary = tl.where(swap, expert_a, expert_b)

        for expert_idx in range(num_experts):
            primary_match = mask & (primary == expert_idx)
            secondary_match = mask & (secondary == expert_idx)
            match = primary_match | secondary_match
            local_positions = (
                tl.cumsum(match.to(tl.int32), axis=0) - 1
            )
            block_start = tl.load(
                block_offsets_ptr
                + block_id * num_experts
                + expert_idx
            )
            expert_start = tl.load(expert_offsets_ptr + expert_idx)
            destinations = expert_start + block_start + local_positions
            assignment_indices = tl.where(
                primary_match,
                offsets,
                total_tokens + offsets,
            )
            tl.store(
                sorted_indices_ptr + destinations,
                assignment_indices,
                mask=match,
            )
            tl.store(
                inverse_indices_ptr + assignment_indices,
                destinations,
                mask=match,
            )

    @triton.jit
    def _pair_hash_reduce_kernel(
        sorted_values_ptr,
        inverse_indices_ptr,
        output_ptr,
        token_count: tl.constexpr,
        feature_count: tl.constexpr,
        stride_s_row,
        stride_s_col,
        stride_o_row,
        stride_o_col,
        SCALE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Reduce two expert-sorted hash assignments into token order."""

        linear = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        total = token_count * feature_count
        mask = linear < total
        token_rows = linear // feature_count
        feature_cols = linear % feature_count
        primary_rows = tl.load(
            inverse_indices_ptr + token_rows,
            mask=mask,
            other=0,
        )
        secondary_rows = tl.load(
            inverse_indices_ptr + token_count + token_rows,
            mask=mask,
            other=0,
        )
        primary = tl.load(
            sorted_values_ptr
            + primary_rows * stride_s_row
            + feature_cols * stride_s_col,
            mask=mask,
            other=0.0,
        )
        secondary = tl.load(
            sorted_values_ptr
            + secondary_rows * stride_s_row
            + feature_cols * stride_s_col,
            mask=mask,
            other=0.0,
        )
        tl.store(
            output_ptr
            + token_rows * stride_o_row
            + feature_cols * stride_o_col,
            (primary + secondary) * SCALE,
            mask=mask,
        )

    @triton.jit
    def _pair_hash_expand_kernel(
        token_values_ptr,
        sorted_indices_ptr,
        output_ptr,
        token_count: tl.constexpr,
        feature_count: tl.constexpr,
        stride_t_row,
        stride_t_col,
        stride_o_row,
        stride_o_col,
        SCALE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Expand token-ordered values into expert-sorted assignment order."""

        linear = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        assignment_count = 2 * token_count
        total = assignment_count * feature_count
        mask = linear < total
        sorted_rows = linear // feature_count
        feature_cols = linear % feature_count
        assignment_rows = tl.load(
            sorted_indices_ptr + sorted_rows,
            mask=mask,
            other=0,
        )
        token_rows = assignment_rows % token_count
        values = tl.load(
            token_values_ptr
            + token_rows * stride_t_row
            + feature_cols * stride_t_col,
            mask=mask,
            other=0.0,
        )
        tl.store(
            output_ptr
            + sorted_rows * stride_o_row
            + feature_cols * stride_o_col,
            values * SCALE,
            mask=mask,
        )

    @triton.jit
    def _pair_hash_weighted_reduce_kernel(
        sorted_values_ptr,
        inverse_indices_ptr,
        primary_weights_ptr,
        output_ptr,
        token_count: tl.constexpr,
        feature_count: tl.constexpr,
        stride_s_row,
        stride_s_col,
        stride_o_row,
        stride_o_col,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Reduce two hash routes with one learned weight per token."""

        linear = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        total = token_count * feature_count
        mask = linear < total
        token_rows = linear // feature_count
        feature_cols = linear % feature_count
        primary_rows = tl.load(
            inverse_indices_ptr + token_rows,
            mask=mask,
            other=0,
        )
        secondary_rows = tl.load(
            inverse_indices_ptr + token_count + token_rows,
            mask=mask,
            other=0,
        )
        primary = tl.load(
            sorted_values_ptr
            + primary_rows * stride_s_row
            + feature_cols * stride_s_col,
            mask=mask,
            other=0.0,
        )
        secondary = tl.load(
            sorted_values_ptr
            + secondary_rows * stride_s_row
            + feature_cols * stride_s_col,
            mask=mask,
            other=0.0,
        )
        primary_weight = tl.load(
            primary_weights_ptr + token_rows,
            mask=mask,
            other=0.5,
        ).to(tl.float32)
        output = (
            primary.to(tl.float32) * primary_weight
            + secondary.to(tl.float32) * (1.0 - primary_weight)
        )
        tl.store(
            output_ptr
            + token_rows * stride_o_row
            + feature_cols * stride_o_col,
            output,
            mask=mask,
        )

    @triton.jit
    def _pair_hash_weighted_expand_kernel(
        token_values_ptr,
        sorted_indices_ptr,
        primary_weights_ptr,
        output_ptr,
        token_count: tl.constexpr,
        feature_count: tl.constexpr,
        stride_t_row,
        stride_t_col,
        stride_o_row,
        stride_o_col,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Backpropagate through the weighted hash route reduction."""

        linear = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        assignment_count = 2 * token_count
        total = assignment_count * feature_count
        mask = linear < total
        sorted_rows = linear // feature_count
        feature_cols = linear % feature_count
        assignment_rows = tl.load(
            sorted_indices_ptr + sorted_rows,
            mask=mask,
            other=0,
        )
        token_rows = assignment_rows % token_count
        primary_weight = tl.load(
            primary_weights_ptr + token_rows,
            mask=mask,
            other=0.5,
        ).to(tl.float32)
        route_weight = tl.where(
            assignment_rows < token_count,
            primary_weight,
            1.0 - primary_weight,
        )
        values = tl.load(
            token_values_ptr
            + token_rows * stride_t_row
            + feature_cols * stride_t_col,
            mask=mask,
            other=0.0,
        )
        tl.store(
            output_ptr
            + sorted_rows * stride_o_row
            + feature_cols * stride_o_col,
            values.to(tl.float32) * route_weight,
            mask=mask,
        )

    @triton.jit
    def _pair_hash_weight_grad_kernel(
        sorted_values_ptr,
        inverse_indices_ptr,
        grad_output_ptr,
        grad_weights_ptr,
        token_count: tl.constexpr,
        feature_count: tl.constexpr,
        stride_s_row,
        stride_s_col,
        stride_g_row,
        stride_g_col,
        BLOCK_FEATURES: tl.constexpr,
    ):
        """Compute dL/d(primary_weight) for each token."""

        token_row = tl.program_id(0)
        feature_cols = tl.arange(0, BLOCK_FEATURES)
        feature_mask = feature_cols < feature_count
        primary_row = tl.load(inverse_indices_ptr + token_row)
        secondary_row = tl.load(
            inverse_indices_ptr + token_count + token_row
        )
        primary = tl.load(
            sorted_values_ptr
            + primary_row * stride_s_row
            + feature_cols * stride_s_col,
            mask=feature_mask,
            other=0.0,
        ).to(tl.float32)
        secondary = tl.load(
            sorted_values_ptr
            + secondary_row * stride_s_row
            + feature_cols * stride_s_col,
            mask=feature_mask,
            other=0.0,
        ).to(tl.float32)
        grad = tl.load(
            grad_output_ptr
            + token_row * stride_g_row
            + feature_cols * stride_g_col,
            mask=feature_mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            grad_weights_ptr + token_row,
            tl.sum(grad * (primary - secondary), axis=0),
        )

    # Autotune configs cover the MoE shapes we actually run:
    #   hidden ∈ {640, 1024}, expert_inter ∈ {448, 502, 2008}, shared_inter ∈ {..., 2008}
    # Tuning keys are the matmul dims (in_dim, out_dim); num_experts is
    # dispatch-only and doesn't affect per-block perf.
    # NVIDIA configs — Hopper/Ada/Blackwell. num_stages=3-4 hides global memory
    # latency through software pipelining. Tile sizes target sm_80+ Tensor Cores.
    _CGGR_CONFIGS_CUDA = [
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64,  "BLOCK_K": 32},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 32},  num_warps=8, num_stages=3),
    ]

    # AMD/CDNA configs — gfx9xx (MI200/MI300/MI350). num_stages capped at 2
    # because CDNA has fewer SMs and deep pipelining over-allocates registers.
    # Larger BLOCK_K helps fill the 16x16x16 MFMA pipes.
    _CGGR_CONFIGS_ROCM = [
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=8, num_stages=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64},  num_warps=8, num_stages=1),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=1),
    ]

    _CGGR_CONFIGS = _CGGR_CONFIGS_ROCM if _is_rocm() else _CGGR_CONFIGS_CUDA

    # Separate autotune config list for grad_W — different reduction pattern
    # (reduce over tokens, tiles are (in_dim, out_dim)) benefits from narrower
    # BLOCK_M (tokens per block in K dim) and wider (BLOCK_N, BLOCK_O) tiles.
    _CGGR_GRAD_W_CONFIGS_CUDA = [
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64,  "BLOCK_O": 64},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128, "BLOCK_O": 64},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64,  "BLOCK_O": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_O": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_O": 64},  num_warps=4, num_stages=4),
    ]

    _CGGR_GRAD_W_CONFIGS_ROCM = [
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64,  "BLOCK_O": 64},  num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_O": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_O": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_O": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128, "BLOCK_O": 128}, num_warps=8, num_stages=1),
    ]

    _CGGR_GRAD_W_CONFIGS = _CGGR_GRAD_W_CONFIGS_ROCM if _is_rocm() else _CGGR_GRAD_W_CONFIGS_CUDA

    @triton.autotune(configs=_CGGR_GRAD_W_CONFIGS, key=["in_dim", "out_dim"])
    @triton.jit
    def _cggr_grad_w_kernel(
        sorted_x_ptr,       # [total_tokens, in_dim]  (fwd sorted activations)
        grad_out_ptr,       # [total_tokens, out_dim] (grad of fwd output)
        offsets_ptr,        # [num_experts + 1]
        grad_w_ptr,         # [num_experts, in_dim, out_dim]  OUT
        in_dim,
        out_dim,
        stride_x_row, stride_x_col,
        stride_g_row, stride_g_col,
        stride_w_exp, stride_w_in, stride_w_out,
        BLOCK_M: tl.constexpr,   # tokens per reduction step
        BLOCK_N: tl.constexpr,   # in_dim tile
        BLOCK_O: tl.constexpr,   # out_dim tile
    ):
        """
        Compute grad_W[e] = sorted_x[expert_e].T @ grad_output[expert_e]
        per expert WITHOUT padding. Each kernel instance owns one
        (expert, in_tile, out_tile) and reduces over the expert's token range.

        Output shape: [num_experts, in_dim, out_dim] = same layout as forward
        weights so it can be used directly as the grad in .backward().
        """
        pid_n = tl.program_id(0)   # in_dim tile
        pid_o = tl.program_id(1)   # out_dim tile
        pid_e = tl.program_id(2)   # expert id

        expert_start = tl.load(offsets_ptr + pid_e)
        expert_end = tl.load(offsets_ptr + pid_e + 1)
        if expert_end == expert_start:
            # Still write zeros so grad_W[e] is defined
            n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            o_offs = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
            n_mask = n_offs < in_dim
            o_mask = o_offs < out_dim
            w_ptrs = (grad_w_ptr + pid_e * stride_w_exp
                      + n_offs[:, None] * stride_w_in
                      + o_offs[None, :] * stride_w_out)
            tl.store(w_ptrs, tl.zeros([BLOCK_N, BLOCK_O], dtype=grad_w_ptr.dtype.element_ty),
                     mask=n_mask[:, None] & o_mask[None, :])
            return

        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        o_offs = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
        n_mask = n_offs < in_dim
        o_mask = o_offs < out_dim

        acc = tl.zeros([BLOCK_N, BLOCK_O], dtype=tl.float32)

        # Reduce over token range for this expert. Each step loads a
        # [BLOCK_M, BLOCK_N] slice of x and [BLOCK_M, BLOCK_O] slice of g,
        # then accumulates x.T @ g into the [BLOCK_N, BLOCK_O] tile.
        for m_start in range(expert_start, expert_end, BLOCK_M):
            m_offs = m_start + tl.arange(0, BLOCK_M)
            m_mask = m_offs < expert_end

            x_ptrs = (sorted_x_ptr
                      + m_offs[:, None] * stride_x_row
                      + n_offs[None, :] * stride_x_col)
            g_ptrs = (grad_out_ptr
                      + m_offs[:, None] * stride_g_row
                      + o_offs[None, :] * stride_g_col)

            x_blk = tl.load(x_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
            g_blk = tl.load(g_ptrs, mask=m_mask[:, None] & o_mask[None, :], other=0.0)

            # x.T @ g : [BLOCK_N, BLOCK_M] @ [BLOCK_M, BLOCK_O] → [BLOCK_N, BLOCK_O]
            # Native bf16×bf16→fp32 acc (Tensor Cores).
            acc += tl.dot(tl.trans(x_blk), g_blk)

        w_ptrs = (grad_w_ptr + pid_e * stride_w_exp
                  + n_offs[:, None] * stride_w_in
                  + o_offs[None, :] * stride_w_out)
        tl.store(w_ptrs, acc.to(grad_w_ptr.dtype.element_ty),
                 mask=n_mask[:, None] & o_mask[None, :])

    @triton.autotune(
        configs=_CGGR_GRAD_W_CONFIGS,
        key=["in_dim", "out_dim"],
    )
    @triton.jit
    def _hash_cggr_grad_w_kernel(
        tokens_ptr,
        sorted_indices_ptr,
        grad_out_ptr,
        offsets_ptr,
        grad_w_ptr,
        source_rows,
        in_dim,
        out_dim,
        stride_x_row,
        stride_x_col,
        stride_g_row,
        stride_g_col,
        stride_w_exp,
        stride_w_in,
        stride_w_out,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_O: tl.constexpr,
    ):
        """CGGR weight gradient with hash-native indirect token reads."""

        pid_n = tl.program_id(0)
        pid_o = tl.program_id(1)
        pid_e = tl.program_id(2)
        expert_start = tl.load(offsets_ptr + pid_e)
        expert_end = tl.load(offsets_ptr + pid_e + 1)

        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        o_offs = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
        n_mask = n_offs < in_dim
        o_mask = o_offs < out_dim

        if expert_end == expert_start:
            w_ptrs = (
                grad_w_ptr
                + pid_e * stride_w_exp
                + n_offs[:, None] * stride_w_in
                + o_offs[None, :] * stride_w_out
            )
            tl.store(
                w_ptrs,
                tl.zeros(
                    [BLOCK_N, BLOCK_O],
                    dtype=grad_w_ptr.dtype.element_ty,
                ),
                mask=n_mask[:, None] & o_mask[None, :],
            )
            return

        acc = tl.zeros([BLOCK_N, BLOCK_O], dtype=tl.float32)
        for m_start in range(expert_start, expert_end, BLOCK_M):
            sorted_rows = m_start + tl.arange(0, BLOCK_M)
            m_mask = sorted_rows < expert_end
            assignment_rows = tl.load(
                sorted_indices_ptr + sorted_rows,
                mask=m_mask,
                other=0,
            )
            source_indices = assignment_rows % source_rows
            x_ptrs = (
                tokens_ptr
                + source_indices[:, None] * stride_x_row
                + n_offs[None, :] * stride_x_col
            )
            g_ptrs = (
                grad_out_ptr
                + sorted_rows[:, None] * stride_g_row
                + o_offs[None, :] * stride_g_col
            )
            x_blk = tl.load(
                x_ptrs,
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            g_blk = tl.load(
                g_ptrs,
                mask=m_mask[:, None] & o_mask[None, :],
                other=0.0,
            )
            acc += tl.dot(tl.trans(x_blk), g_blk)

        w_ptrs = (
            grad_w_ptr
            + pid_e * stride_w_exp
            + n_offs[:, None] * stride_w_in
            + o_offs[None, :] * stride_w_out
        )
        tl.store(
            w_ptrs,
            acc.to(grad_w_ptr.dtype.element_ty),
            mask=n_mask[:, None] & o_mask[None, :],
        )


    def cggr_grad_w_triton(
        sorted_x: torch.Tensor,       # [T, in_dim]
        grad_output: torch.Tensor,    # [T, out_dim]
        expert_offsets: torch.Tensor, # [E + 1]
        num_experts: int,
    ) -> torch.Tensor:
        """Compute grad_W [E, in_dim, out_dim] without padding."""
        in_dim = sorted_x.shape[1]
        out_dim = grad_output.shape[1]
        grad_W = torch.empty(num_experts, in_dim, out_dim,
                             device=sorted_x.device, dtype=sorted_x.dtype)

        grid = lambda META: (
            triton.cdiv(in_dim, META["BLOCK_N"]),
            triton.cdiv(out_dim, META["BLOCK_O"]),
            num_experts,
        )
        _cggr_grad_w_kernel[grid](
            sorted_x, grad_output, expert_offsets, grad_W,
            in_dim, out_dim,
            sorted_x.stride(0), sorted_x.stride(1),
            grad_output.stride(0), grad_output.stride(1),
            grad_W.stride(0), grad_W.stride(1), grad_W.stride(2),
        )
        return grad_W

    def hash_cggr_grad_w_triton(
        tokens: torch.Tensor,
        sorted_indices: torch.Tensor,
        grad_output: torch.Tensor,
        expert_offsets: torch.Tensor,
        num_experts: int,
    ) -> torch.Tensor:
        """Compute CGGR weight gradients without materializing sorted input."""

        source_rows, in_dim = tokens.shape
        out_dim = grad_output.shape[1]
        grad_w = torch.empty(
            num_experts,
            in_dim,
            out_dim,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        grid = lambda META: (
            triton.cdiv(in_dim, META["BLOCK_N"]),
            triton.cdiv(out_dim, META["BLOCK_O"]),
            num_experts,
        )
        _hash_cggr_grad_w_kernel[grid](
            tokens,
            sorted_indices,
            grad_output,
            expert_offsets,
            grad_w,
            source_rows,
            in_dim,
            out_dim,
            tokens.stride(0),
            tokens.stride(1),
            grad_output.stride(0),
            grad_output.stride(1),
            grad_w.stride(0),
            grad_w.stride(1),
            grad_w.stride(2),
        )
        return grad_w


    @triton.autotune(configs=_CGGR_CONFIGS, key=["in_dim", "out_dim"])
    @triton.jit
    def _cggr_grouped_gemm_kernel(
        tokens_ptr,
        weights_ptr,
        offsets_ptr,
        output_ptr,
        in_dim,
        out_dim,
        num_experts,
        total_tokens,
        stride_t_row,
        stride_t_col,
        stride_w_exp,
        stride_w_in,
        stride_w_out,
        stride_o_row,
        stride_o_col,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        CGGR Grouped GEMM kernel.

        Computes matmuls for all experts in parallel.
        Tokens are pre-sorted by expert for coalesced access.

        Keeps bf16/fp16 inputs native to tl.dot so Tensor Cores are used;
        accumulator stays fp32, output is cast back to the buffer dtype.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_expert = tl.program_id(2)

        # Expert boundaries
        expert_start = tl.load(offsets_ptr + pid_expert)
        expert_end = tl.load(offsets_ptr + pid_expert + 1)
        n_tokens_expert = expert_end - expert_start

        if n_tokens_expert == 0:
            return

        token_start = expert_start + pid_m * BLOCK_M
        if token_start >= expert_end:
            return

        token_offs = token_start + tl.arange(0, BLOCK_M)
        token_mask = token_offs < expert_end

        out_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        out_mask = out_offs < out_dim

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for k in range(0, in_dim, BLOCK_K):
            k_offs = k + tl.arange(0, BLOCK_K)
            k_mask = k_offs < in_dim

            t_ptrs = tokens_ptr + token_offs[:, None] * stride_t_row + k_offs[None, :] * stride_t_col
            t = tl.load(t_ptrs, mask=token_mask[:, None] & k_mask[None, :], other=0.0)

            w_ptrs = weights_ptr + pid_expert * stride_w_exp + k_offs[:, None] * stride_w_in + out_offs[None, :] * stride_w_out
            w = tl.load(w_ptrs, mask=k_mask[:, None] & out_mask[None, :], other=0.0)

            # Native bf16/fp16 × bf16/fp16 → fp32 accumulator. This is the
            # canonical Tensor Core path; previously we upcast to fp32 here
            # which disabled Tensor Cores and gave 3-4× slower throughput.
            acc += tl.dot(t, w)

        o_ptrs = output_ptr + token_offs[:, None] * stride_o_row + out_offs[None, :] * stride_o_col
        # Cast back to the output buffer dtype (bf16/fp16) before store
        tl.store(o_ptrs, acc.to(output_ptr.dtype.element_ty),
                 mask=token_mask[:, None] & out_mask[None, :])

    @triton.autotune(configs=_CGGR_CONFIGS, key=["in_dim", "out_dim"])
    @triton.jit
    def _hash_cggr_grouped_gemm_kernel(
        tokens_ptr,
        sorted_indices_ptr,
        weights_ptr,
        offsets_ptr,
        output_ptr,
        source_rows,
        in_dim,
        out_dim,
        num_experts,
        total_assignments,
        stride_t_row,
        stride_t_col,
        stride_w_exp,
        stride_w_in,
        stride_w_out,
        stride_o_row,
        stride_o_col,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """CGGR projection reading token rows through the compact hash order."""

        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_expert = tl.program_id(2)
        expert_start = tl.load(offsets_ptr + pid_expert)
        expert_end = tl.load(offsets_ptr + pid_expert + 1)
        if expert_end == expert_start:
            return

        sorted_start = expert_start + pid_m * BLOCK_M
        if sorted_start >= expert_end:
            return

        sorted_rows = sorted_start + tl.arange(0, BLOCK_M)
        row_mask = sorted_rows < expert_end
        assignment_rows = tl.load(
            sorted_indices_ptr + sorted_rows,
            mask=row_mask,
            other=0,
        )
        source_indices = assignment_rows % source_rows
        out_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        out_mask = out_offs < out_dim
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for k in range(0, in_dim, BLOCK_K):
            k_offs = k + tl.arange(0, BLOCK_K)
            k_mask = k_offs < in_dim
            token_ptrs = (
                tokens_ptr
                + source_indices[:, None] * stride_t_row
                + k_offs[None, :] * stride_t_col
            )
            token_block = tl.load(
                token_ptrs,
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            weight_ptrs = (
                weights_ptr
                + pid_expert * stride_w_exp
                + k_offs[:, None] * stride_w_in
                + out_offs[None, :] * stride_w_out
            )
            weight_block = tl.load(
                weight_ptrs,
                mask=k_mask[:, None] & out_mask[None, :],
                other=0.0,
            )
            acc += tl.dot(token_block, weight_block)

        output_ptrs = (
            output_ptr
            + sorted_rows[:, None] * stride_o_row
            + out_offs[None, :] * stride_o_col
        )
        tl.store(
            output_ptrs,
            acc.to(output_ptr.dtype.element_ty),
            mask=row_mask[:, None] & out_mask[None, :],
        )


    @triton.jit
    def _fused_swiglu_kernel(
        gate_ptr,
        up_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused SwiGLU: silu(gate) * up
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0)

        # SiLU: x * sigmoid(x)
        silu_gate = gate * tl.sigmoid(gate)
        out = silu_gate * up

        tl.store(output_ptr + offsets, out, mask=mask)


    def cggr_grouped_gemm_triton(
        sorted_tokens: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """
        CGGR Grouped GEMM using Triton.

        BLOCK_M/N/K are picked by @triton.autotune per (in_dim, out_dim) key,
        cached across calls. The grid uses total_tokens as a safe upper bound
        on the M axis — blocks outside an expert's range early-return (line
        ~155). This avoids a CPU sync on `max(expert_counts).item()` every
        step; a handful of no-op SM launches is cheaper than a sync on
        modern GPUs.
        """
        total_tokens, in_dim = sorted_tokens.shape
        num_experts, _, out_dim = expert_weights.shape

        output = torch.empty(total_tokens, out_dim, device=sorted_tokens.device, dtype=sorted_tokens.dtype)

        # Grid: (ceil_div(total_tokens, BLOCK_M), ceil_div(out_dim, BLOCK_N), num_experts).
        # Autotune picks BLOCK_M / BLOCK_N; we pass the grid as a lambda so
        # it re-evaluates once autotune has chosen the config.
        grid = lambda META: (
            triton.cdiv(total_tokens, META["BLOCK_M"]),
            triton.cdiv(out_dim, META["BLOCK_N"]),
            num_experts,
        )

        _cggr_grouped_gemm_kernel[grid](
            sorted_tokens, expert_weights, expert_offsets,
            output,
            in_dim, out_dim, num_experts, total_tokens,
            sorted_tokens.stride(0), sorted_tokens.stride(1),
            expert_weights.stride(0), expert_weights.stride(1), expert_weights.stride(2),
            output.stride(0), output.stride(1),
        )

        return output

    def hash_cggr_grouped_gemm_triton(
        tokens: torch.Tensor,
        sorted_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Grouped GEMM over hash assignments without a sorted-token copy."""

        source_rows, in_dim = tokens.shape
        total_assignments = int(sorted_indices.numel())
        num_experts, _, out_dim = expert_weights.shape
        output = torch.empty(
            total_assignments,
            out_dim,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        grid = lambda META: (
            triton.cdiv(total_assignments, META["BLOCK_M"]),
            triton.cdiv(out_dim, META["BLOCK_N"]),
            num_experts,
        )
        _hash_cggr_grouped_gemm_kernel[grid](
            tokens,
            sorted_indices,
            expert_weights,
            expert_offsets,
            output,
            source_rows,
            in_dim,
            out_dim,
            num_experts,
            total_assignments,
            tokens.stride(0),
            tokens.stride(1),
            expert_weights.stride(0),
            expert_weights.stride(1),
            expert_weights.stride(2),
            output.stride(0),
            output.stride(1),
        )
        return output

    def pair_hash_reduce_triton(
        sorted_values: torch.Tensor,
        inverse_indices: torch.Tensor,
        token_count: int,
        *,
        scale: float,
    ) -> torch.Tensor:
        """Reduce the two sorted hash routes directly into token order."""

        feature_count = int(sorted_values.shape[1])
        output = torch.empty(
            token_count,
            feature_count,
            device=sorted_values.device,
            dtype=sorted_values.dtype,
        )
        block_size = 256
        total = token_count * feature_count
        _pair_hash_reduce_kernel[
            (triton.cdiv(total, block_size),)
        ](
            sorted_values,
            inverse_indices,
            output,
            token_count=token_count,
            feature_count=feature_count,
            stride_s_row=sorted_values.stride(0),
            stride_s_col=sorted_values.stride(1),
            stride_o_row=output.stride(0),
            stride_o_col=output.stride(1),
            SCALE=float(scale),
            BLOCK_SIZE=block_size,
        )
        return output

    def pair_hash_expand_triton(
        token_values: torch.Tensor,
        sorted_indices: torch.Tensor,
        token_count: int,
        *,
        scale: float,
    ) -> torch.Tensor:
        """Expand token rows directly into sorted top-2 assignment order."""

        feature_count = int(token_values.shape[1])
        output = torch.empty(
            2 * token_count,
            feature_count,
            device=token_values.device,
            dtype=token_values.dtype,
        )
        block_size = 256
        total = 2 * token_count * feature_count
        _pair_hash_expand_kernel[
            (triton.cdiv(total, block_size),)
        ](
            token_values,
            sorted_indices,
            output,
            token_count=token_count,
            feature_count=feature_count,
            stride_t_row=token_values.stride(0),
            stride_t_col=token_values.stride(1),
            stride_o_row=output.stride(0),
            stride_o_col=output.stride(1),
            SCALE=float(scale),
            BLOCK_SIZE=block_size,
        )
        return output

    def pair_hash_weighted_reduce_triton(
        sorted_values: torch.Tensor,
        inverse_indices: torch.Tensor,
        primary_weights: torch.Tensor,
        token_count: int,
    ) -> torch.Tensor:
        """Reduce sorted routes with token-specific primary weights."""

        feature_count = int(sorted_values.shape[1])
        output = torch.empty(
            token_count,
            feature_count,
            device=sorted_values.device,
            dtype=sorted_values.dtype,
        )
        block_size = 256
        total = token_count * feature_count
        _pair_hash_weighted_reduce_kernel[
            (triton.cdiv(total, block_size),)
        ](
            sorted_values,
            inverse_indices,
            primary_weights,
            output,
            token_count=token_count,
            feature_count=feature_count,
            stride_s_row=sorted_values.stride(0),
            stride_s_col=sorted_values.stride(1),
            stride_o_row=output.stride(0),
            stride_o_col=output.stride(1),
            BLOCK_SIZE=block_size,
        )
        return output

    def pair_hash_weighted_expand_triton(
        token_values: torch.Tensor,
        sorted_indices: torch.Tensor,
        primary_weights: torch.Tensor,
        token_count: int,
    ) -> torch.Tensor:
        """Expand token gradients with token-specific route weights."""

        feature_count = int(token_values.shape[1])
        output = torch.empty(
            2 * token_count,
            feature_count,
            device=token_values.device,
            dtype=token_values.dtype,
        )
        block_size = 256
        total = 2 * token_count * feature_count
        _pair_hash_weighted_expand_kernel[
            (triton.cdiv(total, block_size),)
        ](
            token_values,
            sorted_indices,
            primary_weights,
            output,
            token_count=token_count,
            feature_count=feature_count,
            stride_t_row=token_values.stride(0),
            stride_t_col=token_values.stride(1),
            stride_o_row=output.stride(0),
            stride_o_col=output.stride(1),
            BLOCK_SIZE=block_size,
        )
        return output

    def pair_hash_weight_grad_triton(
        sorted_values: torch.Tensor,
        inverse_indices: torch.Tensor,
        grad_output: torch.Tensor,
        token_count: int,
    ) -> torch.Tensor:
        """Compute gradients for learned hash-pair mixture weights."""

        feature_count = int(sorted_values.shape[1])
        block_features = triton.next_power_of_2(feature_count)
        if block_features > 65536:
            raise ValueError(
                "hash-pair weight gradients support at most 65,536 features"
            )
        grad_weights = torch.empty(
            token_count,
            device=sorted_values.device,
            dtype=torch.float32,
        )
        _pair_hash_weight_grad_kernel[(token_count,)](
            sorted_values,
            inverse_indices,
            grad_output,
            grad_weights,
            token_count=token_count,
            feature_count=feature_count,
            stride_s_row=sorted_values.stride(0),
            stride_s_col=sorted_values.stride(1),
            stride_g_row=grad_output.stride(0),
            stride_g_col=grad_output.stride(1),
            BLOCK_FEATURES=block_features,
        )
        return grad_weights


    def fused_swiglu_triton(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """Fused SwiGLU activation."""
        n_elements = gate.numel()
        output = torch.empty_like(gate)

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        _fused_swiglu_kernel[grid](
            gate.view(-1), up.view(-1), output.view(-1),
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return output


    class CGGRGroupedGEMM(torch.autograd.Function):
        """
        Autograd-aware wrapper around cggr_grouped_gemm_triton.

        Forward:  out[start_e:end_e] = sorted_x[start_e:end_e] @ W[e]   for each expert e
        Backward:
            grad_x[start_e:end_e] = grad_out[start_e:end_e] @ W[e].T   for each e
            grad_W[e]            = sorted_x[start_e:end_e].T @ grad_out[start_e:end_e]

        - grad_x is computed by reusing the same CGGR kernel with a transposed
          weight tensor (still O(1) Triton launch).
        - grad_W is a small loop over experts (num_experts iterations, typically
          4) — fast enough since each iteration is a single GEMM and num_experts
          is tiny. Could be replaced with a Triton kernel later if it shows up
          in profiles.

        Without this wrapper, the routed expert weights receive zero gradients
        because the underlying Triton kernel is forward-only — see commit
        8f43035 / discussion in KellerJordan/Muon#65 for the failure mode.
        """

        @staticmethod
        def forward(ctx, sorted_x: torch.Tensor, expert_weights: torch.Tensor,
                    expert_offsets: torch.Tensor) -> torch.Tensor:
            output = cggr_grouped_gemm_triton(sorted_x, expert_weights, expert_offsets)
            ctx.save_for_backward(sorted_x, expert_weights, expert_offsets)
            return output

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            sorted_x, expert_weights, expert_offsets = ctx.saved_tensors
            num_experts, in_dim, out_dim = expert_weights.shape

            grad_x = None
            grad_W = None
            grad_output = grad_output.contiguous()

            # grad_x = grad_output @ W.T per expert. Reuse CGGR with W transposed.
            if ctx.needs_input_grad[0]:
                W_T = expert_weights.transpose(-2, -1).contiguous()  # [E, out, in]
                grad_x = cggr_grouped_gemm_triton(grad_output, W_T, expert_offsets)

            # grad_W[e] = sorted_x[e].T @ grad_output[e] — unpadded CGGR kernel.
            # No zero-padding waste: each expert's token range is reduced directly
            # by the grad_w Triton kernel. Saves ~30% FLOPs under Zipf imbalance
            # vs the old padded bmm, and avoids the Python copy loop.
            if ctx.needs_input_grad[1]:
                grad_W = cggr_grad_w_triton(
                    sorted_x, grad_output, expert_offsets, num_experts,
                )

            return grad_x, grad_W, None  # offsets is non-differentiable

    class HashCGGRGroupedGEMM(torch.autograd.Function):
        """Autograd CGGR projection over compact hash assignments.

        Forward reads the original token matrix indirectly and therefore never
        creates the duplicated ``[2*N, hidden]`` sorted input. Backward reuses
        the inverse assignment to sum the two route gradients per token and
        computes weight gradients through the same indirect order.
        """

        @staticmethod
        def forward(
            ctx,
            tokens: torch.Tensor,
            expert_weights: torch.Tensor,
            expert_offsets: torch.Tensor,
            sorted_indices: torch.Tensor,
            inverse_indices: torch.Tensor,
        ) -> torch.Tensor:
            output = hash_cggr_grouped_gemm_triton(
                tokens,
                sorted_indices,
                expert_weights,
                expert_offsets,
            )
            ctx.save_for_backward(
                tokens,
                expert_weights,
                expert_offsets,
                sorted_indices,
                inverse_indices,
            )
            return output

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            (
                tokens,
                expert_weights,
                expert_offsets,
                sorted_indices,
                inverse_indices,
            ) = ctx.saved_tensors
            num_experts = int(expert_weights.shape[0])
            grad_output = grad_output.contiguous()
            grad_tokens = None
            grad_weights = None

            if ctx.needs_input_grad[0]:
                weights_t = expert_weights.transpose(-2, -1).contiguous()
                sorted_grad_tokens = cggr_grouped_gemm_triton(
                    grad_output,
                    weights_t,
                    expert_offsets,
                )
                grad_tokens = pair_hash_reduce_triton(
                    sorted_grad_tokens,
                    inverse_indices,
                    int(tokens.shape[0]),
                    scale=1.0,
                )

            if ctx.needs_input_grad[1]:
                grad_weights = hash_cggr_grad_w_triton(
                    tokens,
                    sorted_indices,
                    grad_output,
                    expert_offsets,
                    num_experts,
                )

            return (
                grad_tokens,
                grad_weights,
                None,
                None,
                None,
            )

    class PairHashReduce(torch.autograd.Function):
        """Fuse hash unsort and equal top-2 reduction."""

        @staticmethod
        def forward(
            ctx,
            sorted_values: torch.Tensor,
            sorted_indices: torch.Tensor,
            inverse_indices: torch.Tensor,
            token_count: int,
            scale: float,
        ) -> torch.Tensor:
            ctx.token_count = int(token_count)
            ctx.scale = float(scale)
            ctx.save_for_backward(sorted_indices)
            return pair_hash_reduce_triton(
                sorted_values,
                inverse_indices,
                ctx.token_count,
                scale=ctx.scale,
            )

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            (sorted_indices,) = ctx.saved_tensors
            grad_sorted = pair_hash_expand_triton(
                grad_output.contiguous(),
                sorted_indices,
                ctx.token_count,
                scale=ctx.scale,
            )
            return grad_sorted, None, None, None, None

    class PairHashWeightedReduce(torch.autograd.Function):
        """Fuse hash unsort with a learned deterministic pair mixture."""

        @staticmethod
        def forward(
            ctx,
            sorted_values: torch.Tensor,
            sorted_indices: torch.Tensor,
            inverse_indices: torch.Tensor,
            primary_weights: torch.Tensor,
            token_count: int,
        ) -> torch.Tensor:
            ctx.token_count = int(token_count)
            ctx.save_for_backward(
                sorted_values,
                sorted_indices,
                inverse_indices,
                primary_weights,
            )
            return pair_hash_weighted_reduce_triton(
                sorted_values,
                inverse_indices,
                primary_weights,
                ctx.token_count,
            )

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            (
                sorted_values,
                sorted_indices,
                inverse_indices,
                primary_weights,
            ) = ctx.saved_tensors
            grad_output = grad_output.contiguous()
            grad_sorted = pair_hash_weighted_expand_triton(
                grad_output,
                sorted_indices,
                primary_weights,
                ctx.token_count,
            )
            grad_primary_weights = pair_hash_weight_grad_triton(
                sorted_values,
                inverse_indices,
                grad_output,
                ctx.token_count,
            )
            return (
                grad_sorted,
                None,
                None,
                grad_primary_weights,
                None,
            )


    def cggr_grouped_gemm_autograd(
        sorted_x: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Autograd-aware entry point. Use this instead of cggr_grouped_gemm_triton
        whenever the call may be inside a training graph."""
        return CGGRGroupedGEMM.apply(sorted_x, expert_weights, expert_offsets)

    def hash_cggr_grouped_gemm_autograd(
        tokens: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_offsets: torch.Tensor,
        sorted_indices: torch.Tensor,
        inverse_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Autograd-aware hash-native grouped projection."""

        return HashCGGRGroupedGEMM.apply(
            tokens,
            expert_weights,
            expert_offsets,
            sorted_indices,
            inverse_indices,
        )

    def pair_hash_reduce_autograd(
        sorted_values: torch.Tensor,
        sorted_indices: torch.Tensor,
        inverse_indices: torch.Tensor,
        token_count: int,
        *,
        scale: float,
    ) -> torch.Tensor:
        """Autograd-aware hash unsort and route reduction."""

        return PairHashReduce.apply(
            sorted_values,
            sorted_indices,
            inverse_indices,
            int(token_count),
            float(scale),
        )

    def pair_hash_weighted_reduce_autograd(
        sorted_values: torch.Tensor,
        sorted_indices: torch.Tensor,
        inverse_indices: torch.Tensor,
        primary_weights: torch.Tensor,
        token_count: int,
    ) -> torch.Tensor:
        """Autograd-aware learned hash-pair reduction."""

        return PairHashWeightedReduce.apply(
            sorted_values,
            sorted_indices,
            inverse_indices,
            primary_weights,
            int(token_count),
        )

else:
    # PyTorch fallback when Triton is not available
    def fused_swiglu_triton(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """Fused SwiGLU activation - PyTorch fallback."""
        return F.silu(gate) * up

    def fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Fused RMSNorm - PyTorch fallback."""
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return x * rms * weight

    def cggr_grouped_gemm_autograd(
        sorted_x: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Autograd-aware entry point — PyTorch fallback when Triton is unavailable.

        Identical semantics to the Triton path: per-expert grouped GEMM where
        each token's row picks the slice of expert_weights it belongs to.
        Uses regular @ which PyTorch differentiates natively.
        """
        return grouped_gemm_pytorch(
            sorted_x, expert_weights, expert_offsets,
            torch.diff(expert_offsets),
        )

    def hash_cggr_grouped_gemm_autograd(
        tokens: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_offsets: torch.Tensor,
        sorted_indices: torch.Tensor,
        inverse_indices: torch.Tensor,
    ) -> torch.Tensor:
        """PyTorch reference for the hash-native grouped projection."""

        del inverse_indices
        source_rows = int(tokens.shape[0])
        sorted_tokens = tokens[
            sorted_indices.remainder(source_rows).long()
        ]
        return grouped_gemm_pytorch(
            sorted_tokens,
            expert_weights,
            expert_offsets,
            torch.diff(expert_offsets),
        )

    def pair_hash_reduce_autograd(
        sorted_values: torch.Tensor,
        sorted_indices: torch.Tensor,
        inverse_indices: torch.Tensor,
        token_count: int,
        *,
        scale: float,
    ) -> torch.Tensor:
        """PyTorch reference for fused hash unsort and top-2 reduction."""

        del sorted_indices
        primary = sorted_values[
            inverse_indices[:token_count].long()
        ]
        secondary = sorted_values[
            inverse_indices[token_count : 2 * token_count].long()
        ]
        return (primary + secondary) * float(scale)

    def pair_hash_weighted_reduce_autograd(
        sorted_values: torch.Tensor,
        sorted_indices: torch.Tensor,
        inverse_indices: torch.Tensor,
        primary_weights: torch.Tensor,
        token_count: int,
    ) -> torch.Tensor:
        """PyTorch reference for learned hash-pair route reduction."""

        del sorted_indices
        primary = sorted_values[
            inverse_indices[:token_count].long()
        ]
        secondary = sorted_values[
            inverse_indices[token_count : 2 * token_count].long()
        ]
        primary_weights = primary_weights.to(primary.dtype).unsqueeze(-1)
        return (
            primary * primary_weights
            + secondary * (1.0 - primary_weights)
        )


def pair_coverage_hash_expert_ids(
    token_ids: torch.Tensor,
    route_codes: torch.Tensor,
    expert_pairs: torch.Tensor,
    *,
    vocab_size: int,
) -> torch.Tensor:
    """Decode both pair-coverage hash routes in one CUDA planning kernel.

    The returned tensor has shape ``[2, *token_ids.shape]``. CUDA/Triton reads
    one compact uint8 hash code per token and derives both experts. CPU/MPS
    executes the identical decode with PyTorch, which also provides a reference
    for kernel-equivalence tests.
    """

    if route_codes.ndim != 1 or int(route_codes.numel()) != int(vocab_size):
        raise ValueError(
            "route_codes must contain one entry per vocabulary ID"
        )
    if route_codes.dtype != torch.uint8:
        raise ValueError("route_codes must use the compact uint8 format")
    if expert_pairs.ndim != 2 or expert_pairs.size(1) != 2:
        raise ValueError("expert_pairs must be shaped [pair_count, 2]")

    original_shape = token_ids.shape
    flat_token_ids = token_ids.contiguous().view(-1)
    total_tokens = int(flat_token_ids.numel())
    output = torch.empty(
        (2, total_tokens),
        dtype=torch.int32,
        device=token_ids.device,
    )
    if total_tokens == 0:
        return output.view(2, *original_shape)

    if HAS_TRITON and flat_token_ids.is_cuda:
        if not (route_codes.is_cuda and expert_pairs.is_cuda):
            raise ValueError(
                "hash metadata must be on the same CUDA device as token_ids"
            )
        block_size = 256
        _pair_coverage_hash_kernel[
            (triton.cdiv(total_tokens, block_size),)
        ](
            flat_token_ids,
            route_codes,
            expert_pairs,
            output,
            total_tokens=total_tokens,
            vocab_size=int(vocab_size),
            BLOCK_SIZE=block_size,
        )
        return output.view(2, *original_shape)

    clamped = flat_token_ids.clamp(0, int(vocab_size) - 1)
    codes = route_codes[clamped].to(torch.int64)
    pair_indices = codes & 0x7
    selected = expert_pairs[pair_indices.long()].long()
    swap = (codes & 0x8).ne(0)
    output[0] = torch.where(swap, selected[:, 1], selected[:, 0])
    output[1] = torch.where(swap, selected[:, 0], selected[:, 1])
    return output.view(2, *original_shape)


def sort_pair_hash_by_expert(
    token_ids: torch.Tensor,
    route_codes: torch.Tensor,
    expert_pairs: torch.Tensor,
    *,
    vocab_size: int,
    num_experts: int,
    return_inverse: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """Counting-partition two hash routes without materializing expert IDs.

    Returns assignment indices in ``[0, 2*N)`` where ``[0, N)`` denotes the
    primary route and ``[N, 2*N)`` the secondary route, followed by expert
    offsets and counts. With ``return_inverse=True``, the inverse assignment
    permutation is returned second so subsequent CGGR projections can read and
    reduce directly without materializing sorted token buffers. Unlike
    ``torch.sort``, work is linear in the number of tokens and the hash decode
    is fused into both counting/scatter kernels.
    """

    if route_codes.ndim != 1 or int(route_codes.numel()) != int(vocab_size):
        raise ValueError(
            "route_codes must contain one entry per vocabulary ID"
        )
    if route_codes.dtype != torch.uint8:
        raise ValueError("route_codes must use the compact uint8 format")
    if expert_pairs.ndim != 2 or expert_pairs.size(1) != 2:
        raise ValueError("expert_pairs must be shaped [pair_count, 2]")

    flat_token_ids = token_ids.contiguous().view(-1)
    total_tokens = int(flat_token_ids.numel())
    if total_tokens == 0:
        result = (
            torch.empty(0, dtype=torch.long, device=token_ids.device),
            torch.zeros(
                num_experts + 1,
                dtype=torch.int32,
                device=token_ids.device,
            ),
            torch.zeros(
                num_experts,
                dtype=torch.int32,
                device=token_ids.device,
            ),
        )
        if return_inverse:
            return result[0], result[0].clone(), result[1], result[2]
        return result

    if HAS_TRITON and flat_token_ids.is_cuda:
        if not (route_codes.is_cuda and expert_pairs.is_cuda):
            raise ValueError(
                "hash metadata must be on the same CUDA device as token_ids"
            )
        block_size = 256
        num_blocks = triton.cdiv(total_tokens, block_size)
        block_counts = torch.empty(
            (num_blocks, num_experts),
            dtype=torch.int32,
            device=token_ids.device,
        )
        _pair_hash_block_counts_kernel[(num_blocks,)](
            flat_token_ids,
            route_codes,
            expert_pairs,
            block_counts,
            total_tokens=total_tokens,
            vocab_size=int(vocab_size),
            num_experts=int(num_experts),
            BLOCK_SIZE=block_size,
        )
        block_offsets = torch.cumsum(
            block_counts, dim=0, dtype=torch.int32
        ) - block_counts
        expert_counts = block_counts.sum(dim=0, dtype=torch.int32)
        expert_offsets = torch.zeros(
            num_experts + 1,
            dtype=torch.int32,
            device=token_ids.device,
        )
        expert_offsets[1:] = torch.cumsum(
            expert_counts, dim=0, dtype=torch.int32
        )
        sorted_indices = torch.empty(
            2 * total_tokens,
            dtype=torch.long,
            device=token_ids.device,
        )
        inverse_indices = torch.empty_like(sorted_indices)
        _pair_hash_scatter_kernel[(num_blocks,)](
            flat_token_ids,
            route_codes,
            expert_pairs,
            block_offsets,
            expert_offsets,
            sorted_indices,
            inverse_indices,
            total_tokens=total_tokens,
            vocab_size=int(vocab_size),
            num_experts=int(num_experts),
            BLOCK_SIZE=block_size,
        )
        if return_inverse:
            return (
                sorted_indices,
                inverse_indices,
                expert_offsets,
                expert_counts,
            )
        return sorted_indices, expert_offsets, expert_counts

    expert_ids = pair_coverage_hash_expert_ids(
        flat_token_ids,
        route_codes,
        expert_pairs,
        vocab_size=vocab_size,
    ).reshape(-1)
    _, sorted_indices = torch.sort(expert_ids, stable=True)
    expert_counts = torch.bincount(
        expert_ids, minlength=num_experts
    ).to(torch.int32)
    expert_offsets = torch.zeros(
        num_experts + 1,
        dtype=torch.int32,
        device=token_ids.device,
    )
    expert_offsets[1:] = torch.cumsum(
        expert_counts, dim=0, dtype=torch.int32
    )
    if return_inverse:
        inverse_indices = torch.empty_like(sorted_indices)
        inverse_indices[sorted_indices] = torch.arange(
            sorted_indices.numel(),
            dtype=sorted_indices.dtype,
            device=sorted_indices.device,
        )
        return (
            sorted_indices,
            inverse_indices,
            expert_offsets,
            expert_counts,
        )
    return sorted_indices, expert_offsets, expert_counts


# =============================================================================
# TRITON-ACCELERATED TOKEN-ROUTED MLP
# =============================================================================

class TokenRoutedMLPTriton(nn.Module):
    """
    Token-Routed MLP with CGGR Triton optimization.

    5-6x faster than bmm version, 10x faster than loop version.

    Deterministic routing based on token ID:
    - Low token IDs -> Expert 0 (frequent tokens)
    - High token IDs -> Expert N-1 (rare tokens)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 4,
        vocab_size: int = 100000,
        hidden_act: str = "silu",
        use_cggr: bool = True,
        token_frequencies: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.vocab_size = vocab_size
        self.use_cggr = use_cggr and HAS_TRITON

        self.expert_intermediate_size = intermediate_size // num_experts

        # Expert weights [num_experts, in_dim, out_dim]
        self.gate_proj = nn.Parameter(
            torch.randn(num_experts, hidden_size, self.expert_intermediate_size) * 0.02
        )
        self.up_proj = nn.Parameter(
            torch.randn(num_experts, hidden_size, self.expert_intermediate_size) * 0.02
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, self.expert_intermediate_size, hidden_size) * 0.02
        )

        self.act_fn = F.silu if hidden_act == "silu" else F.gelu
        self._token_frequencies = token_frequencies

        # Token -> expert mapping
        self.register_buffer(
            "token_to_expert",
            self._create_token_mapping(vocab_size, num_experts),
        )

    def _create_token_mapping(self, vocab_size: int, num_experts: int) -> torch.Tensor:
        """Zipf-balanced round-robin if frequencies provided, else modulo."""
        if self._token_frequencies is not None:
            sorted_indices = self._token_frequencies.argsort(descending=True)
            mapping = torch.empty(vocab_size, dtype=torch.long)
            mapping[sorted_indices] = torch.arange(vocab_size, dtype=torch.long) % num_experts
            return mapping
        return torch.arange(vocab_size, dtype=torch.long) % num_experts

    def _cggr_forward(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        CGGR-optimized forward pass.

        Steps:
        1. Sort tokens by expert
        2. Grouped GEMM for gate_proj
        3. Grouped GEMM for up_proj
        4. Fused SwiGLU
        5. Grouped GEMM for down_proj
        6. Unsort back
        """
        total_tokens = hidden_states.shape[0]

        # Convert DTensor params to local (FSDP v2 compat)
        gate_proj = _to_local(self.gate_proj)
        up_proj = _to_local(self.up_proj)
        down_proj = _to_local(self.down_proj)

        # Sort by expert
        sorted_hidden, sorted_indices, expert_offsets, expert_counts = sort_tokens_by_expert(
            hidden_states, expert_ids, self.num_experts
        )

        # Gate projection
        if HAS_TRITON and hidden_states.is_cuda:
            gate_out = cggr_grouped_gemm_triton(sorted_hidden, gate_proj, expert_offsets)
        else:
            gate_out = grouped_gemm_pytorch(sorted_hidden, gate_proj, expert_offsets, expert_counts)

        # Up projection
        if HAS_TRITON and hidden_states.is_cuda:
            up_out = cggr_grouped_gemm_triton(sorted_hidden, up_proj, expert_offsets)
        else:
            up_out = grouped_gemm_pytorch(sorted_hidden, up_proj, expert_offsets, expert_counts)

        # Fused SwiGLU
        if HAS_TRITON and hidden_states.is_cuda:
            intermediate = fused_swiglu_triton(gate_out, up_out)
        else:
            intermediate = self.act_fn(gate_out) * up_out

        # Down projection
        if HAS_TRITON and hidden_states.is_cuda:
            sorted_output = cggr_grouped_gemm_triton(intermediate, down_proj, expert_offsets)
        else:
            sorted_output = grouped_gemm_pytorch(intermediate, down_proj, expert_offsets, expert_counts)

        # Unsort
        output = torch.zeros_like(sorted_output)
        output[sorted_indices] = sorted_output

        return output

    def _bmm_forward(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fallback bmm-based forward (v1).
        """
        # Gather weights for each token's expert (DTensor compat)
        gate_weights = _to_local(self.gate_proj)[expert_ids]
        up_weights = _to_local(self.up_proj)[expert_ids]
        down_weights = _to_local(self.down_proj)[expert_ids]

        # SwiGLU
        gate_out = torch.bmm(hidden_states.unsqueeze(1), gate_weights).squeeze(1)
        up_out = torch.bmm(hidden_states.unsqueeze(1), up_weights).squeeze(1)

        intermediate = self.act_fn(gate_out) * up_out

        output = torch.bmm(intermediate.unsqueeze(1), down_weights).squeeze(1)

        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            token_ids: [batch, seq_len] - for routing

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        if token_ids is None:
            expert_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=hidden_states.device)
        else:
            token_ids_clamped = token_ids.clamp(0, self.vocab_size - 1)
            expert_ids = self.token_to_expert[token_ids_clamped]

        # Flatten
        flat_hidden = hidden_states.view(-1, self.hidden_size)
        flat_expert_ids = expert_ids.view(-1)

        # Use CGGR if available
        if self.use_cggr:
            output = self._cggr_forward(flat_hidden, flat_expert_ids)
        else:
            output = self._bmm_forward(flat_hidden, flat_expert_ids)

        return output.view(batch_size, seq_len, self.hidden_size)


# =============================================================================
# BENCHMARK
# =============================================================================

def benchmark_token_routed_mlp(
    batch_size: int = 32,
    seq_len: int = 512,
    hidden_size: int = 1024,
    intermediate_size: int = 4096,
    num_experts: int = 4,
    vocab_size: int = 100000,
    n_iter: int = 100
):
    """Benchmark CGGR vs bmm Token-Routed MLP."""
    import time

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    # Create modules
    cggr_mlp = TokenRoutedMLPTriton(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        vocab_size=vocab_size,
        use_cggr=True,
    ).to(device).eval()

    bmm_mlp = TokenRoutedMLPTriton(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        vocab_size=vocab_size,
        use_cggr=False,
    ).to(device).eval()

    # Test inputs
    hidden = torch.randn(batch_size, seq_len, hidden_size, device=device)
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    for _ in range(10):
        _ = cggr_mlp(hidden, token_ids)
        _ = bmm_mlp(hidden, token_ids)
    torch.cuda.synchronize()

    # Benchmark CGGR
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = cggr_mlp(hidden, token_ids)
    torch.cuda.synchronize()
    cggr_time = (time.perf_counter() - start) / n_iter * 1000

    # Benchmark bmm
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = bmm_mlp(hidden, token_ids)
    torch.cuda.synchronize()
    bmm_time = (time.perf_counter() - start) / n_iter * 1000

    print(f"\nToken-Routed MLP Benchmark (batch={batch_size}, seq={seq_len}, h={hidden_size})")
    print(f"=" * 60)
    print(f"  BMM:      {bmm_time:.3f} ms (v1)")
    print(f"  CGGR:     {cggr_time:.3f} ms (v2)")
    print(f"  Speedup:  {bmm_time / cggr_time:.2f}x")
    print(f"=" * 60)

    return cggr_time, bmm_time


# =============================================================================
# FUSED RMSNORM KERNEL
# =============================================================================

if HAS_TRITON:
    @triton.jit
    def _fused_rmsnorm_kernel(
        x_ptr, weight_ptr, out_ptr,
        batch_size, seq_len, dim, eps,
        BLOCK_SIZE: tl.constexpr
    ):
        """Fused RMSNorm."""
        pid = tl.program_id(0)
        if pid >= batch_size * seq_len:
            return

        base_offset = pid * dim
        sum_sq = 0.0

        for i in range(0, dim, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < dim
            x = tl.load(x_ptr + base_offset + offsets, mask=mask, other=0.0)
            sum_sq += tl.sum(x * x, axis=0)

        inv_rms = 1.0 / tl.sqrt(sum_sq / dim + eps)

        for i in range(0, dim, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < dim
            x = tl.load(x_ptr + base_offset + offsets, mask=mask, other=0.0)
            weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
            tl.store(out_ptr + base_offset + offsets, x * inv_rms * weight, mask=mask)


def fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Fused RMSNorm."""
    if not HAS_TRITON or not x.is_cuda:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return x * rms * weight

    original_shape = x.shape
    if x.dim() == 2:
        batch_size, dim = x.shape
        seq_len = 1
        x_3d = x.unsqueeze(1)
    else:
        batch_size, seq_len, dim = x.shape
        x_3d = x

    out = torch.empty_like(x_3d)
    BLOCK_SIZE = min(1024, dim)

    _fused_rmsnorm_kernel[(batch_size * seq_len,)](
        x_3d.contiguous(), weight, out,
        batch_size, seq_len, dim, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out.view(original_shape)


# =============================================================================
# ROBOTICS CONTROL LOOP KERNEL - Pacific Prime Pattern (Token-Routed Variant)
# =============================================================================
# Inspired by real-time robotics control: sense -> process -> actuate
# Adapted for Token-Routed MLP with per-token expert routing
#
# Control Loop Pattern:
#   1. SENSE:    RMSNorm (observe normalized state)
#   2. PROCESS:  Token routing decision (select expert)
#   3. ACTUATE:  Expert MLP + Residual (apply specialized action)
# =============================================================================

if HAS_TRITON:
    @triton.jit
    def _fused_token_route_kernel(
        # Inputs
        x_ptr,              # [batch, seq, dim] - normalized input
        residual_ptr,       # [batch, seq, dim] - residual connection
        token_ids_ptr,      # [batch, seq] - token IDs for routing
        # Expert weights (simplified - single expert set for demo)
        gate_proj_ptr,      # [dim, intermediate]
        up_proj_ptr,        # [dim, intermediate]
        down_proj_ptr,      # [intermediate, dim]
        # Outputs
        x_out_ptr,          # [batch, seq, dim]
        # Dimensions
        batch_size,
        seq_len,
        dim,
        intermediate_dim,
        num_experts,
        BLOCK_SIZE: tl.constexpr
    ):
        """
        Fused token routing with expert selection.

        Each token routes to an expert based on token_id % num_experts.
        """
        pid = tl.program_id(0)
        token_idx = pid

        if token_idx >= batch_size * seq_len:
            return

        base = token_idx * dim

        # Load token ID for routing
        token_id = tl.load(token_ids_ptr + token_idx)
        expert_id = token_id % num_experts

        # Process token through selected expert
        for i in range(0, dim, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < dim

            x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)
            residual = tl.load(residual_ptr + base + offsets, mask=mask, other=0.0)

            # Simplified: just add residual (full MLP would require tiled GEMM)
            out = residual + x

            tl.store(x_out_ptr + base + offsets, out, mask=mask)


def fused_token_route_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    token_ids: torch.Tensor,
    num_experts: int = 8
) -> torch.Tensor:
    """
    Fused token routing with residual.

    Robotics pattern:
        SENSE: Token ID observation
        PROCESS: Expert routing decision
        ACTUATE: Residual connection

    Args:
        x: Processed hidden states [batch, seq, dim]
        residual: Residual connection [batch, seq, dim]
        token_ids: Token IDs for routing [batch, seq]
        num_experts: Number of experts

    Returns:
        out: residual + x (with routing metadata)
    """
    # For now, simple residual - full routing in TokenRoutedMLPTriton
    return residual + x


class RoboticsTokenRoutedLayer(torch.nn.Module):
    """
    Robotics-inspired Token-Routed layer with fused CUDA operations.

    Control loop pattern:
        1. SENSE:    RMSNorm (observe state)
        2. PROCESS:  Token routing (select expert per token)
        3. ACTUATE:  Expert MLP + Residual (apply specialized action)

    Uses CGGR optimization for expert computation.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        vocab_size: int = 32000,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.eps = eps

        # RMSNorm weight
        self.norm_weight = torch.nn.Parameter(torch.ones(hidden_size))

        # Token-Routed MLP (uses CGGR if available)
        self.mlp = TokenRoutedMLPTriton(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            vocab_size=vocab_size,
            use_cggr=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with robotics control loop.

        Args:
            x: [batch, seq, dim]
            token_ids: [batch, seq] token IDs for routing

        Returns:
            out: [batch, seq, dim]
        """
        residual = x

        # === SENSE: RMSNorm ===
        x_normed = fused_rmsnorm(x, _to_local(self.norm_weight), self.eps)

        # === PROCESS + ACTUATE: Token-Routed MLP ===
        mlp_out = self.mlp(x_normed, token_ids=token_ids)

        # Residual
        out = residual + mlp_out

        return out


if __name__ == "__main__":
    benchmark_token_routed_mlp()
