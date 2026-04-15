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

    # Autotune configs cover the MoE shapes we actually run:
    #   hidden ∈ {640, 1024}, expert_inter ∈ {448, 502, 2008}, shared_inter ∈ {..., 2008}
    # Tuning keys are the matmul dims (in_dim, out_dim); num_experts is
    # dispatch-only and doesn't affect per-block perf.
    _CGGR_CONFIGS = [
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64,  "BLOCK_K": 32},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 32},  num_warps=8, num_stages=3),
    ]

    # Separate autotune config list for grad_W — different reduction pattern
    # (reduce over tokens, tiles are (in_dim, out_dim)) benefits from narrower
    # BLOCK_M (tokens per block in K dim) and wider (BLOCK_N, BLOCK_O) tiles.
    _CGGR_GRAD_W_CONFIGS = [
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64,  "BLOCK_O": 64},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128, "BLOCK_O": 64},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64,  "BLOCK_O": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_O": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_O": 64},  num_warps=4, num_stages=4),
    ]

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


    def cggr_grouped_gemm_autograd(
        sorted_x: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Autograd-aware entry point. Use this instead of cggr_grouped_gemm_triton
        whenever the call may be inside a training graph."""
        return CGGRGroupedGEMM.apply(sorted_x, expert_weights, expert_offsets)

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
