"""
Token-Routed MLP — Deterministic Mixture-of-Experts.

Innovation from Complexity-ML (2026):
  Each token is routed to exactly one expert based on its token ID.
  Routing is deterministic (no learned router, no load-balancing loss).

Features:
  - Zipf-balanced routing via greedy bin-packing on token frequencies
  - Shared Lexical Expert (dense SwiGLU all tokens pass through)
  - Sparse dispatch (loop over experts with masking)
  - Falls back to simple modulo routing without token frequencies

Usage:
    config = MLPConfig(hidden_size=512, intermediate_size=2048, num_experts=4)
    mlp = TokenRoutedMLP(config)
    out = mlp(hidden_states, token_ids=token_ids)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

from .base import MLPBase, MLPConfig
from .fused_activations import fused_silu_mul
from ..registry import register_mlp

logger = logging.getLogger(__name__)

# Try to import CGGR acceleration
try:
    from complexity_cuda.triton_token_routed import (
        sort_tokens_by_expert,
        cggr_grouped_gemm_triton,
        cggr_grouped_gemm_autograd,
        grouped_gemm_pytorch,
        fused_swiglu_triton,
        HAS_TRITON,
    )
    HAS_CGGR = HAS_TRITON
except ImportError:
    HAS_CGGR = False
    cggr_grouped_gemm_autograd = None

    def sort_tokens_by_expert(tokens, expert_ids, num_experts):
        """Pure-PyTorch fallback — stable sort + cumsum offsets.
        Used when complexity_cuda is not installed (Mac/CPU dev setups).
        """
        sorted_expert_ids, sorted_indices = torch.sort(expert_ids, stable=True)
        sorted_tokens = tokens[sorted_indices]
        expert_counts = torch.bincount(expert_ids, minlength=num_experts)
        expert_offsets = torch.zeros(num_experts + 1, dtype=torch.long, device=tokens.device)
        expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)
        return sorted_tokens, sorted_indices, expert_offsets, expert_counts


def _to_local(t: torch.Tensor) -> torch.Tensor:
    """Convert DTensor to local tensor (FSDP v2 compat)."""
    if hasattr(t, 'to_local'):
        return t.to_local()
    return t


@register_mlp("token_routed")
@register_mlp("sort_split")
@register_mlp("sort_split_moe")
@register_mlp("deterministic_moe")
@register_mlp("complexity")
class TokenRoutedMLP(MLPBase):
    """
    Token-Routed MLP with Shared Lexical Expert.

    Routes tokens to experts deterministically (token_id -> expert_id)
    via Zipf-balanced bin-packing, then dispatches with sparse masking.
    """

    def __init__(self, config: MLPConfig):
        super().__init__(config)

        self.num_experts = config.num_experts
        self.vocab_size = config.vocab_size
        self.expert_intermediate_size = self.intermediate_size // self.num_experts
        # Top-K deterministic: each token activates K experts via cyclic shift
        # of the Zipf primary. K=1 is the classic single-expert Zipf routing.
        # K>1 increases active FLOPs linearly while keeping zero learned routing
        # and zero load-balance loss (Zipf guarantees uniform load at every k).
        self.top_k = max(1, int(getattr(config, "top_k", 1)))

        # Routed expert weights: gate, up, down.
        # down_proj_w will be re-initialized with GPT-2 residual scaling by
        # ComplexityModel._init_residual_scaling() after the module tree is built.
        self.gate_proj_w = nn.Parameter(
            torch.randn(self.num_experts, self.hidden_size,
                        self.expert_intermediate_size) * 0.02
        )
        self.up_proj_w = nn.Parameter(
            torch.randn(self.num_experts, self.hidden_size,
                        self.expert_intermediate_size) * 0.02
        )
        self.down_proj_w = nn.Parameter(
            torch.randn(self.num_experts, self.expert_intermediate_size,
                        self.hidden_size) * 0.02
        )

        # Shared lexical expert: dense SwiGLU all tokens pass through.
        # Default size = intermediate_size (full dense width). shared_down is
        # also rescaled by _init_residual_scaling() (residual output projection).
        self.use_shared_expert = getattr(config, 'shared_expert', False)
        if self.use_shared_expert:
            shared_size = getattr(config, 'shared_intermediate_size', None) or self.intermediate_size
            self.shared_gate = nn.Linear(self.hidden_size, shared_size, bias=False)
            self.shared_up = nn.Linear(self.hidden_size, shared_size, bias=False)
            self.shared_down = nn.Linear(shared_size, self.hidden_size, bias=False)

        # Token -> expert mapping (Zipf-balanced or modulo)
        self.register_buffer(
            "token_to_expert",
            self._create_token_mapping(self.vocab_size, self.num_experts),
        )

        # Expert utilization counters — accumulated across forward calls.
        # Reset manually via reset_expert_counts() (e.g. once per CSV log step).
        # Non-persistent so checkpoint save/load doesn't snapshot stale counters.
        self.register_buffer(
            "expert_counts",
            torch.zeros(self.num_experts, dtype=torch.long),
            persistent=False,
        )

    def reset_expert_counts(self) -> None:
        """Zero the expert utilization counter. Call once per log interval."""
        self.expert_counts.zero_()

    def get_expert_counts(self) -> torch.Tensor:
        """Return current expert counts [num_experts] on-device."""
        return self.expert_counts

    def _create_token_mapping(self, vocab_size: int, num_experts: int) -> torch.Tensor:
        """
        Create deterministic mapping from token ID to expert ID.

        With token_frequencies: greedy bin-packing so each expert gets
        equal corpus frequency load (Zipf-balanced).
        Without: simple modulo fallback (token_id % E).
        """
        if getattr(self.config, 'token_frequencies', None) is not None:
            freqs = self.config.token_frequencies
            sorted_indices = freqs.argsort(descending=True)
            mapping = torch.empty(vocab_size, dtype=torch.long)
            expert_loads = [0.0] * num_experts
            for rank_pos in range(vocab_size):
                token_id = sorted_indices[rank_pos].item()
                e = min(range(num_experts), key=lambda i: expert_loads[i])
                mapping[token_id] = e
                expert_loads[e] += freqs[token_id].item()
            return mapping
        return torch.arange(vocab_size, dtype=torch.long) % num_experts

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with sparse dispatch.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            token_ids: [batch, seq_len] — original input token IDs

        Returns:
            output: [batch, seq_len, hidden_size]
                    = SharedMLP(x) + Expert_e(x)
        """
        B, S, H = hidden_states.shape

        if token_ids is None:
            return self._forward_all_experts(hidden_states)

        # Look up expert assignment per token
        token_ids_clamped = token_ids.clamp(0, self.vocab_size - 1)
        expert_ids = self.token_to_expert[token_ids_clamped]  # [B, S]

        flat_x = hidden_states.view(-1, H)
        flat_expert_ids = expert_ids.view(-1)

        # Track expert utilization (in-place, non-differentiable)
        with torch.no_grad():
            batch_counts = torch.bincount(flat_expert_ids, minlength=self.num_experts)
            self.expert_counts += batch_counts.to(self.expert_counts.dtype)

        # Shared expert (dense, all tokens) — fused SwiGLU via Liger on CUDA
        if self.use_shared_expert:
            shared_out = self.shared_down(
                fused_silu_mul(self.shared_gate(flat_x), self.shared_up(flat_x))
            ).to(flat_x.dtype)
        else:
            shared_out = 0

        # Routed experts — CGGR Triton (autograd-aware) or sparse-loop fallback.
        # CGGRGroupedGEMM (in complexity_cuda.triton_token_routed) wraps the
        # forward-only Triton kernel with a proper torch.autograd.Function so
        # gradients flow back to gate/up/down_proj_w. fused_swiglu_triton stays
        # forward-only so we use plain F.silu(gate) * up which PyTorch
        # differentiates natively.
        gate_w = _to_local(self.gate_proj_w)
        up_w = _to_local(self.up_proj_w)
        down_w = _to_local(self.down_proj_w)

        # FSDP diagnostic: verify expert weights are fully gathered (not sharded)
        if not hasattr(self, "_fsdp_checked"):
            expected = (self.num_experts, self.hidden_size, self.expert_intermediate_size)
            if gate_w.shape != expected:
                logger.error(
                    f"FSDP BUG: gate_proj_w shape {tuple(gate_w.shape)} != expected {expected}. "
                    f"Expert weights are SHARDED — matmuls are corrupted!"
                )
            else:
                logger.info(f"FSDP OK: expert weights shape {tuple(gate_w.shape)}")
            self._fsdp_checked = True

        # Routing path selection:
        #   - bmm (default, universal) : one batched matmul over all experts,
        #     cuBLAS on CUDA, MPS-native on Apple, perfectly autograd-friendly.
        #   - CGGR Triton (cuda + opt-in) : custom grouped-GEMM kernel,
        #     kept as fallback via config flag `use_cggr=True`.
        use_cggr = (getattr(self.config, "use_cggr", False)
                    and HAS_CGGR and flat_x.is_cuda
                    and cggr_grouped_gemm_autograd is not None)

        # Top-K deterministic Zipf: dispatch K times with cyclic-shifted expert
        # IDs `(primary + k) % E`. Because Zipf primary is already perfectly
        # balanced (each expert ≈ 1/E of the frequency mass), any cyclic shift
        # preserves that balance. Outputs are averaged.
        routed_out = torch.zeros_like(flat_x)
        for k in range(self.top_k):
            if k == 0:
                expert_ids_k = flat_expert_ids
            else:
                expert_ids_k = (flat_expert_ids + k) % self.num_experts
            part = self._dispatch_once(
                flat_x, expert_ids_k, gate_w, up_w, down_w, use_cggr, H,
            )
            routed_out = routed_out + part
        if self.top_k > 1:
            routed_out = routed_out / float(self.top_k)

        out = shared_out + routed_out
        return out.view(B, S, H)

    def _dispatch_once(
        self,
        flat_x: torch.Tensor,
        expert_ids: torch.Tensor,
        gate_w: torch.Tensor,
        up_w: torch.Tensor,
        down_w: torch.Tensor,
        use_cggr: bool,
        H: int,
    ) -> torch.Tensor:
        """Run one expert-dispatch pass for a given [N] expert assignment.

        Returns an [N, H] tensor in the same token order as flat_x.
        """
        sorted_x, sorted_idx, expert_offsets, expert_counts = sort_tokens_by_expert(
            flat_x, expert_ids, self.num_experts
        )

        if use_cggr:
            gate_out = cggr_grouped_gemm_autograd(sorted_x, gate_w, expert_offsets)
            up_out = cggr_grouped_gemm_autograd(sorted_x, up_w, expert_offsets)
            intermediate = fused_silu_mul(gate_out, up_out)
            sorted_routed = cggr_grouped_gemm_autograd(intermediate, down_w, expert_offsets)

            out = torch.empty_like(flat_x)
            out[sorted_idx] = sorted_routed.to(out.dtype)
            return out

        # bmm path — pad each bucket to max(counts), three torch.bmm.
        counts_cpu = expert_counts.cpu().tolist()
        offsets_cpu = expert_offsets.cpu().tolist()
        capacity = max(counts_cpu) if counts_cpu else 0

        if capacity == 0:
            return torch.zeros_like(flat_x)

        padded = sorted_x.new_zeros(self.num_experts, capacity, H)
        for e in range(self.num_experts):
            n = counts_cpu[e]
            if n == 0:
                continue
            s = offsets_cpu[e]
            padded[e, :n] = sorted_x[s:s + n]

        gate = torch.bmm(padded, gate_w)
        up = torch.bmm(padded, up_w)
        inter = fused_silu_mul(gate, up)
        out_padded = torch.bmm(inter, down_w)

        out = torch.empty_like(flat_x)
        for e in range(self.num_experts):
            n = counts_cpu[e]
            if n == 0:
                continue
            s = offsets_cpu[e]
            out[sorted_idx[s:s + n]] = out_padded[e, :n].to(out.dtype)
        return out

    def _forward_all_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Fallback: average all experts (inference without token_ids)."""
        flat = hidden_states.view(-1, self.hidden_size)
        gate_w = _to_local(self.gate_proj_w)
        up_w = _to_local(self.up_proj_w)
        down_w = _to_local(self.down_proj_w)
        out = torch.zeros_like(flat)
        for e in range(self.num_experts):
            gate_e = flat @ gate_w[e]
            up_e = flat @ up_w[e]
            out = out + fused_silu_mul(gate_e, up_e) @ down_w[e]
        out = out / self.num_experts
        if self.use_shared_expert:
            shared = self.shared_down(
                fused_silu_mul(self.shared_gate(flat), self.shared_up(flat))
            )
            out = out + shared
        return out.view_as(hidden_states)
