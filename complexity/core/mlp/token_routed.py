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

        # Learnable α gate on routed path: out = shared + α·routed
        self.routed_gate = getattr(config, 'routed_gate', False)
        if self.routed_gate:
            alpha_init = float(getattr(config, 'routed_gate_init', 0.0))
            self.routed_alpha = nn.Parameter(torch.full((1,), alpha_init))

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

        # Shared expert (dense, all tokens)
        if self.use_shared_expert:
            shared_out = self.shared_down(
                F.silu(self.shared_gate(flat_x)) * self.shared_up(flat_x)
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

        use_cggr = HAS_CGGR and flat_x.is_cuda and cggr_grouped_gemm_autograd is not None

        if use_cggr:
            sorted_x, sorted_idx, expert_offsets, expert_counts = sort_tokens_by_expert(
                flat_x, flat_expert_ids, self.num_experts
            )
            gate_out = cggr_grouped_gemm_autograd(sorted_x, gate_w, expert_offsets)
            up_out = cggr_grouped_gemm_autograd(sorted_x, up_w, expert_offsets)
            intermediate = F.silu(gate_out) * up_out  # autograd-friendly SwiGLU
            sorted_routed = cggr_grouped_gemm_autograd(intermediate, down_w, expert_offsets)
            routed_out = torch.zeros_like(flat_x)
            routed_out[sorted_idx] = sorted_routed.to(routed_out.dtype)
        else:
            # Autograd-aware path: sparse dispatch (loop over experts with masking).
            # Used during training (gradients flow back to gate_w/up_w/down_w) and as
            # the CPU/no-Triton fallback.
            routed_out = torch.zeros_like(flat_x)
            for e in range(self.num_experts):
                mask = (flat_expert_ids == e)
                if not mask.any():
                    continue
                x_e = flat_x[mask]
                gate_e = x_e @ gate_w[e]
                up_e = x_e @ up_w[e]
                inter_e = F.silu(gate_e) * up_e
                routed_out[mask] = (inter_e @ down_w[e]).to(routed_out.dtype)

        if self.routed_gate:
            out = shared_out + self.routed_alpha * routed_out
        else:
            out = shared_out + routed_out
        return out.view(B, S, H)

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
            out = out + (F.silu(gate_e) * up_e) @ down_w[e]
        out = out / self.num_experts
        if self.use_shared_expert:
            shared = self.shared_down(
                F.silu(self.shared_gate(flat)) * self.shared_up(flat)
            )
            out = out + shared
        return out.view_as(hidden_states)
