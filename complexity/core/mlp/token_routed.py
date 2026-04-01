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

from .base import MLPBase, MLPConfig
from ..registry import register_mlp


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

        # Routed expert weights: gate, up, down — separate like supplementary code
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

        # Shared lexical expert: dense SwiGLU all tokens pass through
        self.use_shared_expert = getattr(config, 'shared_expert', False)
        if self.use_shared_expert:
            shared_size = getattr(config, 'shared_intermediate_size', None) or self.expert_intermediate_size
            self.shared_gate = nn.Linear(self.hidden_size, shared_size, bias=False)
            self.shared_up = nn.Linear(self.hidden_size, shared_size, bias=False)
            self.shared_down = nn.Linear(shared_size, self.hidden_size, bias=False)

        # Token -> expert mapping (Zipf-balanced or modulo)
        self.register_buffer(
            "token_to_expert",
            self._create_token_mapping(self.vocab_size, self.num_experts),
        )

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

        # Shared expert (dense, all tokens)
        if self.use_shared_expert:
            shared_out = self.shared_down(
                F.silu(self.shared_gate(flat_x)) * self.shared_up(flat_x)
            ).to(flat_x.dtype)
        else:
            shared_out = 0

        # Routed experts (sparse dispatch)
        routed_out = torch.zeros_like(flat_x)
        for e in range(self.num_experts):
            mask = (flat_expert_ids == e)
            if not mask.any():
                continue
            x_e = flat_x[mask]
            gate_e = x_e @ self.gate_proj_w[e]
            up_e = x_e @ self.up_proj_w[e]
            inter_e = F.silu(gate_e) * up_e
            routed_out[mask] = (inter_e @ self.down_proj_w[e]).to(routed_out.dtype)

        out = routed_out + shared_out
        return out.view(B, S, H)

    def _forward_all_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Fallback: average all experts (inference without token_ids)."""
        flat = hidden_states.view(-1, self.hidden_size)
        out = torch.zeros_like(flat)
        for e in range(self.num_experts):
            gate_e = flat @ self.gate_proj_w[e]
            up_e = flat @ self.up_proj_w[e]
            out = out + (F.silu(gate_e) * up_e) @ self.down_proj_w[e]
        out = out / self.num_experts
        if self.use_shared_expert:
            shared = self.shared_down(
                F.silu(self.shared_gate(flat)) * self.shared_up(flat)
            )
            out = out + shared
        return out.view_as(hidden_states)
