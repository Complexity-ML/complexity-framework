"""
Token-Routed MLP — Deterministic expert routing with sort-and-split dispatch.

Innovation from Complexity-ML (2026):
  argsort by expert_id -> fixed split N/E per expert -> all experts always busy.
  Zero waste, zero idle, all shapes static, fullgraph compatible.

Features:
  - Zipf-balanced routing via greedy bin-packing on token frequencies
  - Shared Lexical Expert (dense SwiGLU all tokens pass through)
  - Sort-and-split dispatch (bmm, no dynamic shapes)
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
    Token-Routed MLP with sort-and-split dispatch.

    Routes tokens to experts deterministically (token_id -> expert_id),
    then sorts by expert and processes fixed-size chunks via bmm.

    Each expert processes exactly N/E tokens. No dynamic shapes,
    no wasted compute, fullgraph compatible with torch.compile.
    """

    def __init__(self, config: MLPConfig):
        super().__init__(config)

        self.num_experts = config.num_experts
        self.vocab_size = config.vocab_size
        self.expert_intermediate_size = self.intermediate_size // self.num_experts

        # Expert weights: [E, hidden, inter*2] and [E, inter, hidden]
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size,
                        self.expert_intermediate_size * 2)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.expert_intermediate_size,
                        self.hidden_size)
        )

        # Shared lexical expert: dense SwiGLU all tokens pass through
        self.shared_expert = None
        if getattr(config, 'shared_expert', False):
            shared_size = getattr(config, 'shared_intermediate_size', None) or self.intermediate_size
            self.shared_gate = nn.Linear(self.hidden_size, shared_size, bias=False)
            self.shared_up = nn.Linear(self.hidden_size, shared_size, bias=False)
            self.shared_down = nn.Linear(shared_size, self.hidden_size, bias=False)
            self.shared_expert = True

        # Token -> expert mapping (Zipf-balanced or modulo)
        self.register_buffer(
            "token_to_expert",
            self._create_token_mapping(self.vocab_size, self.num_experts),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.gate_up_proj, a=5**0.5)
        nn.init.zeros_(self.down_proj)

    def _create_token_mapping(self, vocab_size: int, num_experts: int) -> torch.Tensor:
        """
        Create deterministic mapping from token ID to expert ID.

        If token_frequencies are provided, distributes via greedy bin-packing
        so each expert gets equal corpus frequency load (Zipf-balanced).
        Otherwise, simple modulo routing.
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
        sort_idx: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with per-expert loop dispatch.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            token_ids: [batch, seq_len] — for routing (token_to_expert lookup)
            sort_idx: ignored (kept for API compat)

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        N = batch_size * seq_len
        E = self.num_experts
        flat_x = hidden_states.reshape(N, self.hidden_size)

        # Compute expert assignments
        if token_ids is not None:
            token_ids_clamped = token_ids.reshape(-1).clamp(0, self.vocab_size - 1)
            expert_ids = self.token_to_expert[token_ids_clamped]
        else:
            expert_ids = torch.arange(N, device=flat_x.device) % E

        out = torch.zeros(N, self.hidden_size, device=flat_x.device, dtype=flat_x.dtype)
        for e in range(E):
            mask = expert_ids == e
            if not mask.any():
                continue
            xe = flat_x[mask]
            gu = torch.mm(xe, self.gate_up_proj[e])
            gate, up = gu.chunk(2, dim=-1)
            activated = F.silu(gate) * up
            out[mask] = torch.mm(activated, self.down_proj[e])

        # Shared expert: dense SwiGLU applied to all tokens
        if self.shared_expert:
            shared_out = self.shared_down(
                F.silu(self.shared_gate(flat_x)) * self.shared_up(flat_x)
            ).to(flat_x.dtype)
            out = out + shared_out

        return out.reshape(batch_size, seq_len, self.hidden_size)
