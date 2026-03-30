"""
Mixtral-style MoE — Learned router with top-k gating and load balancing loss.

Standard MoE baseline for comparison with Token-Routed MLP.
Uses a learned router (nn.Linear + softmax + top-1) to assign tokens to experts,
with an auxiliary load balancing loss to prevent expert collapse.

Reference: Mixtral of Experts (Jiang et al., 2024)

Usage:
    config = MLPConfig(hidden_size=512, intermediate_size=2048, num_experts=4)
    mlp = MixtralMoE(config)
    out = mlp(hidden_states)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base import MLPBase, MLPConfig
from ..registry import register_mlp


@register_mlp("mixtral")
@register_mlp("learned_router")
@register_mlp("standard_moe")
class MixtralMoE(MLPBase):
    """
    Mixtral-style MoE with learned router and load balancing.

    Each token is routed to top-1 expert via a learned router
    (nn.Linear → softmax → argmax). An auxiliary load balancing
    loss encourages uniform expert utilization.

    This serves as the MoE baseline for comparison with
    deterministic Token-Routed MLP.
    """

    # Weight for the auxiliary load balancing loss
    LOAD_BALANCE_WEIGHT = 0.01

    def __init__(self, config: MLPConfig):
        super().__init__(config)

        self.num_experts = config.num_experts
        self.expert_intermediate_size = self.intermediate_size // self.num_experts

        # Learned router: hidden_size → num_experts
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        # Expert weights: [E, hidden, inter*2] and [E, inter, hidden]
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size,
                        self.expert_intermediate_size * 2)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.expert_intermediate_size,
                        self.hidden_size)
        )

        # Shared expert (optional, for fair comparison)
        self.shared_expert = None
        if getattr(config, 'shared_expert', False):
            shared_size = self.expert_intermediate_size
            self.shared_gate = nn.Linear(self.hidden_size, shared_size, bias=False)
            self.shared_up = nn.Linear(self.hidden_size, shared_size, bias=False)
            self.shared_down = nn.Linear(shared_size, self.hidden_size, bias=False)
            self.shared_expert = True

        self._init_weights()

        # Store last load balancing loss for training
        self.last_aux_loss = 0.0

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.gate_up_proj, a=5**0.5)
        nn.init.zeros_(self.down_proj)
        nn.init.kaiming_uniform_(self.router.weight, a=5**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with learned routing.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            token_ids: ignored (kept for API compat with TokenRoutedMLP)

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        N = batch_size * seq_len
        chunk = N // self.num_experts
        E = self.num_experts
        flat_x = hidden_states.reshape(N, self.hidden_size)

        # Router: learned gating
        router_logits = self.router(flat_x)  # [N, E]
        router_probs = F.softmax(router_logits, dim=-1)  # [N, E]
        expert_ids = router_logits.argmax(dim=-1)  # [N]

        # Load balancing loss (Mixtral-style)
        # Encourages uniform expert assignment
        if self.training:
            # Fraction of tokens assigned to each expert
            tokens_per_expert = F.one_hot(expert_ids, E).float().mean(dim=0)  # [E]
            # Average router probability per expert
            router_prob_per_expert = router_probs.mean(dim=0)  # [E]
            # Auxiliary loss: dot product (penalizes correlation between assignment and probability)
            self.last_aux_loss = (E * (tokens_per_expert * router_prob_per_expert).sum()).item()

        # Sort-and-split dispatch (same as TokenRoutedMLP)
        sort_idx = expert_ids.argsort(stable=True)
        sorted_x = flat_x[sort_idx]

        # bmm gate+up
        gu = torch.bmm(
            sorted_x.view(E, chunk, self.hidden_size), self.gate_up_proj
        )
        gate, up = gu.chunk(2, dim=-1)
        activated = F.silu(gate) * up

        # bmm down
        sorted_out = torch.bmm(activated, self.down_proj).reshape(N, self.hidden_size)

        # Weight output by router probability (soft gating)
        sorted_probs = router_probs[sort_idx]
        expert_idx_sorted = expert_ids[sort_idx]
        # Gather the probability for the selected expert
        selected_probs = sorted_probs.gather(1, expert_idx_sorted.unsqueeze(-1)).squeeze(-1)
        sorted_out = sorted_out * selected_probs.unsqueeze(-1)

        out = torch.zeros(N, self.hidden_size, device=flat_x.device, dtype=sorted_out.dtype)
        out[sort_idx] = sorted_out

        # Shared expert
        if self.shared_expert:
            shared_out = self.shared_down(
                F.silu(self.shared_gate(flat_x)) * self.shared_up(flat_x)
            ).to(flat_x.dtype)
            out = out + shared_out

        return out.reshape(batch_size, seq_len, self.hidden_size)
