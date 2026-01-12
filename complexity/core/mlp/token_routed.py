"""
Token-Routed MLP (Deterministic MoE) - Complexity Framework Innovation.

Routes tokens to specialized experts based on token ID.
Deterministic routing = no router to learn, stable, 100% parallel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base import MLPBase, MLPConfig
from ..registry import register_mlp


@register_mlp("token_routed")
@register_mlp("deterministic_moe")
@register_mlp("complexity")
class TokenRoutedMLP(MLPBase):
    """
    Token-Routed MLP (Deterministic MoE).

    Each token is routed to a specific expert based on its token ID.
    This is a key innovation of the Complexity framework.

    Benefits:
    - 1/num_experts compute per token (faster)
    - Specialized experts per token type
    - Deterministic = stable training, no load balancing loss
    - 100% parallel (no routing decisions at runtime)

    Token routing strategy:
        Expert assignment = token_id % num_experts

    This ensures uniform distribution regardless of token frequency.
    """

    def __init__(self, config: MLPConfig):
        super().__init__(config)

        self.num_experts = config.num_experts
        self.vocab_size = config.vocab_size

        # Expert size = intermediate_size / num_experts (same total params)
        self.expert_intermediate_size = self.intermediate_size // self.num_experts

        # Create experts (SwiGLU style)
        self.experts = nn.ModuleList([
            Expert(
                self.hidden_size,
                self.expert_intermediate_size,
                config.hidden_act,
                config.bias,
            )
            for _ in range(self.num_experts)
        ])

        # Precompute token -> expert mapping
        self.register_buffer(
            "token_to_expert",
            self._create_token_mapping(self.vocab_size, self.num_experts),
        )

    def _create_token_mapping(self, vocab_size: int, num_experts: int) -> torch.Tensor:
        """
        Create deterministic mapping from token ID to expert ID.

        Uses modulo routing for uniform distribution.
        """
        return torch.arange(vocab_size, dtype=torch.long) % num_experts

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with token-based routing.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            token_ids: [batch, seq_len] - original token IDs for routing

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        if token_ids is None:
            # Fallback: use all experts equally (for inference without token_ids)
            return self._forward_all_experts(hidden_states)

        # Get expert assignment for each token
        token_ids_clamped = token_ids.clamp(0, self.vocab_size - 1)
        expert_ids = self.token_to_expert[token_ids_clamped]  # [batch, seq_len]

        # Process each expert's tokens
        output = torch.zeros_like(hidden_states)

        for expert_id in range(self.num_experts):
            # Mask for tokens routed to this expert
            mask = (expert_ids == expert_id)  # [batch, seq_len]

            if not mask.any():
                continue

            # Get tokens for this expert
            expert_input = hidden_states[mask]  # [num_tokens, hidden_size]

            # Process through expert
            expert_output = self.experts[expert_id](expert_input)

            # Put back in output
            output[mask] = expert_output

        return output

    def _forward_all_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Fallback: average all experts (for inference without token_ids)."""
        outputs = [expert(hidden_states) for expert in self.experts]
        return torch.stack(outputs, dim=0).mean(dim=0)


class Expert(nn.Module):
    """Single expert MLP (SwiGLU)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        bias: bool = False,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act_fn = F.silu if hidden_act == "silu" else F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


@register_mlp("token_routed_parallel")
@register_mlp("batched_moe")
class TokenRoutedMLPParallel(MLPBase):
    """
    Optimized Token-Routed MLP using batched operations.

    Instead of looping over experts, process all at once with gather.
    Better GPU utilization for large batches.
    """

    def __init__(self, config: MLPConfig):
        super().__init__(config)

        self.num_experts = config.num_experts
        self.vocab_size = config.vocab_size
        self.expert_intermediate_size = self.intermediate_size // self.num_experts

        # Batched expert weights [num_experts, hidden, intermediate]
        self.gate_proj = nn.Parameter(
            torch.randn(self.num_experts, self.hidden_size, self.expert_intermediate_size) * 0.02
        )
        self.up_proj = nn.Parameter(
            torch.randn(self.num_experts, self.hidden_size, self.expert_intermediate_size) * 0.02
        )
        self.down_proj = nn.Parameter(
            torch.randn(self.num_experts, self.expert_intermediate_size, self.hidden_size) * 0.02
        )

        self.act_fn = F.silu if config.hidden_act == "silu" else F.gelu

        # Token mapping
        self.register_buffer(
            "token_to_expert",
            torch.arange(self.vocab_size, dtype=torch.long) % self.num_experts,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Batched forward pass.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            token_ids: [batch, seq_len]

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
        flat_hidden = hidden_states.view(-1, self.hidden_size)  # [B*S, H]
        flat_expert_ids = expert_ids.view(-1)  # [B*S]

        # Gather weights for each token's expert
        gate_weights = self.gate_proj[flat_expert_ids]  # [B*S, H, I]
        up_weights = self.up_proj[flat_expert_ids]      # [B*S, H, I]
        down_weights = self.down_proj[flat_expert_ids]  # [B*S, I, H]

        # SwiGLU: down(act(gate(x)) * up(x))
        gate_out = torch.bmm(flat_hidden.unsqueeze(1), gate_weights).squeeze(1)
        up_out = torch.bmm(flat_hidden.unsqueeze(1), up_weights).squeeze(1)

        intermediate = self.act_fn(gate_out) * up_out

        output = torch.bmm(intermediate.unsqueeze(1), down_weights).squeeze(1)

        return output.view(batch_size, seq_len, self.hidden_size)
