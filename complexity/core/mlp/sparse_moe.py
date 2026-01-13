"""
Sparse MoE (Mixture of Experts) with Learned Routing.

Standard MoE implementation similar to Mixtral, GPT-4, Switch Transformer.
Uses a learned router network to select top-k experts per token.

Note: For our deterministic approach, see TokenRoutedMLP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass

from .base import MLPBase, MLPConfig
from .token_routed import Expert
from ..registry import register_mlp


@dataclass
class SparseMoEConfig(MLPConfig):
    """Configuration for SparseMoE."""
    top_k: int = 2  # Number of experts per token
    load_balancing_weight: float = 0.01  # aux_loss weight
    expert_capacity: Optional[int] = None  # Max tokens per expert (None = auto)
    normalize_router: bool = True  # Normalize router logits


@register_mlp("sparse_moe")
@register_mlp("learned_moe")
@register_mlp("mixtral")
class SparseMoE(MLPBase):
    """
    Sparse Mixture of Experts with learned routing.

    Standard MoE approach used by Mixtral, GPT-4, Switch Transformer.

    Each token is routed to top-k experts based on a learned router network.
    Requires auxiliary load balancing loss to prevent expert collapse.

    Args:
        config: SparseMoEConfig with MoE parameters

    Returns:
        output: [batch, seq_len, hidden_size]
        aux_loss: Load balancing loss (add to total loss)
    """

    def __init__(self, config: SparseMoEConfig):
        super().__init__(config)

        self.num_experts = config.num_experts
        self.top_k = getattr(config, 'top_k', 2)
        self.load_balancing_weight = getattr(config, 'load_balancing_weight', 0.01)
        self.normalize_router = getattr(config, 'normalize_router', True)

        # Expert intermediate size (same total params as dense)
        self.expert_intermediate_size = self.intermediate_size // self.num_experts

        # Learned router network
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        # Expert modules
        self.experts = nn.ModuleList([
            Expert(
                self.hidden_size,
                self.expert_intermediate_size,
                config.hidden_act,
                config.bias,
            )
            for _ in range(self.num_experts)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,  # Unused, for API compatibility
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with learned routing.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            token_ids: Unused (for API compatibility with TokenRoutedMLP)

        Returns:
            output: [batch, seq_len, hidden_size]
            aux_loss: Load balancing loss
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Compute router logits
        router_logits = self.router(hidden_states)  # [batch, seq_len, num_experts]

        # Get routing weights (softmax over experts)
        if self.normalize_router:
            router_probs = F.softmax(router_logits, dim=-1)
        else:
            router_probs = router_logits

        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # Normalize top-k weights to sum to 1
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Compute output
        output = torch.zeros_like(hidden_states)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)  # [batch, seq_len]

            if not expert_mask.any():
                continue

            # Get tokens for this expert
            expert_input = hidden_states[expert_mask]  # [num_tokens, hidden_size]

            # Process through expert
            expert_output = self.experts[expert_idx](expert_input)

            # Get weights for this expert
            # Find which top-k slot contains this expert
            expert_weights = torch.zeros(expert_mask.sum(), device=hidden_states.device)
            for k in range(self.top_k):
                k_mask = top_k_indices[expert_mask, k] == expert_idx
                expert_weights[k_mask] = top_k_probs[expert_mask][k_mask, k]

            # Weighted contribution
            output[expert_mask] += expert_output * expert_weights.unsqueeze(-1)

        # Compute load balancing loss
        aux_loss = self._compute_load_balancing_loss(router_probs)

        return output, aux_loss

    def _compute_load_balancing_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute auxiliary load balancing loss.

        Encourages uniform distribution of tokens across experts.
        Without this, some experts get all tokens (expert collapse).

        Args:
            router_probs: [batch, seq_len, num_experts]

        Returns:
            aux_loss: Scalar loss value
        """
        # Average probability per expert across all tokens
        # Shape: [num_experts]
        expert_probs = router_probs.mean(dim=[0, 1])

        # Fraction of tokens routed to each expert (based on argmax)
        # Shape: [num_experts]
        expert_usage = F.one_hot(
            router_probs.argmax(dim=-1),
            num_classes=self.num_experts
        ).float().mean(dim=[0, 1])

        # Load balancing loss: product of probs and usage
        # Minimized when both are uniform (1/num_experts each)
        aux_loss = self.num_experts * (expert_probs * expert_usage).sum()

        return self.load_balancing_weight * aux_loss


@register_mlp("sparse_moe_parallel")
class SparseMoEParallel(MLPBase):
    """
    Optimized SparseMoE using batched matrix operations.

    Better GPU utilization for large batches.
    Same functionality as SparseMoE but faster.
    """

    def __init__(self, config: SparseMoEConfig):
        super().__init__(config)

        self.num_experts = config.num_experts
        self.top_k = getattr(config, 'top_k', 2)
        self.load_balancing_weight = getattr(config, 'load_balancing_weight', 0.01)

        self.expert_intermediate_size = self.intermediate_size // self.num_experts

        # Router
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        # Batched expert weights
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batched forward pass."""
        batch_size, seq_len, _ = hidden_states.shape
        num_tokens = batch_size * seq_len

        # Router
        router_logits = self.router(hidden_states)
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-k selection
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Flatten for batched processing
        flat_hidden = hidden_states.view(num_tokens, self.hidden_size)
        flat_top_k_indices = top_k_indices.view(num_tokens, self.top_k)
        flat_top_k_probs = top_k_probs.view(num_tokens, self.top_k)

        # Process all top-k experts
        output = torch.zeros(num_tokens, self.hidden_size, device=hidden_states.device)

        for k in range(self.top_k):
            expert_indices = flat_top_k_indices[:, k]  # [num_tokens]
            expert_weights = flat_top_k_probs[:, k]    # [num_tokens]

            # Gather weights for each token's expert
            gate_w = self.gate_proj[expert_indices]  # [num_tokens, H, I]
            up_w = self.up_proj[expert_indices]
            down_w = self.down_proj[expert_indices]

            # SwiGLU computation
            gate_out = torch.bmm(flat_hidden.unsqueeze(1), gate_w).squeeze(1)
            up_out = torch.bmm(flat_hidden.unsqueeze(1), up_w).squeeze(1)
            intermediate = self.act_fn(gate_out) * up_out
            expert_out = torch.bmm(intermediate.unsqueeze(1), down_w).squeeze(1)

            output += expert_out * expert_weights.unsqueeze(-1)

        # Load balancing loss
        aux_loss = self._compute_load_balancing_loss(router_probs)

        return output.view(batch_size, seq_len, self.hidden_size), aux_loss

    def _compute_load_balancing_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        """Compute auxiliary load balancing loss."""
        expert_probs = router_probs.mean(dim=[0, 1])
        expert_usage = F.one_hot(
            router_probs.argmax(dim=-1),
            num_classes=self.num_experts
        ).float().mean(dim=[0, 1])
        aux_loss = self.num_experts * (expert_probs * expert_usage).sum()
        return self.load_balancing_weight * aux_loss
