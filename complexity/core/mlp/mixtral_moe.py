"""Learned contextual top-k MoE used as the Token-Routed control.

The shared SwiGLU path and routed expert tensors intentionally match
``TokenRoutedMLP``.  The only architectural addition is a small learned
``hidden_state -> expert logits`` projection.  This makes the module useful
for controlled comparisons in which routing is the independent variable.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import MLPBase, MLPConfig
from ..registry import register_mlp


@register_mlp("mixtral")
@register_mlp("learned_router")
@register_mlp("standard_moe")
class MixtralMoE(MLPBase):
    """Shared SwiGLU plus learned contextual top-k residual experts."""

    def __init__(self, config: MLPConfig):
        super().__init__(config)

        self.num_experts = int(config.num_experts)
        self.top_k = int(config.top_k)
        self.expert_intermediate_size = (
            self.intermediate_size // self.num_experts
        )
        if self.intermediate_size % self.num_experts:
            raise ValueError(
                "intermediate_size must be divisible by num_experts for "
                "the learned-router control"
            )

        # Keep names and shapes identical to TokenRoutedMLP so parameter and
        # gradient diagnostics compare the same expert tensors.
        self.gate_proj_w = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.hidden_size,
                self.expert_intermediate_size,
            )
        )
        self.up_proj_w = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.hidden_size,
                self.expert_intermediate_size,
            )
        )
        self.down_proj_w = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.expert_intermediate_size,
                self.hidden_size,
            )
        )

        for expert_idx in range(self.num_experts):
            nn.init.kaiming_uniform_(
                self.gate_proj_w[expert_idx], a=5**0.5
            )
            nn.init.kaiming_uniform_(
                self.up_proj_w[expert_idx], a=5**0.5
            )
            nn.init.kaiming_uniform_(
                self.down_proj_w[expert_idx], a=5**0.5
            )
        self.use_shared_expert = bool(
            getattr(config, "shared_expert", False)
        )
        if self.use_shared_expert:
            shared_size = (
                getattr(config, "shared_intermediate_size", None)
                or self.intermediate_size
            )
            self.shared_gate = nn.Linear(
                self.hidden_size, shared_size, bias=False
            )
            self.shared_up = nn.Linear(
                self.hidden_size, shared_size, bias=False
            )
            self.shared_down = nn.Linear(
                shared_size, self.hidden_size, bias=False
            )

        # This projection is the only parameter tensor absent from the fixed
        # Token-Routed control. Isolate its initialization RNG so constructing
        # the router does not perturb any shared/expert initialization in this
        # or later layers. The common tensors are then bit-identical between
        # fixed and learned controls under the same model seed.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(
                0x1EA2_0000 + int(getattr(config, "layer_idx", 0))
            )
            self.router = nn.Linear(
                self.hidden_size, self.num_experts, bias=False
            )

        # The current differentiable auxiliary loss is read by the trainer
        # immediately after forward. It is deliberately not checkpoint state.
        self.router_aux_loss: Optional[torch.Tensor] = None
        self.register_buffer(
            "expert_counts",
            torch.zeros(self.num_experts, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "last_shared_rms",
            torch.tensor(float("nan")),
            persistent=False,
        )
        self.register_buffer(
            "last_routed_rms",
            torch.tensor(float("nan")),
            persistent=False,
        )

    def reset_expert_counts(self) -> None:
        self.expert_counts.zero_()

    def get_expert_counts(self) -> torch.Tensor:
        return self.expert_counts

    def training_control_capabilities(self) -> frozenset[str]:
        return frozenset({"learned_router_aux"})

    def training_telemetry(self) -> dict[str, float]:
        if self.router_aux_loss is None:
            return {}
        return {
            "router_aux": float(
                self.router_aux_loss.detach().float().item()
            )
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Route each contextual hidden state to its learned top-k experts."""

        del token_ids, kwargs
        batch_size, seq_len, hidden_size = hidden_states.shape
        flat_x = hidden_states.reshape(-1, hidden_size)

        router_logits = self.router(flat_x)
        router_probs = F.softmax(router_logits.float(), dim=-1)
        route_weights, route_expert_ids = torch.topk(
            router_probs, k=self.top_k, dim=-1
        )
        route_weights = route_weights / route_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-9)
        route_weights = route_weights.to(flat_x.dtype)

        # Switch-style load balancing objective generalized to top-k. The hard
        # assignment fraction is treated as a target while router_probs keeps
        # the gradient path to the learned projection.
        assignment_fraction = F.one_hot(
            route_expert_ids, num_classes=self.num_experts
        ).float().sum(dim=1)
        assignment_fraction = (
            assignment_fraction.mean(dim=0) / float(self.top_k)
        )
        probability_fraction = router_probs.mean(dim=0)
        self.router_aux_loss = self.num_experts * torch.sum(
            assignment_fraction.detach() * probability_fraction
        )

        collect_telemetry = bool(
            getattr(self.config, "collect_moe_telemetry", False)
        )
        if collect_telemetry:
            with torch.no_grad():
                self.expert_counts += torch.bincount(
                    route_expert_ids.reshape(-1),
                    minlength=self.num_experts,
                ).to(self.expert_counts.dtype)

        # MPS/CPU-friendly masked-dense dispatch. TokenRoutedMLP uses the same
        # expert-loop fallback when custom grouped-GEMM kernels are disabled,
        # so local throughput remains a meaningful comparison.
        routed_out = torch.zeros_like(flat_x)
        for expert_idx in range(self.num_experts):
            expert_weight = (
                route_weights
                * (route_expert_ids == expert_idx).to(route_weights.dtype)
            ).sum(dim=-1)
            gate = flat_x @ self.gate_proj_w[expert_idx]
            up = flat_x @ self.up_proj_w[expert_idx]
            expert_out = (
                F.silu(gate) * up
            ) @ self.down_proj_w[expert_idx]
            routed_out = routed_out + (
                expert_weight.unsqueeze(-1) * expert_out
            )

        if self.use_shared_expert:
            shared_out = self.shared_down(
                F.silu(self.shared_gate(flat_x)) * self.shared_up(flat_x)
            )
            output = shared_out + routed_out
        else:
            shared_out = None
            output = routed_out

        if collect_telemetry:
            with torch.no_grad():
                self.last_routed_rms.copy_(
                    routed_out.float().pow(2).mean().sqrt()
                )
                if shared_out is not None:
                    self.last_shared_rms.copy_(
                        shared_out.float().pow(2).mean().sqrt()
                    )

        return output.reshape(batch_size, seq_len, hidden_size)
