"""Token-routed multi-head attention with context-verified Q/V adapters.

TR-MHA keeps a complete, shared MHA path: every query head has its own key and
value head.  Token identity supplies a fixed layer-specific prior over narrow
residual adapters, while the current hidden state provides a learned
verification term.  The selected top-2 adapters modify Q and V; K remains on
the shared path in this first, deliberately bounded experiment.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_attention
from .base import AttentionConfig
from .gqa import MultiHeadAttention


@register_attention("tr_mha")
@register_attention("token_routed_mha")
class TokenRoutedMultiHeadAttention(MultiHeadAttention):
    """Full MHA plus token-routed low-rank residual adapters for Q and V."""

    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        self.num_route_experts = int(config.tr_mha_num_experts)
        self.adapter_rank = int(config.tr_mha_adapter_rank)
        self.route_top_k = int(config.tr_mha_top_k)
        self.layer_idx = int(config.layer_idx)

        # Raw parameters keep their explicit initialization intact when the
        # model applies its generic Linear/Embedding initialization pass.
        shape_down = (
            self.num_route_experts,
            self.hidden_size,
            self.adapter_rank,
        )
        shape_up = (
            self.num_route_experts,
            self.adapter_rank,
            self.hidden_size,
        )
        self.q_adapter_down = nn.Parameter(torch.empty(shape_down))
        self.q_adapter_up = nn.Parameter(torch.empty(shape_up))
        self.v_adapter_down = nn.Parameter(torch.empty(shape_down))
        self.v_adapter_up = nn.Parameter(torch.empty(shape_up))
        for parameter in (self.q_adapter_down, self.v_adapter_down):
            nn.init.normal_(parameter, mean=0.0, std=0.02)
        for parameter in (self.q_adapter_up, self.v_adapter_up):
            nn.init.normal_(parameter, mean=0.0, std=0.02)

        self.adapter_output_gate = nn.Parameter(
            torch.tensor(float(config.tr_mha_adapter_gate_init))
        )
        self.context_router_weight = nn.Parameter(
            torch.empty(self.num_route_experts, self.hidden_size)
        )
        nn.init.normal_(self.context_router_weight, mean=0.0, std=0.002)
        self.verifier_gate_weight = nn.Parameter(
            torch.zeros(1, self.hidden_size)
        )
        gate_init = min(
            1.0 - 1e-6,
            max(1e-6, float(config.tr_mha_verifier_gate_init)),
        )
        self.verifier_gate_bias = nn.Parameter(
            torch.tensor(math.log(gate_init / (1.0 - gate_init)))
        )

        self.id_primary_logit = float(config.tr_mha_id_primary_logit)
        self.id_secondary_logit = float(config.tr_mha_id_secondary_logit)
        self.id_other_logit = float(config.tr_mha_id_other_logit)
        self.verifier_temperature = float(
            config.tr_mha_verifier_temperature
        )

        self._last_route_weights: Optional[torch.Tensor] = None
        self._last_verifier_gate: Optional[torch.Tensor] = None
        self._last_route_entropy: Optional[torch.Tensor] = None

    def _id_prior_logits(
        self,
        token_ids: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        primary = (token_ids + self.layer_idx).remainder(
            self.num_route_experts
        )
        secondary = (primary + 1).remainder(self.num_route_experts)
        primary_hot = F.one_hot(
            primary, num_classes=self.num_route_experts
        ).to(dtype=dtype)
        secondary_hot = F.one_hot(
            secondary, num_classes=self.num_route_experts
        ).to(dtype=dtype)
        prior = torch.full_like(primary_hot, self.id_other_logit)
        prior = prior + (
            self.id_primary_logit - self.id_other_logit
        ) * primary_hot
        prior = prior + (
            self.id_secondary_logit - self.id_other_logit
        ) * secondary_hot
        return prior

    def routing_distribution(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return dense route probabilities and the fixed-shape top-k route."""

        if token_ids.shape != hidden_states.shape[:-1]:
            raise ValueError(
                "token_ids must match hidden_states batch/sequence shape; "
                f"got {tuple(token_ids.shape)} and "
                f"{tuple(hidden_states.shape[:-1])}"
            )
        prior_logits = self._id_prior_logits(
            token_ids, dtype=hidden_states.dtype
        )
        context_logits = F.linear(
            hidden_states, self.context_router_weight
        )
        context_logits = context_logits / self.verifier_temperature
        verifier_gate = torch.sigmoid(
            F.linear(hidden_states, self.verifier_gate_weight)
            + self.verifier_gate_bias
        )
        joint_logits = prior_logits + verifier_gate * context_logits
        probabilities = F.softmax(joint_logits.float(), dim=-1).to(
            hidden_states.dtype
        )
        top_values, top_indices = torch.topk(
            joint_logits, k=self.route_top_k, dim=-1
        )
        top_weights = F.softmax(top_values.float(), dim=-1).to(
            hidden_states.dtype
        )
        return probabilities, top_indices, top_weights, verifier_gate

    @staticmethod
    def _adapter_delta(
        hidden_states: torch.Tensor,
        down: torch.Tensor,
        up: torch.Tensor,
        top_indices: torch.Tensor,
        top_weights: torch.Tensor,
    ) -> torch.Tensor:
        low_rank = torch.einsum("...d,edr->...er", hidden_states, down)
        all_deltas = torch.einsum("...er,erd->...ed", low_rank, up)
        selected = torch.gather(
            all_deltas,
            -2,
            top_indices[..., None].expand(
                *top_indices.shape, hidden_states.shape[-1]
            ),
        )
        return (selected * top_weights[..., None]).sum(dim=-2)

    def _project_kqv(
        self,
        hidden_states: torch.Tensor,
        *,
        mu_prev: Optional[torch.Tensor] = None,
        token_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if token_ids is None:
            raise ValueError("TR-MHA requires token_ids")
        k, q, v = super()._project_kqv(
            hidden_states,
            mu_prev=mu_prev,
            token_ids=token_ids,
        )
        probabilities, indices, weights, verifier_gate = (
            self.routing_distribution(hidden_states, token_ids)
        )
        q_delta = self._adapter_delta(
            hidden_states,
            self.q_adapter_down,
            self.q_adapter_up,
            indices,
            weights,
        )
        v_delta = self._adapter_delta(
            hidden_states,
            self.v_adapter_down,
            self.v_adapter_up,
            indices,
            weights,
        )
        q = q + self.adapter_output_gate * q_delta
        v = v + self.adapter_output_gate * v_delta

        dense_route_weights = torch.zeros_like(probabilities).scatter(
            -1, indices, weights
        )
        self._last_route_weights = dense_route_weights.detach()
        self._last_verifier_gate = verifier_gate.detach()
        self._last_route_entropy = (
            -probabilities.float()
            * probabilities.float().clamp_min(1e-9).log()
        ).sum(dim=-1).detach()
        return k, q, v

    def training_control_capabilities(self) -> frozenset[str]:
        return frozenset({"tr_mha_routing"})

    def training_telemetry(self) -> dict[str, float]:
        values = {
            "tr_mha_gate": float(
                self.adapter_output_gate.detach().float().item()
            ),
            "tr_mha_verifier_strength": float(
                torch.sigmoid(self.verifier_gate_bias.detach()).float().item()
            ),
        }
        if self._last_verifier_gate is not None:
            values["tr_mha_verifier_use"] = float(
                self._last_verifier_gate.float().mean().item()
            )
        if self._last_route_entropy is not None:
            values["tr_mha_route_entropy"] = float(
                self._last_route_entropy.float().mean().item()
            )
        if self._last_route_weights is not None:
            reduce_dims = tuple(range(self._last_route_weights.ndim - 1))
            shares = self._last_route_weights.float().mean(dim=reduce_dims)
            for expert_idx, share in enumerate(shares):
                values[f"tr_mha_route_{expert_idx}"] = float(share.item())
        return values
