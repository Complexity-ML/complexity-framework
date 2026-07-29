"""Baseline-preserving token-routed multi-head attention.

TR-MHA v2 keeps the complete shared MHA path and adds a small token-routed
Q/V residual.  Compared with the first prototype:

* the routed branch is exactly zero at initialization;
* Q and V share one expert down-projection;
* only the two fixed token-ID candidates are contextually reweighted;
* dispatch computes only selected expert/token pairs.

The shared MHA path therefore remains a valid starting point while the routed
branch learns whether token identity provides useful conditional capacity.
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


@register_attention("tr_mha_v2")
@register_attention("token_routed_mha_v2")
class TokenRoutedMultiHeadAttentionV2(MultiHeadAttention):
    """Full MHA plus a neutral, selected-only token-routed Q/V residual."""

    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        self.num_route_experts = int(config.tr_mha_num_experts)
        self.adapter_rank = int(config.tr_mha_adapter_rank)
        self.route_top_k = int(config.tr_mha_top_k)
        self.layer_idx = int(config.layer_idx)
        self.route_targets = str(config.tr_mha_targets)
        self._target_count = len(self.route_targets)

        self.qv_adapter_down = nn.Parameter(
            torch.empty(
                self.num_route_experts,
                self.hidden_size,
                self.adapter_rank,
            )
        )
        self.qv_adapter_up = nn.Parameter(
            torch.zeros(
                self.num_route_experts,
                self.adapter_rank,
                self._target_count * self.hidden_size,
            )
        )
        nn.init.normal_(self.qv_adapter_down, mean=0.0, std=0.02)

        # The zero up-projection makes the routed branch exactly neutral even
        # with a non-zero gate, while still giving qv_adapter_up a gradient on
        # the first optimization step.
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

    def _fixed_candidates(
        self,
        token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        primary = (token_ids + self.layer_idx).remainder(
            self.num_route_experts
        )
        secondary = (primary + 1).remainder(self.num_route_experts)
        return primary, secondary

    def _id_prior_logits(
        self,
        token_ids: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        primary, secondary = self._fixed_candidates(token_ids)
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
        """Return telemetry probabilities and the verified fixed top-2 route."""

        if token_ids.shape != hidden_states.shape[:-1]:
            raise ValueError(
                "token_ids must match hidden_states batch/sequence shape; "
                f"got {tuple(token_ids.shape)} and "
                f"{tuple(hidden_states.shape[:-1])}"
            )
        if self.route_top_k != 2:
            raise ValueError("TR-MHA v2 currently requires top_k=2")

        prior_logits = self._id_prior_logits(
            token_ids, dtype=hidden_states.dtype
        )
        context_logits = F.linear(
            hidden_states, self.context_router_weight
        ) / self.verifier_temperature
        verifier_gate = torch.sigmoid(
            F.linear(hidden_states, self.verifier_gate_weight)
            + self.verifier_gate_bias
        )
        joint_logits = prior_logits + verifier_gate * context_logits
        probabilities = F.softmax(joint_logits.float(), dim=-1).to(
            hidden_states.dtype
        )

        primary, secondary = self._fixed_candidates(token_ids)
        indices = torch.stack((primary, secondary), dim=-1)
        candidate_logits = torch.gather(joint_logits, -1, indices)
        weights = F.softmax(candidate_logits.float(), dim=-1).to(
            hidden_states.dtype
        )
        return probabilities, indices, weights, verifier_gate

    def _selected_delta(
        self,
        hidden_states: torch.Tensor,
        top_indices: torch.Tensor,
        top_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply only selected expert/token pairs."""

        flat_hidden = hidden_states.reshape(-1, self.hidden_size)
        flat_indices = top_indices.reshape(-1, self.route_top_k)
        flat_weights = top_weights.reshape(-1, self.route_top_k)
        routed = flat_hidden.new_zeros(
            flat_hidden.shape[0], self._target_count * self.hidden_size
        )

        for expert_idx in range(self.num_route_experts):
            locations = (flat_indices == expert_idx).nonzero(as_tuple=False)
            if locations.numel() == 0:
                continue
            token_rows = locations[:, 0]
            route_slots = locations[:, 1]
            selected_hidden = flat_hidden.index_select(0, token_rows)
            low_rank = selected_hidden @ self.qv_adapter_down[expert_idx]
            qv_delta = low_rank @ self.qv_adapter_up[expert_idx]
            route_weight = flat_weights[token_rows, route_slots].unsqueeze(-1)
            routed = routed.index_add(
                0, token_rows, qv_delta * route_weight
            )

        return routed.reshape(
            *hidden_states.shape[:-1],
            self._target_count * self.hidden_size,
        )

    def _project_kqv(
        self,
        hidden_states: torch.Tensor,
        *,
        mu_prev: Optional[torch.Tensor] = None,
        token_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if token_ids is None:
            raise ValueError("TR-MHA v2 requires token_ids")
        k, q, v = super()._project_kqv(
            hidden_states,
            mu_prev=mu_prev,
            token_ids=token_ids,
        )
        probabilities, indices, weights, verifier_gate = (
            self.routing_distribution(hidden_states, token_ids)
        )
        routed_delta = self._selected_delta(
            hidden_states, indices, weights
        )
        deltas = routed_delta.chunk(self._target_count, dim=-1)
        delta_index = 0
        if "q" in self.route_targets:
            q = q + self.adapter_output_gate * deltas[delta_index]
            delta_index += 1
        if "v" in self.route_targets:
            v = v + self.adapter_output_gate * deltas[delta_index]

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
        return frozenset(
            {
                "tr_mha_routing",
                "tr_mha_v2_routing",
                f"tr_mha_targets_{self.route_targets}",
            }
        )

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
