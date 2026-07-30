"""Regression tests for optional token-routed telemetry."""

from __future__ import annotations

import math

import pytest
import torch

from complexity.core.mlp import MLPConfig, TokenRoutedMLP
from complexity.training.moe_telemetry import (
    global_expert_shares,
    global_tr_diagnostics,
)


def _make_mlp(*, collect_telemetry: bool) -> TokenRoutedMLP:
    return TokenRoutedMLP(
        MLPConfig(
            hidden_size=16,
            intermediate_size=32,
            num_experts=4,
            vocab_size=64,
            shared_expert=True,
            shared_intermediate_size=32,
            top_k=2,
            top_k_primary_weight=0.5,
            collect_moe_telemetry=collect_telemetry,
        )
    )


def test_disabled_utilization_is_unavailable_not_all_dead() -> None:
    mlp = _make_mlp(collect_telemetry=False)
    hidden = torch.randn(2, 4, 16)
    token_ids = torch.arange(8).reshape(2, 4)

    mlp(hidden, token_ids=token_ids)
    shares, dead = global_expert_shares(mlp, num_experts=4)

    assert all(math.isnan(share) for share in shares)
    assert dead is None


def test_enabled_utilization_reports_balanced_fixed_routes() -> None:
    mlp = _make_mlp(collect_telemetry=True)
    hidden = torch.randn(2, 4, 16)
    token_ids = torch.arange(8).reshape(2, 4)

    mlp(hidden, token_ids=token_ids)
    shares, dead = global_expert_shares(mlp, num_experts=4)

    assert shares == pytest.approx([0.25, 0.25, 0.25, 0.25])
    assert dead == 0


def test_disabled_activation_telemetry_reports_nan_not_zero() -> None:
    mlp = _make_mlp(collect_telemetry=False)
    hidden = torch.randn(2, 4, 16)
    token_ids = torch.arange(8).reshape(2, 4)

    loss = mlp(hidden, token_ids=token_ids).pow(2).mean()
    loss.backward()
    diagnostics = global_tr_diagnostics(mlp, num_experts=4)

    assert math.isnan(diagnostics["shared_rms"])
    assert math.isnan(diagnostics["routed_rms"])
    assert diagnostics["shared_grad_norm"] > 0
    assert diagnostics["routed_grad_norm"] > 0
