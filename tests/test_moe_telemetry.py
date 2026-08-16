"""Regression tests for optional MoE telemetry.

``global_expert_shares``/``global_tr_diagnostics`` are duck-typed against the
historical ``TokenRoutedMLP``'s attribute names (``expert_counts``,
``gate_proj_w``, ``last_shared_rms``, ...). Now that class is removed and
``TRHashEngineMLP`` doesn't expose equivalents yet (an accepted, tracked gap
— see complexity-framework session history), these helpers must keep
degrading gracefully (empty/None) rather than crashing, for any model.
"""

from __future__ import annotations

import torch

from complexity.core.mlp import MLPConfig, TRHashEngineMLP
from complexity.training.moe_telemetry import (
    detect_num_experts,
    global_expert_shares,
    global_tr_diagnostics,
)


def _make_mlp() -> TRHashEngineMLP:
    return TRHashEngineMLP(
        MLPConfig(
            hidden_size=16,
            intermediate_size=32,
            num_experts=4,
            vocab_size=64,
            shared_expert=True,
            shared_intermediate_size=32,
            top_k=2,
            top_k_primary_weight=0.5,
        )
    )


def test_detect_num_experts_returns_none_without_token_routed_counters() -> None:
    mlp = _make_mlp()
    assert detect_num_experts(mlp) is None


def test_global_expert_shares_is_empty_not_a_crash() -> None:
    mlp = _make_mlp()
    hidden = torch.randn(2, 4, 16)
    token_ids = torch.arange(8).reshape(2, 4)

    mlp(hidden, token_ids=token_ids)
    shares, dead = global_expert_shares(mlp, num_experts=4)

    assert shares == []
    assert dead is None


def test_global_tr_diagnostics_is_empty_not_a_crash() -> None:
    mlp = _make_mlp()
    hidden = torch.randn(2, 4, 16)
    token_ids = torch.arange(8).reshape(2, 4)

    loss = mlp(hidden, token_ids=token_ids).pow(2).mean()
    loss.backward()
    diagnostics = global_tr_diagnostics(mlp, num_experts=4)

    assert diagnostics == {}
