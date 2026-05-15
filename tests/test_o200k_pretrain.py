"""Regression tests for the o200k pretraining runner utilities."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def test_chunked_hidden_loss_matches_full_loss():
    from complexity.core.losses import causal_lm_loss, causal_lm_loss_from_hidden

    torch.manual_seed(0)
    hidden = torch.randn(2, 5, 7, requires_grad=True)
    weight = torch.randn(13, 7, requires_grad=True)
    labels = torch.randint(0, 13, (2, 5))

    full_loss, _ = causal_lm_loss(torch.nn.functional.linear(hidden, weight), labels)
    chunked_loss, _ = causal_lm_loss_from_hidden(
        hidden,
        weight,
        labels,
        chunk_tokens=3,
        checkpoint_chunks=False,
    )

    assert torch.allclose(full_loss, chunked_loss, atol=1e-6)


def test_profile_param_counts_are_stable():
    from complexity.models import ComplexityModel
    from complexity.training.o200k_pretrain import PROFILES, make_config

    common = {
        "vocab_size": 200019,
        "use_mu_guidance": False,
        "learn_shared_routed_gates": True,
        "shared_gate_init": 1.0,
        "routed_gate_init": 0.1,
        "top_k": 2,
        "top_k_primary_weight": 0.5,
        "mu_clamp": False,
        "mu_norm": False,
        "mu_alpha_init": 1.0,
        "mu_init_value": 0.0,
        "mu_context_min": -2.0,
        "mu_context_max": 2.0,
    }

    expected = {"50m": 51.9, "100m": 99.7}
    for name, profile in PROFILES.items():
        args = SimpleNamespace(**common, **profile)
        model = ComplexityModel(make_config(args))
        assert model.num_parameters() / 1e6 == pytest.approx(expected[name], abs=0.1)


def test_latest_checkpoint_resolution(tmp_path):
    from complexity.training.o200k_pretrain import resolve_checkpoint_path

    root = tmp_path / "ckpts"
    step_1 = root / "step_000001"
    step_2 = root / "step_000002"
    step_1.mkdir(parents=True)
    step_2.mkdir()
    (step_1 / "checkpoint.pt").write_bytes(b"1")
    (step_2 / "checkpoint.pt").write_bytes(b"2")
    (root / "latest").write_text("step_000001\n")

    assert resolve_checkpoint_path(str(root / "latest")) == step_1


def test_local_checkpoint_save_latest_and_rotation(tmp_path):
    from complexity.utils.local_checkpoint import load_local_checkpoint, save_local_checkpoint

    root = tmp_path / "ckpts"
    save_local_checkpoint(root, step=1, state={"step": 1, "value": torch.tensor([1])}, total_limit=2)
    save_local_checkpoint(root, step=2, state={"step": 2, "value": torch.tensor([2])}, total_limit=2)
    latest_dir = save_local_checkpoint(root, step=3, state={"step": 3, "value": torch.tensor([3])}, total_limit=2)

    assert latest_dir == root / "step_000003"
    assert sorted(path.name for path in root.glob("step_*")) == ["step_000002", "step_000003"]
    assert (root / "latest").read_text(encoding="utf-8").strip() == "step_000003"

    resolved, state = load_local_checkpoint(root / "latest")
    assert resolved == latest_dir
    assert state["step"] == 3
    assert state["value"].item() == 3


def test_tr_diagnostics_reports_gates_rms_and_grads():
    from complexity.core.mlp import MLPConfig, TokenRoutedMLP
    from complexity.training.moe_telemetry import global_tr_diagnostics

    torch.manual_seed(0)
    config = MLPConfig(
        hidden_size=16,
        intermediate_size=32,
        num_experts=4,
        vocab_size=64,
        shared_expert=True,
        shared_intermediate_size=32,
        use_shared_routed_gates=True,
        shared_gate_init=1.0,
        routed_gate_init=0.1,
        top_k=2,
        top_k_primary_weight=0.5,
    )
    mlp = TokenRoutedMLP(config)
    hidden = torch.randn(2, 5, 16)
    token_ids = torch.randint(0, 64, (2, 5))

    loss = mlp(hidden, token_ids=token_ids).pow(2).mean()
    loss.backward()
    diagnostics = global_tr_diagnostics(mlp, num_experts=4)

    assert diagnostics["shared_gate"] == pytest.approx(1.0)
    assert diagnostics["routed_gate"] == pytest.approx(0.1)
    assert diagnostics["shared_rms"] > 0
    assert diagnostics["routed_rms"] > 0
    assert diagnostics["shared_grad_norm"] > 0
    assert diagnostics["routed_grad_norm"] > 0
    assert all(diagnostics[f"expert_{idx}_grad_norm"] > 0 for idx in range(4))


def test_plan_run_math():
    from complexity.training.plan_run import parse_tokens

    assert parse_tokens("30B") == 30_000_000_000
    assert parse_tokens("1.5M") == 1_500_000
