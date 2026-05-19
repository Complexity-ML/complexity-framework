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
        "use_custom_kernels": "auto",
        "moe_telemetry": False,
    }

    expected = {"50m": 51.9, "100m": 99.7, "300m": 300.8, "1b": 1030.8, "8b": 8201.5}
    for name, profile in PROFILES.items():
        args = SimpleNamespace(**common, **profile)
        with torch.device("meta"):
            model = ComplexityModel(make_config(args))
        assert model.num_parameters() / 1e6 == pytest.approx(expected[name], abs=0.1)


def test_random_dataset_infers_vocab_from_tokenizer(monkeypatch):
    from complexity.training import o200k_pretrain

    class FakeTokenizer:
        vocab_size = 200019

    monkeypatch.setattr(o200k_pretrain.Tokenizer, "load", lambda path: FakeTokenizer())
    args = SimpleNamespace(vocab_size=None, dataset="random", tokenizer="./tokenizer-o200k")

    assert o200k_pretrain.infer_vocab_size(args) == 200019


def test_vocab_size_override_wins(monkeypatch):
    from complexity.training import o200k_pretrain

    def fail_if_called(path):
        raise AssertionError("Tokenizer.load should not be called when --vocab-size is set")

    monkeypatch.setattr(o200k_pretrain.Tokenizer, "load", fail_if_called)
    args = SimpleNamespace(vocab_size=32000, dataset="random", tokenizer="./tokenizer-o200k")

    assert o200k_pretrain.infer_vocab_size(args) == 32000


def test_token_routed_topk_reuses_sort_without_changing_output():
    from complexity.core.mlp.base import MLPConfig
    from complexity.core.mlp.token_routed import TokenRoutedMLP

    torch.manual_seed(0)
    cfg = MLPConfig(
        hidden_size=16,
        intermediate_size=32,
        num_experts=4,
        vocab_size=64,
        shared_expert=False,
        top_k=2,
        top_k_primary_weight=0.5,
    )
    mlp = TokenRoutedMLP(cfg)
    hidden = torch.randn(2, 5, 16)
    token_ids = torch.randint(0, 64, (2, 5))

    out_fast = mlp(hidden, token_ids=token_ids)

    flat_x = hidden.reshape(-1, hidden.size(-1))
    expert_ids = mlp.token_to_expert[token_ids.clamp(0, mlp.vocab_size - 1)].reshape(-1)
    gate_w = mlp.gate_proj_w
    up_w = mlp.up_proj_w
    down_w = mlp.down_proj_w
    out_ref = 0.5 * mlp._dispatch_once(flat_x, expert_ids, gate_w, up_w, down_w, False, 16)
    out_ref = out_ref + 0.5 * mlp._dispatch_once(
        flat_x,
        (expert_ids + 1) % mlp.num_experts,
        gate_w,
        up_w,
        down_w,
        False,
        16,
    )

    assert torch.allclose(out_fast.reshape_as(out_ref), out_ref, atol=1e-6)


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
        collect_moe_telemetry=True,
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


def test_zipf_token_class_routing_balances_each_class():
    from complexity.core.mlp import MLPConfig, TokenRoutedMLP

    token_classes = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    config = MLPConfig(
        hidden_size=8,
        intermediate_size=16,
        num_experts=2,
        vocab_size=8,
        routing_strategy="zipf_token_class",
        token_classes=token_classes,
        shared_expert=False,
    )

    mlp = TokenRoutedMLP(config)
    mapping = mlp.token_to_expert.cpu()
    for class_id in [0, 1]:
        assigned = mapping[token_classes == class_id]
        counts = torch.bincount(assigned, minlength=2)
        assert counts.tolist() == [2, 2]


def test_muon_tr_optimizer_builds_for_o200k_runner():
    from types import SimpleNamespace

    from complexity.models import ComplexityModel
    from complexity.training.o200k_pretrain import build_optimizer, make_config

    args = SimpleNamespace(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=32,
        shared_intermediate_size=64,
        vocab_size=128,
        use_mu_guidance=False,
        learn_shared_routed_gates=True,
        shared_gate_init=1.0,
        routed_gate_init=0.1,
        top_k=2,
        top_k_primary_weight=0.5,
        static_expert_capacity=False,
        routing_strategy="zipf",
        mu_clamp=False,
        mu_norm=False,
        mu_alpha_init=1.0,
        mu_init_value=0.0,
        mu_context_min=-2.0,
        mu_context_max=2.0,
        optimizer="muon_tr",
        lr=3e-4,
        weight_decay=0.1,
        muon_lr=0.01,
        muon_scope="expert",
        expert_lr_scale=1.5,
        shared_lr_scale=1.0,
        expert_weight_decay=0.005,
        shared_weight_decay=0.01,
        muon_ns_steps=5,
        muon_adaptive_ns=False,
        muon_max_lr_ratio=2.0,
        muon_lr_warmup_steps=50,
        muon_skip_ns_warmup_steps=0,
        muon_token_count_scaling=False,
        muon_max_update_rms=1.0,
    )
    model = ComplexityModel(make_config(args))

    optimizer, stats = build_optimizer(args, model)

    assert hasattr(optimizer, "update_token_counts")
    assert stats["muon_expert_params"] > 0
    assert stats["muon_shared_params"] == 0
    assert stats["adamw_params"] > 0


def test_batch_expert_counts_counts_current_batch():
    from complexity.config import ModelConfig
    from complexity.models import ComplexityModel
    from complexity.training.o200k_pretrain import batch_expert_counts

    config = ModelConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        intermediate_size=16,
        vocab_size=16,
        mlp_type="token_routed",
        num_experts=4,
        shared_expert=False,
    )
    model = ComplexityModel(config)
    input_ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])

    counts = batch_expert_counts(model, input_ids, num_experts=4, distributed=False)

    assert counts.sum().item() == input_ids.numel()
    assert counts.shape == (4,)


def test_plan_run_math():
    from complexity.training.plan_run import parse_tokens

    assert parse_tokens("30B") == 30_000_000_000
    assert parse_tokens("1.5M") == 1_500_000
