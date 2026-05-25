"""Regression tests for the o200k pretraining runner utilities."""

from __future__ import annotations

import math
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


def test_chunked_hidden_loss_can_skip_metric_sync():
    from complexity.core.losses import causal_lm_loss_from_hidden

    torch.manual_seed(0)
    hidden = torch.randn(2, 5, 7, requires_grad=True)
    weight = torch.randn(13, 7, requires_grad=True)
    labels = torch.randint(0, 13, (2, 5))

    loss, metrics = causal_lm_loss_from_hidden(
        hidden,
        weight,
        labels,
        chunk_tokens=3,
        checkpoint_chunks=False,
        sync_metrics=False,
    )

    assert torch.isfinite(loss)
    assert math.isnan(metrics.ce)


def test_full_hidden_loss_can_skip_metric_sync():
    from complexity.core.losses import causal_lm_loss_from_hidden

    torch.manual_seed(0)
    hidden = torch.randn(2, 5, 7, requires_grad=True)
    weight = torch.randn(13, 7, requires_grad=True)
    labels = torch.randint(0, 13, (2, 5))

    loss, metrics = causal_lm_loss_from_hidden(
        hidden,
        weight,
        labels,
        chunk_tokens=0,
        sync_metrics=False,
    )

    assert torch.isfinite(loss)
    assert math.isnan(metrics.ce)


def test_reduce_average_tensor_defers_to_single_item_sync():
    from complexity.training.o200k.runtime import reduce_average_tensor

    value = torch.tensor(3.5)

    assert reduce_average_tensor(value, distributed=False) == pytest.approx(3.5)


def test_topk_primary_weight_schedule_ramps_toward_final():
    from complexity.training.o200k.runtime import scheduled_topk_primary_weight

    assert scheduled_topk_primary_weight(0, 100, 0.5, 0.9, 0.5) == pytest.approx(0.5)
    assert scheduled_topk_primary_weight(50, 100, 0.5, 0.9, 0.5) == pytest.approx(0.9)
    assert scheduled_topk_primary_weight(100, 100, 0.5, 0.9, 0.5) == pytest.approx(0.9)

    mid = scheduled_topk_primary_weight(25, 100, 0.5, 0.9, 0.5)
    assert 0.5 < mid < 0.9


def test_apply_topk_primary_weight_updates_token_routed_layers():
    from complexity.core.mlp.base import MLPConfig
    from complexity.core.mlp.token_routed import TokenRoutedMLP
    from complexity.training.o200k.runtime import apply_topk_primary_weight

    mlp = TokenRoutedMLP(
        MLPConfig(
            hidden_size=8,
            intermediate_size=16,
            num_experts=4,
            vocab_size=16,
            shared_expert=False,
            top_k=2,
            top_k_primary_weight=0.5,
        )
    )

    assert apply_topk_primary_weight(mlp, 0.85) == 1
    assert mlp._primary_weight == pytest.approx(0.85)


def test_liger_fused_ce_availability_is_exposed(monkeypatch):
    from complexity.core.losses import fused_ce

    monkeypatch.setattr(fused_ce, "_liger_available", lambda: True)

    assert fused_ce.has_liger_fused_linear_ce() is True


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


def test_o200k_parser_enables_grad_checkpointing_by_default():
    from complexity.training.o200k_pretrain import build_parser

    args = build_parser().parse_args([])

    assert args.grad_ckpt is True


def test_o200k_parser_can_disable_grad_checkpointing():
    from complexity.training.o200k_pretrain import build_parser

    args = build_parser().parse_args(["--no-grad-ckpt"])

    assert args.grad_ckpt is False


def test_o200k_parser_disables_grad_clipping_by_default():
    from complexity.training.o200k_pretrain import build_parser

    args = build_parser().parse_args([])

    assert args.max_grad_norm == 0.0


def test_o200k_parser_uses_auto_loss_backend_by_default():
    from complexity.training.o200k_pretrain import build_parser

    args = build_parser().parse_args([])

    assert args.loss_backend == "auto"


def test_o200k_parser_supports_token_shards():
    from complexity.training.o200k_pretrain import build_parser

    args = build_parser().parse_args(["--dataset", "tokens", "--tokens-path", "data/tokens"])

    assert args.dataset == "tokens"
    assert args.tokens_path == "data/tokens"


def test_token_shard_dataset_and_frequencies(tmp_path):
    from complexity.data.token_shards import (
        TokenShardDataset,
        load_token_shard,
        token_shard_frequencies,
        write_token_shard,
    )

    write_token_shard(tmp_path, range(100), vocab_size=128, tokenizer="dummy")
    tokens, metadata = load_token_shard(tmp_path)

    assert tokens.shape == (100,)
    assert metadata["num_tokens"] == 100
    assert metadata["dtype"] == "<u2"
    assert len(metadata["sha256"]) == 64

    freqs = token_shard_frequencies(tmp_path, vocab_size=128)
    assert freqs.sum().item() == 100
    assert freqs[42].item() == 1

    ds = TokenShardDataset(tmp_path, seq_len=8, seed=0, eval_ratio=0.2)
    batch = next(iter(ds))
    assert batch["input_ids"].shape == (8,)
    assert batch["labels"].shape == (8,)


def test_token_routed_topk_uses_precomputed_zipf_routes():
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
    route_ids = mlp.topk_token_to_expert[:, token_ids.clamp(0, mlp.vocab_size - 1)]
    gate_w = mlp.gate_proj_w
    up_w = mlp.up_proj_w
    down_w = mlp.down_proj_w
    out_ref = 0.5 * mlp._dispatch_once(
        flat_x, route_ids[0].reshape(-1), gate_w, up_w, down_w, False, 16
    )
    out_ref = out_ref + 0.5 * mlp._dispatch_once(
        flat_x,
        route_ids[1].reshape(-1),
        gate_w,
        up_w,
        down_w,
        False,
        16,
    )

    assert torch.allclose(out_fast.reshape_as(out_ref), out_ref, atol=1e-6)


def test_token_routed_topk_aux_routes_are_balanced_and_distinct():
    from complexity.core.mlp.base import MLPConfig
    from complexity.core.mlp.token_routed import TokenRoutedMLP

    cfg = MLPConfig(
        hidden_size=8,
        intermediate_size=16,
        num_experts=4,
        vocab_size=16,
        shared_expert=False,
        top_k=2,
        token_frequencies=torch.ones(16),
    )
    mlp = TokenRoutedMLP(cfg)

    routes = mlp.topk_token_to_expert.cpu()
    assert torch.all(routes[0] != routes[1])
    for route_idx in range(2):
        counts = torch.bincount(routes[route_idx], minlength=4)
        assert counts.tolist() == [4, 4, 4, 4]


def test_token_routed_masked_dispatch_matches_token_reference():
    from complexity.core.mlp.base import MLPConfig
    from complexity.core.mlp.token_routed import TokenRoutedMLP, sort_tokens_by_expert

    torch.manual_seed(0)
    cfg = MLPConfig(
        hidden_size=8,
        intermediate_size=16,
        num_experts=4,
        vocab_size=64,
        shared_expert=False,
        top_k=1,
    )
    mlp = TokenRoutedMLP(cfg)
    flat_x = torch.randn(11, 8)
    expert_ids = torch.tensor([0, 3, 1, 0, 2, 3, 1, 2, 0, 3, 2])
    sorted_x, sorted_idx, expert_offsets, expert_counts = sort_tokens_by_expert(
        flat_x, expert_ids, mlp.num_experts
    )

    out = mlp._dispatch_sorted(
        flat_x,
        sorted_x,
        sorted_idx,
        expert_offsets,
        expert_counts,
        mlp.gate_proj_w,
        mlp.up_proj_w,
        mlp.down_proj_w,
        use_cggr=False,
        H=8,
    )

    ref = torch.empty_like(flat_x)
    for i, expert in enumerate(expert_ids.tolist()):
        x = flat_x[i]
        gate = x @ mlp.gate_proj_w[expert]
        up = x @ mlp.up_proj_w[expert]
        ref[i] = (torch.nn.functional.silu(gate) * up) @ mlp.down_proj_w[expert]

    assert torch.allclose(out, ref, atol=1e-6)


def test_cggr_dispatch_auto_selects_triton_when_available(monkeypatch):
    import complexity.core.mlp.token_routed as token_routed

    monkeypatch.setattr(token_routed, "supports_custom_triton", lambda policy: True)

    use_cggr, reasons = token_routed.cggr_dispatch_decision(
        cggr_policy="auto",
        kernel_policy="auto",
        is_cuda=True,
        has_cggr=True,
        has_autograd=True,
    )

    assert use_cggr is True
    assert reasons == []


def test_cggr_dispatch_falls_back_when_auto_is_not_supported(monkeypatch):
    import complexity.core.mlp.token_routed as token_routed

    monkeypatch.setattr(token_routed, "supports_custom_triton", lambda policy: False)

    use_cggr, reasons = token_routed.cggr_dispatch_decision(
        cggr_policy="auto",
        kernel_policy="auto",
        is_cuda=True,
        has_cggr=True,
        has_autograd=True,
    )

    assert use_cggr is False
    assert "supports_custom_triton(policy='auto')=False" in reasons


def test_o200k_config_defaults_cggr_to_auto():
    from complexity.training.o200k_pretrain import make_config

    args = SimpleNamespace(
        hidden_size=32,
        num_hidden_layers=1,
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
        shared_expert_chunk_tokens=0,
        static_expert_capacity=False,
        routing_strategy="zipf",
        mu_clamp=False,
        mu_norm=False,
        mu_alpha_init=1.0,
        mu_init_value=0.0,
        mu_context_min=-2.0,
        mu_context_max=2.0,
        use_custom_kernels="auto",
        moe_telemetry=False,
    )

    assert make_config(args).use_cggr == "auto"


def test_shared_expert_chunking_matches_dense_path():
    from complexity.core.mlp.base import MLPConfig
    from complexity.core.mlp.token_routed import TokenRoutedMLP

    torch.manual_seed(0)
    cfg = MLPConfig(
        hidden_size=8,
        intermediate_size=16,
        num_experts=2,
        vocab_size=32,
        shared_expert=True,
        shared_intermediate_size=24,
        shared_expert_chunk_tokens=0,
    )
    dense = TokenRoutedMLP(cfg)

    chunked_cfg = MLPConfig(
        hidden_size=8,
        intermediate_size=16,
        num_experts=2,
        vocab_size=32,
        shared_expert=True,
        shared_intermediate_size=24,
        shared_expert_chunk_tokens=5,
    )
    chunked = TokenRoutedMLP(chunked_cfg)
    chunked.load_state_dict(dense.state_dict(), strict=False)

    hidden = torch.randn(3, 4, 8, requires_grad=True)
    token_ids = torch.randint(0, 32, (3, 4))

    out_dense = dense(hidden, token_ids=token_ids)
    out_chunked = chunked(hidden, token_ids=token_ids)

    assert torch.allclose(out_chunked, out_dense, atol=1e-6)


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


def test_adamw_optimizer_uses_foreach_for_o200k_runner():
    from types import SimpleNamespace

    from complexity.models import ComplexityModel
    from complexity.training.o200k_pretrain import build_optimizer, make_config

    args = SimpleNamespace(
        hidden_size=32,
        num_hidden_layers=1,
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
        optimizer="adamw",
        lr=3e-4,
        weight_decay=0.1,
        shared_expert_chunk_tokens=0,
        use_custom_kernels="auto",
        moe_telemetry=False,
    )
    model = ComplexityModel(make_config(args))

    optimizer, stats = build_optimizer(args, model)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert stats["adamw_params"] > 0
    assert stats["adamw_impl"] in {"foreach", "default"}


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


def test_batch_expert_counts_counts_all_topk_routes():
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
        top_k=2,
    )
    model = ComplexityModel(config)
    input_ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])

    counts = batch_expert_counts(model, input_ids, num_experts=4, distributed=False)

    assert counts.sum().item() == input_ids.numel() * 2
    assert counts.shape == (4,)


def test_plan_run_math():
    from complexity.training.plan_run import parse_tokens

    assert parse_tokens("30B") == 30_000_000_000
    assert parse_tokens("1.5M") == 1_500_000
