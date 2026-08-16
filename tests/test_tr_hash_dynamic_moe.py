"""Dynamic (runtime-configurable) TR-Hash MoE capacity.

The engine is allocated at (config.num_experts, config.expert_width) but can
be shrunk at runtime to a smaller sub-network via ``set_active_capacity``.
Routing for the reduced pool is re-derived deterministically with the same
token-ID / hash-table construction used at full capacity (see
``build_route_table``) — never a learned or contextual router. These tests
run on CPU: only the PYTORCH/CGGR-eligible reference path is exercised.
"""

from __future__ import annotations

import torch

from complexity.tr_hash import (
    AttentionBackbone,
    TRHashBackend,
    TRHashEngine,
    TRHashEngineConfig,
    TRHashPhase,
    TRHashPrecision,
    TRHashStrategy,
)


def _engine(**overrides) -> TRHashEngine:
    config = dict(
        hidden_size=16,
        vocab_size=97,
        num_experts=8,
        top_k=2,
        shared_width=8,
        expert_width=12,
        routing_strategy=TRHashStrategy.BALANCED_HASH,
        attention_backbone=AttentionBackbone.GQA,
        phase=TRHashPhase.TRAINING,
        precision=TRHashPrecision.FP32,
        backend=TRHashBackend.PYTORCH,
    )
    config.update(overrides)
    return TRHashEngine(TRHashEngineConfig(**config))


def test_default_active_capacity_equals_allocated_capacity():
    engine = _engine()
    assert engine.active_num_experts == 8
    assert engine.active_expert_width == 12
    summary = engine.capability_summary()
    assert summary["active_num_experts"] == 8
    assert summary["active_expert_width"] == 12
    assert summary["is_reduced_capacity"] is False


def test_set_active_capacity_restricts_routing_to_the_active_pool():
    engine = _engine()
    engine.set_active_capacity(num_experts=4)
    assert engine.active_num_experts == 4
    token_ids = torch.arange(97)
    routes = engine._active_route_table
    assert routes.max().item() < 4
    assert routes.min().item() >= 0
    # Full-capacity table (still intact, untouched) can route beyond 4.
    assert engine.route_table.max().item() >= 4


def test_reduced_expert_count_forward_matches_shape_and_changes_output():
    engine = _engine()
    torch.manual_seed(0)
    hidden = torch.randn(2, 5, 16)
    token_ids = torch.randint(0, 97, (2, 5))

    full_output = engine(hidden, token_ids)
    engine.set_active_capacity(num_experts=2)
    reduced_output = engine(hidden, token_ids)

    assert full_output.shape == (2, 5, 16)
    assert reduced_output.shape == (2, 5, 16)
    assert not torch.allclose(full_output, reduced_output)


def test_reduced_expert_width_forward_matches_shape_and_changes_output():
    engine = _engine()
    torch.manual_seed(1)
    hidden = torch.randn(2, 5, 16)
    token_ids = torch.randint(0, 97, (2, 5))

    full_output = engine(hidden, token_ids)
    engine.set_active_capacity(expert_width=4)
    reduced_output = engine(hidden, token_ids)

    assert reduced_output.shape == (2, 5, 16)
    assert not torch.allclose(full_output, reduced_output)


def test_reduced_width_backward_only_updates_the_active_weight_slice():
    engine = _engine()
    engine.set_active_capacity(expert_width=4)
    hidden = torch.randn(2, 5, 16, requires_grad=True)
    token_ids = torch.randint(0, 97, (2, 5))

    output = engine(hidden, token_ids)
    output.float().square().mean().backward()

    grad = engine.expert_gate.grad
    assert grad is not None
    assert torch.any(grad[:, :, :4] != 0)
    assert torch.all(grad[:, :, 4:] == 0)


def test_restoring_full_capacity_reproduces_original_output():
    engine = _engine()
    hidden = torch.randn(2, 5, 16)
    token_ids = torch.randint(0, 97, (2, 5))

    with torch.no_grad():
        full_before = engine(hidden, token_ids)
        engine.set_active_capacity(num_experts=2, expert_width=4)
        engine(hidden, token_ids)
        engine.set_active_capacity(num_experts=8, expert_width=12)
        full_after = engine(hidden, token_ids)

    assert torch.equal(full_before, full_after)


def test_set_active_capacity_rejects_unsupported_expert_counts():
    # 1 is a supported count (a degenerate single-route engine), just not
    # reachable here with top_k=2 — see
    # test_set_active_capacity_rejects_active_experts_below_top_k.
    engine = _engine()
    for bad in (3, 5, 32):
        try:
            engine.set_active_capacity(num_experts=bad)
        except ValueError as exc:
            assert "must be one of" in str(exc)
        else:
            raise AssertionError(f"expected ValueError for num_experts={bad}")


def test_set_active_capacity_rejects_capacity_above_allocation():
    engine = _engine(num_experts=4)
    try:
        engine.set_active_capacity(num_experts=8)
    except ValueError as exc:
        assert "cannot exceed the allocated capacity" in str(exc)
    else:
        raise AssertionError("expected ValueError")

    try:
        engine.set_active_capacity(expert_width=999)
    except ValueError as exc:
        assert "must be in (0," in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_set_active_capacity_rejects_active_experts_below_top_k():
    engine = _engine(top_k=4)
    try:
        engine.set_active_capacity(num_experts=2)
    except ValueError as exc:
        assert "top_k cannot exceed num_experts" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_set_active_capacity_invalidates_captured_cuda_graphs():
    engine = _engine()
    engine._graph_pool["sentinel"] = object()
    engine.set_active_capacity(num_experts=4)
    assert engine._graph_pool == {}


def test_mlp_adapter_exposes_dynamic_capacity_controls():
    from complexity.core.mlp.base import MLPConfig
    from complexity.core.mlp.tr_hash_engine import TRHashEngineMLP

    mlp = TRHashEngineMLP(
        MLPConfig(
            hidden_size=16,
            intermediate_size=96,
            num_experts=8,
            vocab_size=97,
            routing_strategy="token_id_balanced_hash",
            top_k=2,
            shared_expert=False,
        )
    )
    assert "active_num_experts" in mlp.training_control_capabilities()
    assert "active_expert_width" in mlp.training_control_capabilities()
    telemetry = mlp.training_telemetry()
    assert telemetry["active_num_experts"] == 8.0
    assert telemetry["active_expert_width"] == 12.0

    mlp.set_active_experts(4)
    mlp.set_active_expert_width(6)
    telemetry = mlp.training_telemetry()
    assert telemetry["active_num_experts"] == 4.0
    assert telemetry["active_expert_width"] == 6.0
    summary = mlp.capability_summary()
    assert summary["active_num_experts"] == 4
    assert summary["active_expert_width"] == 6
    assert summary["is_reduced_capacity"] is True


def test_model_config_can_declare_initial_reduced_capacity():
    from complexity.config import ModelConfig
    from complexity.models import ComplexityModel

    config = ModelConfig(
        vocab_size=97,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=96,
        mlp_type="tr_hash_engine",
        num_experts=8,
        top_k=2,
        routing_strategy="token_id_balanced_hash",
        active_num_experts=4,
        active_expert_width=6,
    )
    model = ComplexityModel(config)
    mlp = model.layers[0].mlp
    assert mlp.engine.active_num_experts == 4
    assert mlp.engine.active_expert_width == 6

    token_ids = torch.randint(0, 97, (2, 5))
    output = model(token_ids)["logits"]
    assert output.shape == (2, 5, 97)


def test_mlp_config_rejects_active_num_experts_above_num_experts():
    from complexity.core.mlp.base import MLPConfig
    import pytest

    with pytest.raises(ValueError, match="active_num_experts cannot exceed"):
        MLPConfig(
            hidden_size=16,
            intermediate_size=96,
            num_experts=4,
            vocab_size=97,
            active_num_experts=8,
        )


def test_mlp_config_rejects_active_expert_width_above_derived_max():
    from complexity.core.mlp.base import MLPConfig
    import pytest

    with pytest.raises(ValueError, match="active_expert_width cannot exceed"):
        MLPConfig(
            hidden_size=16,
            intermediate_size=96,
            num_experts=8,
            vocab_size=97,
            active_expert_width=999,
        )
