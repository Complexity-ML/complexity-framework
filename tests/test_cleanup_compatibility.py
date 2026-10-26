"""Canonical runtime contracts and explicit rejection of token-only checkpoints."""

import json

import pytest
import torch


def test_token_only_checkpoint_is_rejected_before_model_construction(tmp_path, monkeypatch):
    from complexity.models import ComplexityModel

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "mlp_type": "token_routed",
                "routing_strategy": "token_id_balanced_hash",
                "hidden_size": 8,
                "vocab_size": 17,
            }
        )
    )

    def reject_construction(*args, **kwargs):
        pytest.fail("A token-only checkpoint must be rejected before creating a model")

    monkeypatch.setattr(ComplexityModel, "__init__", reject_construction)
    with pytest.raises(ValueError, match="explicitly declare canonical token_id_composed_modulo"):
        ComplexityModel.from_pretrained(tmp_path)


def test_canonical_tr_hash_engine_weight_shapes_and_backend():
    from complexity.core.mlp.base import MLPConfig
    from complexity.core.mlp.tr_hash_engine import TRHashEngineMLP
    from complexity.tr_hash import TRHashBackend

    mlp = TRHashEngineMLP(
        MLPConfig(
            hidden_size=8,
            intermediate_size=8,
            vocab_size=17,
            num_experts=4,
            top_k=2,
            top_k_primary_weight=0.5,
            use_custom_kernels=False,
        )
    )
    assert mlp.engine.config.backend == TRHashBackend.PYTORCH
    assert mlp.engine.config.routing_strategy.value == "token_id_composed_modulo"
    assert tuple(mlp.engine.expert_gate.shape) == (4, 8, 2)
    assert tuple(mlp.engine.expert_up.shape) == (4, 8, 2)
    assert tuple(mlp.engine.expert_down.shape) == (4, 2, 8)
    assert tuple(mlp.engine.shared_gate.weight.shape) == (16, 8)
    assert tuple(mlp.engine.shared_up.weight.shape) == (16, 8)
    assert tuple(mlp.engine.shared_down.weight.shape) == (8, 16)
    assert mlp.engine.route_table.dtype == torch.int64
    assert tuple(mlp.engine.route_table.shape) == (2, 17)


def test_cuda_extras_remain_available_from_historical_module():
    from complexity_cuda.token_routed_extras import (
        RoboticsTokenRoutedLayer as MovedRoboticsLayer,
        benchmark_token_routed_mlp as moved_benchmark,
    )
    from complexity_cuda.triton_token_routed import (
        RoboticsTokenRoutedLayer,
        benchmark_token_routed_mlp,
    )

    assert RoboticsTokenRoutedLayer is MovedRoboticsLayer
    assert benchmark_token_routed_mlp is moved_benchmark
