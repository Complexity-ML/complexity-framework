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


@pytest.mark.parametrize(
    "module_name,retired_names",
    [
        (
            "complexity_cuda",
            (
                "TokenRoutedMLPTriton",
                "PersistentTokenRoutedMLP",
                "Int8TokenRoutedMLP",
                "RoboticsTokenRoutedLayer",
            ),
        ),
        (
            "complexity_cuda.triton_token_routed",
            ("TokenRoutedMLPTriton", "RoboticsTokenRoutedLayer", "fused_token_route_residual"),
        ),
        ("complexity_cuda.persistent_cggr", ("PersistentTokenRoutedMLP",)),
        ("complexity_cuda.quantization", ("Int8TokenRoutedMLP",)),
    ],
)
def test_cuda_modules_do_not_offer_an_alternative_routing_model(module_name, retired_names):
    from importlib import import_module

    module = import_module(module_name)
    for name in retired_names:
        assert not hasattr(module, name)


@pytest.mark.parametrize("module_name", ["optimized_layer", "token_routed_extras"])
def test_removed_cuda_model_modules_cannot_be_imported(module_name):
    from importlib import import_module

    with pytest.raises(ModuleNotFoundError, match=module_name):
        import_module("complexity_cuda." + module_name)
