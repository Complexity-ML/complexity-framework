from types import SimpleNamespace

import torch

from complexity import ComplexityModel, ModelConfig
from complexity.core.mlp.tr_hash_engine import TRHashEngineMLP
from complexity.training.o200k.optimizer import build_optimizer


def _config() -> ModelConfig:
    return ModelConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        intermediate_size=64,
        shared_intermediate_size=128,
        mlp_type="tr_hash_engine",
        num_experts=4,
        top_k=2,
        top_k_primary_weight=0.5,
        routing_strategy="token_id_balanced_hash",
        use_shared_routed_gates=False,
    )


def test_model_blocks_use_canonical_tr_hash_engine():
    model = ComplexityModel(_config())
    assert all(isinstance(layer.mlp, TRHashEngineMLP) for layer in model.layers)
    assert not hasattr(model.layers[0].mlp, "gate_proj_w")
    assert not torch.equal(
        model.layers[0].mlp.engine.route_table,
        model.layers[1].mlp.engine.route_table,
    )


def test_tr_hash_engine_model_forward_and_backward():
    model = ComplexityModel(_config())
    token_ids = torch.randint(0, 128, (2, 7))
    output = model(token_ids)["logits"]
    assert output.shape == (2, 7, 128)
    output.float().square().mean().backward()
    assert model.layers[0].mlp.engine.expert_down.grad is not None


def test_optimizer_recognizes_engine_shared_and_expert_parameters():
    model = ComplexityModel(_config())
    args = SimpleNamespace(
        optimizer="adamw",
        lr=3e-4,
        weight_decay=0.1,
        expert_lr_scale=2.0,
        shared_lr_scale=1.0,
    )
    _, metadata = build_optimizer(args, model)
    assert metadata["adamw_expert_params"] > 0
    assert metadata["adamw_shared_params"] > 0
    assert metadata["adamw_params"] == model.num_parameters()
