import torch

from complexity import ComplexityModel, ModelConfig
from complexity.core.mlp.tr_hash_engine import TRHashEngineMLP
from complexity.tr_hash import TRHashBackend


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


def test_canonical_engine_honors_explicit_cggr_policy():
    config = _config()
    config.use_cggr = "true"
    model = ComplexityModel(config)
    assert all(
        layer.mlp.engine.config.backend is TRHashBackend.CGGR
        for layer in model.layers
    )


def test_disabling_custom_kernels_forces_reference_backend():
    config = _config()
    config.use_custom_kernels = "false"
    config.use_cggr = "true"
    model = ComplexityModel(config)
    assert all(
        layer.mlp.engine.config.backend is TRHashBackend.PYTORCH
        for layer in model.layers
    )


def test_tr_hash_engine_model_forward_and_backward():
    model = ComplexityModel(_config())
    token_ids = torch.randint(0, 128, (2, 7))
    output = model(token_ids)["logits"]
    assert output.shape == (2, 7, 128)
    output.float().square().mean().backward()
    assert model.layers[0].mlp.engine.expert_down.grad is not None
