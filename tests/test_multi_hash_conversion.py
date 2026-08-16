import torch

from complexity import ComplexityModel, ModelConfig
from complexity.utils.multi_hash_conversion import (
    convert_checkpoint_to_multi_hash,
)


def _config() -> ModelConfig:
    return ModelConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        intermediate_size=32,
        shared_intermediate_size=64,
        vocab_size=97,
        mlp_type="tr_hash_engine",
        num_experts=4,
        top_k=2,
        top_k_primary_weight=0.5,
        routing_strategy="token_id_balanced_hash",
    )


def test_conversion_keeps_weights_but_rebuilds_routes() -> None:
    base = ComplexityModel(_config())
    converted = convert_checkpoint_to_multi_hash(
        base.state_dict(),
        base.config.to_dict(),
        route_hash_count=3,
    )

    assert base.config.routing_strategy == "token_id_balanced_hash"
    assert converted.config.routing_strategy == "token_id_multi_hash"
    assert converted.config.route_hash_count == 3
    assert converted.config.num_experts == 4
    assert converted.config.top_k == 2
    assert not torch.equal(
        base.layers[0].mlp.engine.route_table,
        converted.layers[0].mlp.engine.route_table,
    )
    for key, value in base.state_dict().items():
        if ".engine.route_table" in key or ".engine.fused_" in key:
            continue
        assert torch.equal(value, converted.state_dict()[key]), key


def test_conversion_rejects_non_four_expert_recipe() -> None:
    config = _config().to_dict()
    config["num_experts"] = 8
    try:
        convert_checkpoint_to_multi_hash({}, config)
    except ValueError as exc:
        assert "exactly 4 experts" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected the four-expert guardrail")


def test_multi_hash_conversion_preserves_checkpoint_dtype() -> None:
    base = ComplexityModel(_config()).to(dtype=torch.bfloat16)
    converted = convert_checkpoint_to_multi_hash(
        base.state_dict(),
        base.config.to_dict(),
    )

    assert converted.layers[0].mlp.engine.expert_gate.dtype == torch.bfloat16
    assert converted.embed_tokens.weight.dtype == torch.bfloat16
