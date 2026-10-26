"""The language model derives the canonical attention from its TR-HASH MLP."""

import pytest
import torch
from complexity import ComplexityModel, ModelConfig
from complexity.core.attention.gqa import TRHashAttention, GroupedQueryAttention
from complexity.core.attention import AttentionConfig
from complexity.core.mlp.tr_hash_engine import TRHashEngineMLP


def test_model_builds_canonical_attention_and_mlp_without_profile_fields():
    config = ModelConfig(
        hidden_size=32, vocab_size=64, num_hidden_layers=2, use_custom_kernels=False
    )
    assert not any("profile" in key for key in config.to_dict())
    model = ComplexityModel(config)
    assert all(
        isinstance(layer.self_attn, TRHashAttention) and isinstance(layer.mlp, TRHashEngineMLP)
        for layer in model.layers
    )
    logits = model(torch.tensor([[1, 5, 9, 12]]))["logits"]
    logits.square().sum().backward()
    assert model.layers[1].mlp.engine.expert_down.grad is not None


def test_generic_gqa_component_keeps_its_ordinary_projection_contract():
    attention = GroupedQueryAttention(
        AttentionConfig(hidden_size=24, num_attention_heads=3, num_key_value_heads=1)
    )
    output, _ = attention(torch.randn(2, 5, 24))
    assert output.shape == (2, 5, 24)


@pytest.mark.parametrize(
    "field", ["architecture_profile", "architecture_profile_version", "route_hash_count"]
)
def test_removed_selector_fields_cannot_be_silently_consumed(field):
    with pytest.raises(ValueError, match="Obsolete"):
        ModelConfig.from_dict({field: 1})


@pytest.mark.parametrize(
    "settings",
    [
        {"expert_initialization": "legacy_kaiming"},
        {"hidden_act": "gelu"},
        {"use_shared_routed_gates": True},
        {"learn_hash_pair_gates": True},
        {"learn_hash_channel_modulation": True},
        {"lsh_routing": True},
        {"hash_routing": "learned"},
        {"shared_gate_init": 0.5},
    ],
)
def test_generic_mlp_switches_cannot_change_the_canonical_tr_layer(settings):
    from complexity.core.mlp.base import MLPConfig

    with pytest.raises(ValueError, match="TR-HASH"):
        TRHashEngineMLP(MLPConfig(hidden_size=16, intermediate_size=16, **settings))
