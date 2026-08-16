import torch

from complexity import ComplexityModel, ModelConfig
from complexity.tr_hash.routing import expand_route_table_hierarchically
from complexity.utils.expert_expansion import (
    convert_checkpoint_to_expanded_experts,
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
        top_k_primary_weight=0.7,
        routing_strategy="token_id_balanced_hash",
        use_custom_kernels=False,
    )


def test_hierarchical_route_expansion_preserves_parent_families() -> None:
    source = torch.tensor([[0, 1, 2, 3], [2, 3, 0, 1]])
    expanded = expand_route_table_hierarchically(
        source,
        source_num_experts=4,
        target_num_experts=8,
        layer_index=3,
    )

    assert expanded.shape == source.shape
    assert torch.equal(expanded.remainder(4), source)
    assert expanded.min().item() >= 0
    assert expanded.max().item() < 8
    assert torch.all(expanded[0] != expanded[1])


def test_four_to_eight_expansion_preserves_logits_and_expert_width() -> None:
    torch.manual_seed(17)
    base = ComplexityModel(_config()).eval()
    expanded = convert_checkpoint_to_expanded_experts(
        base.state_dict(),
        base.config.to_dict(),
    ).eval()
    tokens = torch.randint(0, base.config.vocab_size, (3, 11))

    with torch.inference_mode():
        expected = base(tokens)
        actual = expanded(tokens)

    assert expanded.config.num_experts == 8
    assert expanded.config.top_k == 2
    assert expanded.config.intermediate_size == 64
    assert expanded.config.routing_strategy == "token_id_hierarchical_hash"
    assert base.layers[0].mlp.engine.config.expert_width == 8
    assert expanded.layers[0].mlp.engine.config.expert_width == 8
    assert expanded.num_parameters() > base.num_parameters()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_expansion_clones_every_expert_tensor() -> None:
    base = ComplexityModel(_config())
    expanded = convert_checkpoint_to_expanded_experts(
        base.state_dict(),
        base.config.to_dict(),
    )

    for layer_index in range(base.config.num_hidden_layers):
        source_engine = base.layers[layer_index].mlp.engine
        target_engine = expanded.layers[layer_index].mlp.engine
        for name in ("expert_gate", "expert_up", "expert_down"):
            source = getattr(source_engine, name)
            target = getattr(target_engine, name)
            assert torch.equal(target[:4], source)
            assert torch.equal(target[4:], source)


def test_expansion_preserves_checkpoint_dtype() -> None:
    base = ComplexityModel(_config()).to(dtype=torch.bfloat16)
    expanded = convert_checkpoint_to_expanded_experts(
        base.state_dict(),
        base.config.to_dict(),
    )

    assert expanded.layers[0].mlp.engine.expert_gate.dtype == torch.bfloat16
    assert expanded.embed_tokens.weight.dtype == torch.bfloat16
