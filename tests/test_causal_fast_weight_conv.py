import pytest
import torch

from complexity.config import ModelConfig
from complexity.core.context.associative import SharedAssociativeContext
from complexity.models import ComplexityModel


def _config() -> ModelConfig:
    return ModelConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        attention_type="causal_fast_weight_conv",
        causal_conv_kernel_size=3,
        causal_conv_dilation_cycle=2,
        causal_state_rank=8,
        mlp_type="swiglu",
        use_cache=True,
    )


def test_fast_weight_conv_is_causal_and_has_fixed_cache() -> None:
    torch.manual_seed(0)
    model = ComplexityModel(_config()).eval()
    mechanisms = [
        getattr(layer.self_attn, "shared_context") for layer in model.layers
    ]
    assert all(
        isinstance(mechanism, SharedAssociativeContext)
        for mechanism in mechanisms
    )
    assert all(mechanism is mechanisms[0] for mechanism in mechanisms)
    assert [getattr(layer.self_attn, "context_enabled") for layer in model.layers] == [
        True,
        True,
    ]
    assert all(
        name.count("shared_context") <= 1
        for name, _ in model.named_parameters(remove_duplicate=True)
    )
    mixer = mechanisms[0]
    token_ids = torch.tensor([[17, 23, 17]])
    lexical_keys = getattr(mixer, "_lexical_keys")(token_ids)
    assert torch.equal(lexical_keys[:, 0], lexical_keys[:, 2])
    assert not torch.equal(lexical_keys[:, 0], lexical_keys[:, 1])
    separation_probe = SharedAssociativeContext(hidden_size=32, rank=128)
    vocabulary_keys = separation_probe._lexical_keys(
        torch.arange(16, 128)[None]
    )[0]
    similarities = (vocabulary_keys @ vocabulary_keys.T).abs()
    similarities.fill_diagonal_(0)
    assert similarities.max().item() < 0.5
    prefix = torch.randint(0, 128, (1, 8))
    suffix_a = torch.randint(0, 128, (1, 8))
    suffix_b = torch.randint(0, 128, (1, 8))

    with torch.inference_mode():
        output_a = model(
            torch.cat((prefix, suffix_a), dim=1), return_logits=False
        )["last_hidden_state"]
        output_b = model(
            torch.cat((prefix, suffix_b), dim=1), return_logits=False
        )["last_hidden_state"]

        cache = None
        pointers = None
        incremental = []
        sequence = torch.cat((prefix, suffix_a), dim=1)
        for index in range(sequence.shape[1]):
            output = model(
                sequence[:, index : index + 1],
                past_key_values=cache,
                use_cache=True,
                return_logits=False,
            )
            cache = output["past_key_values"]
            assert cache is not None
            current_pointers = [tensor.data_ptr() for layer in cache for tensor in layer]
            if pointers is None:
                pointers = current_pointers
            else:
                assert current_pointers == pointers
            incremental.append(output["last_hidden_state"])

    assert torch.equal(output_a[:, : prefix.shape[1]], output_b[:, : prefix.shape[1]])
    assert torch.allclose(output_a, torch.cat(incremental, dim=1), atol=2e-5, rtol=2e-5)
    assert cache is not None
    assert all(len(layer_cache) == 3 for layer_cache in cache)
    assert all(layer_cache[1].shape == (1, 8, 8) for layer_cache in cache)
    assert all(layer_cache[2].shape == (1, 8) for layer_cache in cache)


@pytest.mark.parametrize(
    ("occurrence_address", "expected_cache_arity"),
    [(False, 4), (True, 5)],
)
def test_collision_context_cache_supports_variable_state_arity(
    occurrence_address: bool, expected_cache_arity: int
) -> None:
    config = _config()
    config.causal_stable_delta = True
    config.causal_delta_collision_normalized = True
    config.causal_delta_lexical_forge = True
    config.causal_delta_occurrence_address = occurrence_address
    config.mlp_type = "lexical_modulated"
    config.lexical_object_rank = 16
    model = ComplexityModel(config).eval()
    sequence = torch.randint(0, config.vocab_size, (1, 6))

    with torch.inference_mode():
        full = model(sequence, return_logits=False)["last_hidden_state"]
        cache = None
        pieces = []
        pointers = None
        for position in range(sequence.shape[1]):
            output = model(
                sequence[:, position : position + 1],
                past_key_values=cache,
                use_cache=True,
                return_logits=False,
            )
            cache = output["past_key_values"]
            assert cache is not None
            assert all(
                len(layer_cache) == expected_cache_arity for layer_cache in cache
            )
            current_pointers = [
                tensor.data_ptr() for layer in cache for tensor in layer
            ]
            if pointers is None:
                pointers = current_pointers
            else:
                assert current_pointers == pointers
            pieces.append(output["last_hidden_state"])

    torch.testing.assert_close(
        torch.cat(pieces, dim=1), full, rtol=2e-5, atol=2e-5
    )
