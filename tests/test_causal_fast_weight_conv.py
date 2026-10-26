from dataclasses import replace

import pytest
import torch
from torch import nn

from complexity.core.attention import AttentionConfig, CausalFastWeightConvMixer
from complexity.core.context.associative import SharedAssociativeContext


def _config() -> AttentionConfig:
    return AttentionConfig(
        vocab_size=128,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        causal_conv_kernel_size=3,
        causal_state_rank=8,
    )


def _mixers(config: AttentionConfig) -> nn.ModuleList:
    """Exercise generic mixers with one context shared by both layers."""
    mixers = nn.ModuleList(
        CausalFastWeightConvMixer(replace(config, layer_idx=index, causal_conv_dilation=2**index))
        for index in range(2)
    )
    mixers[1].shared_context = mixers[0].shared_context
    return mixers.eval()


def _forward(mixers, hidden_states, token_ids, past_key_values=None, *, use_cache=False):
    next_cache = []
    for index, mixer in enumerate(mixers):
        residual, state = mixer(
            hidden_states,
            token_ids=token_ids,
            past_key_value=None if past_key_values is None else past_key_values[index],
            use_cache=use_cache,
        )
        hidden_states = hidden_states + residual
        next_cache.append(state)
    return hidden_states, next_cache if use_cache else None


def test_fast_weight_conv_is_causal_and_has_fixed_cache() -> None:
    torch.manual_seed(0)
    config = _config()
    mixers = _mixers(config)
    embedding = nn.Embedding(config.vocab_size, config.hidden_size)
    mechanisms = [mixer.shared_context for mixer in mixers]
    assert all(isinstance(mechanism, SharedAssociativeContext) for mechanism in mechanisms)
    assert all(mechanism is mechanisms[0] for mechanism in mechanisms)
    assert [mixer.context_enabled for mixer in mixers] == [True, True]
    shared_ids = {id(parameter) for parameter in mechanisms[0].parameters()}
    registered_ids = [id(parameter) for parameter in mixers.parameters()]
    assert all(registered_ids.count(parameter_id) == 1 for parameter_id in shared_ids)
    token_ids = torch.tensor([[17, 23, 17]])
    lexical_keys = mechanisms[0]._lexical_keys(token_ids)
    assert torch.equal(lexical_keys[:, 0], lexical_keys[:, 2])
    assert not torch.equal(lexical_keys[:, 0], lexical_keys[:, 1])
    separation_probe = SharedAssociativeContext(hidden_size=32, rank=128)
    vocabulary_keys = separation_probe._lexical_keys(torch.arange(16, 128)[None])[0]
    similarities = (vocabulary_keys @ vocabulary_keys.T).abs()
    similarities.fill_diagonal_(0)
    assert similarities.max().item() < 0.5
    prefix = torch.randint(0, 128, (1, 8))
    suffix_a = torch.randint(0, 128, (1, 8))
    suffix_b = torch.randint(0, 128, (1, 8))

    with torch.inference_mode():
        sequence = torch.cat((prefix, suffix_a), dim=1)
        alternative = torch.cat((prefix, suffix_b), dim=1)
        output_a, _ = _forward(mixers, embedding(sequence), sequence)
        output_b, _ = _forward(mixers, embedding(alternative), alternative)
        cache = None
        pointers = None
        incremental = []
        for index in range(sequence.shape[1]):
            token = sequence[:, index : index + 1]
            output, cache = _forward(mixers, embedding(token), token, cache, use_cache=True)
            assert cache is not None
            current_pointers = [tensor.data_ptr() for layer in cache for tensor in layer]
            if pointers is None:
                pointers = current_pointers
            else:
                assert current_pointers == pointers
            incremental.append(output)

    assert torch.equal(output_a[:, : prefix.shape[1]], output_b[:, : prefix.shape[1]])
    torch.testing.assert_close(output_a, torch.cat(incremental, dim=1), atol=2e-5, rtol=2e-5)
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
    config = replace(
        _config(),
        causal_stable_delta=True,
        causal_delta_collision_normalized=True,
        causal_delta_lexical_forge=True,
        causal_delta_occurrence_address=occurrence_address,
        lexical_object_rank=16,
    )
    mixers = _mixers(config)
    embedding = nn.Embedding(config.vocab_size, config.hidden_size)
    sequence = torch.randint(0, config.vocab_size, (1, 6))

    with torch.inference_mode():
        full, _ = _forward(mixers, embedding(sequence), sequence)
        cache = None
        pieces = []
        pointers = None
        for position in range(sequence.shape[1]):
            token = sequence[:, position : position + 1]
            output, cache = _forward(mixers, embedding(token), token, cache, use_cache=True)
            assert cache is not None
            assert all(len(layer_cache) == expected_cache_arity for layer_cache in cache)
            current_pointers = [tensor.data_ptr() for layer in cache for tensor in layer]
            if pointers is None:
                pointers = current_pointers
            else:
                assert current_pointers == pointers
            pieces.append(output)

    torch.testing.assert_close(torch.cat(pieces, dim=1), full, rtol=2e-5, atol=2e-5)
