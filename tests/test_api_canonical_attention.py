"""Public attention factory defaults for canonical and generic components."""

import pytest
import torch

from complexity.api import Attention


def test_public_tr_hash_attention_defaults_to_four_query_heads_and_one_kv_head():
    attention = Attention.create("tr_hash_gqa", hidden_size=64, num_heads=4)
    assert attention.attention.num_heads == 4
    assert attention.attention.num_kv_heads == 1
    inputs = torch.randn(2, 5, 64)
    token_ids = torch.arange(inputs.size(1)).expand(inputs.size(0), -1)
    with torch.inference_mode():
        output = attention(inputs, token_ids=token_ids)
    assert output.shape == (2, 5, 64)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("kv_argument", ["num_key_value_heads", "kv_heads", "num_kv_heads"])
def test_public_tr_hash_attention_rejects_explicit_noncanonical_kv(kv_argument):
    with pytest.raises(ValueError, match="four query heads and one key/value head"):
        Attention.create("tr_hash_gqa", hidden_size=64, num_heads=4, **{kv_argument: 4})


@pytest.mark.parametrize("attention_type,expected_kv", [("gqa", 4), ("mha", 4), ("mqa", 1)])
def test_public_generic_attention_factory_keeps_its_kv_defaults(attention_type, expected_kv):
    attention = Attention.create(attention_type, hidden_size=64, num_heads=4)
    assert attention.attention.num_heads == 4
    assert attention.attention.num_kv_heads == expected_kv
