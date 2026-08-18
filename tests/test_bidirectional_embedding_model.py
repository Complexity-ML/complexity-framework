"""Tests for BidirectionalEmbeddingModel (complexity/models/embedding.py)."""

from __future__ import annotations

import tempfile

import pytest
import torch

from complexity.config import ModelConfig
from complexity.models.embedding import (
    BidirectionalEmbeddingModel,
    build_extended_attention_mask,
    mean_pool,
)


def _tiny_config(**overrides) -> ModelConfig:
    values = dict(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=32,
        shared_intermediate_size=128,
        num_experts=2,
        top_k=1,
        vocab_size=200,
        max_position_embeddings=32,
        attention_type="gqa",
        mlp_type="tr_hash_engine",
        routing_strategy="token_id_multi_hash",
        route_hash_count=2,
        shared_expert=True,
        use_qk_norm=True,
        tie_word_embeddings=True,
        is_causal=False,
    )
    values.update(overrides)
    return ModelConfig(**values)


def test_rejects_a_causal_config():
    with pytest.raises(ValueError, match="is_causal=False"):
        BidirectionalEmbeddingModel(_tiny_config(is_causal=True))


def test_forward_returns_l2_normalized_embeddings():
    model = BidirectionalEmbeddingModel(_tiny_config())
    input_ids = torch.randint(0, 200, (3, 8))
    attention_mask = torch.ones(3, 8)

    embeddings = model(input_ids, attention_mask)

    assert embeddings.shape == (3, 64)
    assert torch.isfinite(embeddings).all()
    assert torch.allclose(embeddings.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_padding_does_not_change_real_token_pooling():
    """Two sequences identical except for what's stuffed into the padded
    tail must produce the same embedding -- padding must be excluded from
    both attention and pooling."""
    model = BidirectionalEmbeddingModel(_tiny_config())
    model.eval()

    input_ids = torch.randint(0, 200, (1, 8))
    attention_mask = torch.ones(1, 8)
    attention_mask[0, 5:] = 0

    input_ids_different_padding = input_ids.clone()
    input_ids_different_padding[0, 5:] = torch.randint(0, 200, (3,))

    with torch.no_grad():
        emb_a = model(input_ids, attention_mask)
        emb_b = model(input_ids_different_padding, attention_mask)

    assert torch.allclose(emb_a, emb_b, atol=1e-5)


def test_save_and_load_round_trip_matches():
    model = BidirectionalEmbeddingModel(_tiny_config())
    model.eval()
    input_ids = torch.randint(0, 200, (2, 8))
    attention_mask = torch.ones(2, 8)

    with torch.no_grad():
        before = model(input_ids, attention_mask)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model.save_pretrained(tmp_dir)
        loaded = BidirectionalEmbeddingModel.from_pretrained(tmp_dir)
        loaded.eval()
        with torch.no_grad():
            after = loaded(input_ids, attention_mask)

    assert torch.allclose(before, after, atol=1e-6)
    assert loaded.config.is_causal is False


def test_build_extended_attention_mask_shape_and_values():
    padding_mask = torch.tensor([[1.0, 1.0, 0.0]])
    mask = build_extended_attention_mask(padding_mask, dtype=torch.float32)

    assert mask.shape == (1, 1, 1, 3)
    assert mask[0, 0, 0, 0] == 0.0
    assert mask[0, 0, 0, 1] == 0.0
    assert mask[0, 0, 0, 2] == torch.finfo(torch.float32).min


def test_mean_pool_excludes_padding():
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])  # last token is padding
    padding_mask = torch.tensor([[1.0, 1.0, 0.0]])

    pooled = mean_pool(hidden, padding_mask)

    assert torch.allclose(pooled, torch.tensor([[2.0, 2.0]]))
