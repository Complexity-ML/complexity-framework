"""Tests for WeightedPairMixtureDataset and sqrt_normalized_weights
(complexity/training/embedding_pairs.py)."""

from __future__ import annotations

import pytest
from transformers import PreTrainedTokenizerFast

from complexity.training.embedding_pairs import (
    WeightedPairMixtureDataset,
    sqrt_normalized_weights,
)


@pytest.fixture(scope="module")
def tokenizer():
    return PreTrainedTokenizerFast.from_pretrained("tokenizer")


def _fake_pairs(prefix: str, count: int, *, with_negatives: bool = False) -> list[dict]:
    rows = []
    for i in range(count):
        row = {"query": f"{prefix} q{i}", "document": f"{prefix} d{i}"}
        if with_negatives:
            row["negative"] = [f"{prefix} neg{i}_{j}" for j in range(3)]
        rows.append(row)
    return rows


def test_sqrt_normalized_weights_sums_to_one_and_boosts_small_sources():
    counts = {"huge": 66_204_599, "tiny": 25_117}
    weights = sqrt_normalized_weights(counts)

    assert sum(weights.values()) == pytest.approx(1.0)
    # Raw share of "tiny" is ~0.00038%; sqrt-weighting should lift it far
    # above that raw proportion without making it dominant.
    raw_share = counts["tiny"] / sum(counts.values())
    assert weights["tiny"] > raw_share * 10  # sqrt(66.2M / 25.1K) ~= 51x actual boost
    assert weights["tiny"] < weights["huge"]


def test_sqrt_normalized_weights_rejects_empty_input():
    with pytest.raises(ValueError, match="non-empty"):
        sqrt_normalized_weights({})


def test_mixture_consumes_every_source_exactly_once(tokenizer):
    streams = {"big": _fake_pairs("big", 200), "small": _fake_pairs("small", 10)}
    dataset = WeightedPairMixtureDataset(
        tokenizer, dataset_id="fake", split_weights={"big": 0.9, "small": 0.1},
        max_seq_len=8, streams=streams,
    )

    seen = list(dataset)
    assert len(seen) == 210


def test_mixture_interleaving_matches_configured_weight_while_both_active(tokenizer):
    streams = {"big": _fake_pairs("big", 2000), "small": _fake_pairs("small", 50)}
    dataset = WeightedPairMixtureDataset(
        tokenizer, dataset_id="fake", split_weights={"big": 0.9, "small": 0.1},
        max_seq_len=8, streams=streams,
    )

    order = [
        "small" if tokenizer.decode(ex["anchor_input_ids"], skip_special_tokens=True).startswith("small") else "big"
        for ex in dataset
    ]
    last_small_index = max(i for i, name in enumerate(order) if name == "small")
    window = order[: last_small_index + 1]

    # Small has fully exhausted itself by here, so its share over this
    # window should track its configured weight closely, not its tiny raw
    # proportion of the total (50 / 2050 ~= 2.4%).
    assert window.count("small") / len(window) == pytest.approx(0.1, abs=0.02)


def test_hard_negatives_are_sampled_and_shaped_correctly(tokenizer):
    streams = {"src": _fake_pairs("src", 20, with_negatives=True)}
    dataset = WeightedPairMixtureDataset(
        tokenizer, dataset_id="fake", split_weights={"src": 1.0},
        max_seq_len=8, num_hard_negatives=2, streams=streams,
    )

    example = next(iter(dataset))
    assert example["negative_input_ids"].shape == (2, 8)
    assert example["negative_attention_mask"].shape == (2, 8)


def test_hard_negatives_sample_with_replacement_when_list_is_short(tokenizer):
    streams = {"src": _fake_pairs("src", 5, with_negatives=True)}  # each row has 3 negatives
    dataset = WeightedPairMixtureDataset(
        tokenizer, dataset_id="fake", split_weights={"src": 1.0},
        max_seq_len=8, num_hard_negatives=6, streams=streams,  # more than available
    )

    example = next(iter(dataset))
    assert example["negative_input_ids"].shape == (6, 8)


def test_rows_without_negatives_are_skipped_when_hard_negatives_requested(tokenizer):
    streams = {"src": _fake_pairs("src", 5, with_negatives=False)}
    dataset = WeightedPairMixtureDataset(
        tokenizer, dataset_id="fake", split_weights={"src": 1.0},
        max_seq_len=8, num_hard_negatives=2, streams=streams,
    )

    assert list(dataset) == []


def test_default_target_tokens_none_stops_at_one_natural_pass(tokenizer):
    streams = {"a": _fake_pairs("a", 10)}
    dataset = WeightedPairMixtureDataset(
        tokenizer, dataset_id="fake", split_weights={"a": 1.0}, max_seq_len=8, streams=streams,
    )

    assert len(list(dataset)) == 10


def test_target_tokens_above_natural_total_replays_exhausted_sources(tokenizer):
    streams = {"a": _fake_pairs("a", 10)}
    natural = list(
        WeightedPairMixtureDataset(
            tokenizer, dataset_id="fake", split_weights={"a": 1.0}, max_seq_len=8,
            streams={"a": _fake_pairs("a", 10)},
        )
    )
    per_example_tokens = int(natural[0]["anchor_attention_mask"].sum()) + int(natural[0]["positive_attention_mask"].sum())
    natural_total = per_example_tokens * len(natural)

    dataset = WeightedPairMixtureDataset(
        tokenizer, dataset_id="fake", split_weights={"a": 1.0}, max_seq_len=8,
        streams=streams, target_tokens=natural_total + per_example_tokens * 5,
    )
    replayed = list(dataset)

    # Must exceed one natural pass (proves restart happened) and land at
    # or just above the target once the in-flight example finishes.
    assert len(replayed) > len(natural)
    total_tokens = sum(
        int(ex["anchor_attention_mask"].sum()) + int(ex["positive_attention_mask"].sum())
        for ex in replayed
    )
    assert total_tokens >= natural_total + per_example_tokens * 5


def test_no_hard_negative_keys_when_num_hard_negatives_is_zero(tokenizer):
    streams = {"src": _fake_pairs("src", 3, with_negatives=True)}
    dataset = WeightedPairMixtureDataset(
        tokenizer, dataset_id="fake", split_weights={"src": 1.0},
        max_seq_len=8, num_hard_negatives=0, streams=streams,
    )

    example = next(iter(dataset))
    assert "negative_input_ids" not in example
