"""Tests for the corrective replay plan builder.

A restart with no persisted dataset position (see resume_skip_rows) makes
__iter__() start phase 1 over from shard 0, giving the first few shards of
every source an unplanned extra pass before the run catches up to fresh
material. This corrects the DAMAGE after the fact: phase 1 stays untouched,
but later replay phases stop re-replaying the already-double-exposed shards
and pull in previously-unused ones instead, at the same row counts.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.build_corrective_replay_plan import build_corrective_replay_plan
from scripts.build_tr_hash_70b_replay_plan import build_replay_plan


def _fake_dataset(shards_by_source: dict[str, list[dict]]):
    """seq_len=1 keeps rows == tokens, no unit conversion needed."""
    from complexity.training.corpus_mixture import PretokenizedCorpusMixtureDataset

    dataset = PretokenizedCorpusMixtureDataset.__new__(PretokenizedCorpusMixtureDataset)
    dataset.seq_len = 1
    dataset.sources = tuple(SimpleNamespace(name=name) for name in shards_by_source)
    dataset._source_manifests = {
        name: {"shards": shards} for name, shards in shards_by_source.items()
    }
    return dataset


def _shards(prefix: str, count: int, rows_each: int) -> list[dict]:
    return [{"file": f"{prefix}_{i}.bin", "rows": rows_each} for i in range(count)]


# "b" gets 10 shards @ 2 rows: phase 1 uses the first 5 (unique budget 10),
# leaving shards 5-9 unused and available as fresh backfill material.
BASE_SHARDS = {
    "a": _shards("a", 10, 2),  # unique budget 10, no replay -- correction never touches it
    "b": _shards("b", 10, 2),  # unique budget 10, replay x2 -- correction target
}


def test_corrective_plan_leaves_phase_1_bit_for_bit_identical():
    dataset = _fake_dataset(BASE_SHARDS)
    uncorrected = build_replay_plan(
        dataset,
        unique_token_budgets={"a": 10, "b": 10},
        replay_passes={"a": 1, "b": 2},
        row_alignment=1,
    )

    corrected = build_corrective_replay_plan(
        dataset,
        unique_token_budgets={"a": 10, "b": 10},
        replay_passes={"a": 1, "b": 2},
        already_double_exposed_shards={"b": 2},
        row_alignment=1,
    )

    assert corrected["phases"][0] == uncorrected["phases"][0]
    assert corrected["phases"][0]["name"] == "unique_core"


def test_corrective_plan_drops_burned_shards_and_backfills_with_unused_ones():
    dataset = _fake_dataset(BASE_SHARDS)

    corrected = build_corrective_replay_plan(
        dataset,
        unique_token_budgets={"a": 10, "b": 10},
        replay_passes={"a": 1, "b": 2},
        already_double_exposed_shards={"b": 2},
        row_alignment=1,
    )

    replay_phase = corrected["phases"][1]
    assert replay_phase["name"] == "quality_replay_2_corrected"
    b_selection = replay_phase["sources"]["b"]
    files = [s["file"] for s in b_selection]

    # The 2 burned shards (b_0, b_1 -- phase 1's first two) must not reappear.
    assert "b_0.bin" not in files
    assert "b_1.bin" not in files
    # The kept (non-burned) phase-1 shards (b_2, b_3, b_4) are still replayed.
    assert {"b_2.bin", "b_3.bin", "b_4.bin"}.issubset(set(files))
    # Backfill comes from shards phase 1 never touched at all (b_5..b_9).
    backfill = set(files) - {"b_2.bin", "b_3.bin", "b_4.bin"}
    assert backfill and backfill.issubset({f"b_{i}.bin" for i in range(5, 10)})
    # Same total row count as an uncorrected replay of this source (10 rows).
    assert sum(s["rows"] for s in b_selection) == 10


def test_corrective_plan_preserves_total_trained_tokens():
    dataset = _fake_dataset(BASE_SHARDS)
    uncorrected = build_replay_plan(
        dataset,
        unique_token_budgets={"a": 10, "b": 10},
        replay_passes={"a": 1, "b": 2},
        row_alignment=1,
    )

    corrected = build_corrective_replay_plan(
        dataset,
        unique_token_budgets={"a": 10, "b": 10},
        replay_passes={"a": 1, "b": 2},
        already_double_exposed_shards={"b": 2},
        row_alignment=1,
    )

    assert corrected["trained_tokens"] == uncorrected["trained_tokens"]
    assert corrected["unique_tokens"] == uncorrected["unique_tokens"]
    assert corrected["source_passes"] == uncorrected["source_passes"]


def test_source_with_zero_burned_shards_is_untouched():
    dataset = _fake_dataset(BASE_SHARDS)
    uncorrected = build_replay_plan(
        dataset,
        unique_token_budgets={"a": 10, "b": 10},
        replay_passes={"a": 1, "b": 2},
        row_alignment=1,
    )

    corrected = build_corrective_replay_plan(
        dataset,
        unique_token_budgets={"a": 10, "b": 10},
        replay_passes={"a": 1, "b": 2},
        already_double_exposed_shards={"b": 0},
        row_alignment=1,
    )

    assert corrected["phases"][1]["sources"]["b"] == uncorrected["phases"][1]["sources"]["b"]


def test_a_source_with_no_scheduled_replay_pass_is_never_corrected():
    """dclm-like source: replay_passes=1 means it never appears in any later
    phase regardless of what already_double_exposed_shards says -- there's
    nothing to correct since it was never going to be replayed anyway."""
    dataset = _fake_dataset(BASE_SHARDS)

    corrected = build_corrective_replay_plan(
        dataset,
        unique_token_budgets={"a": 10, "b": 10},
        replay_passes={"a": 1, "b": 2},
        already_double_exposed_shards={"a": 5, "b": 2},  # "a" burned count is a no-op
        row_alignment=1,
    )

    assert len(corrected["phases"]) == 2  # unique_core + one replay phase for "b" only
    assert "a" not in corrected["phases"][1]["sources"]


def test_rejects_a_burned_shard_count_naming_an_unknown_source():
    dataset = _fake_dataset(BASE_SHARDS)
    with pytest.raises(ValueError, match="unknown sources"):
        build_corrective_replay_plan(
            dataset,
            unique_token_budgets={"a": 10, "b": 10},
            replay_passes={"a": 1, "b": 2},
            already_double_exposed_shards={"c": 1},
            row_alignment=1,
        )


def test_rejects_a_burned_shard_count_larger_than_phase_1_actually_used():
    dataset = _fake_dataset(BASE_SHARDS)
    with pytest.raises(ValueError, match="exceeds"):
        build_corrective_replay_plan(
            dataset,
            unique_token_budgets={"a": 10, "b": 10},
            replay_passes={"a": 1, "b": 2},
            already_double_exposed_shards={"b": 6},  # phase 1 only used 5 shards for b
            row_alignment=1,
        )


def test_raises_when_not_enough_fresh_shards_exist_to_backfill():
    # "b" here has exactly 5 shards -- all consumed by phase 1, none left over.
    dataset = _fake_dataset({"a": _shards("a", 10, 2), "b": _shards("b", 5, 2)})
    with pytest.raises(ValueError, match="not enough fresh shards"):
        build_corrective_replay_plan(
            dataset,
            unique_token_budgets={"a": 10, "b": 10},
            replay_passes={"a": 1, "b": 2},
            already_double_exposed_shards={"b": 2},
            row_alignment=1,
        )
