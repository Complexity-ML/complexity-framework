from __future__ import annotations

import pytest

from complexity.training.sequence_packing import (
    pack_example_lengths,
    resolve_epoch_schedule,
)


def test_packing_preserves_every_item_once_and_respects_capacity() -> None:
    plan = pack_example_lengths(
        [4, 2, 3, 1], sequence_length=8, separator_tokens=1
    )

    assert plan.packs == ((0, 1), (2, 3))
    assert plan.source_items == 4
    assert plan.packed_items == 2
    assert plan.separator_count == 2
    plan.assert_exact_coverage()


def test_packing_regression_reduces_short_example_padding() -> None:
    plan = pack_example_lengths(
        [128] * 224, sequence_length=512, separator_tokens=1
    )

    # Three 128-token examples plus their two EOS boundaries fit in 512.
    assert plan.packed_items == 75
    assert plan.payload_utilization > 0.74
    assert plan.payload_utilization > plan.naive_payload_utilization
    assert plan.compression_ratio > 2.9


def test_epoch_schedule_derives_ddp_boundaries_from_realized_packs() -> None:
    schedule = resolve_epoch_schedule(
        items=69_069,
        world_size=4,
        batch_size_per_rank=24,
        epochs=3,
    )

    assert schedule.steps_per_epoch == 720
    assert schedule.total_steps == 2_160
    assert schedule.epoch_steps() == (720, 1_440, 2_160)


@pytest.mark.parametrize("lengths", ([], [0], [513]))
def test_packing_rejects_invalid_or_empty_inputs(lengths: list[int]) -> None:
    with pytest.raises(ValueError):
        pack_example_lengths(lengths, sequence_length=512)
