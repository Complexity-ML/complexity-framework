"""Deterministic sequence-packing and epoch scheduling primitives.

These helpers keep data-efficiency decisions out of individual launchers. A
packing plan never drops or splits an example; callers remain responsible for
placing an EOS separator between the examples listed in each pack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SequencePackingPlan:
    packs: tuple[tuple[int, ...], ...]
    lengths: tuple[int, ...]
    sequence_length: int
    separator_tokens: int

    @property
    def source_items(self) -> int:
        return len(self.lengths)

    @property
    def packed_items(self) -> int:
        return len(self.packs)

    @property
    def source_tokens(self) -> int:
        return sum(self.lengths)

    @property
    def separator_count(self) -> int:
        return sum(max(0, len(pack) - 1) for pack in self.packs)

    @property
    def capacity_tokens(self) -> int:
        return self.packed_items * self.sequence_length

    @property
    def payload_utilization(self) -> float:
        return self.source_tokens / max(1, self.capacity_tokens)

    @property
    def occupied_utilization(self) -> float:
        occupied = self.source_tokens + self.separator_count * self.separator_tokens
        return occupied / max(1, self.capacity_tokens)

    @property
    def naive_payload_utilization(self) -> float:
        naive_capacity = self.source_items * self.sequence_length
        return self.source_tokens / max(1, naive_capacity)

    @property
    def compression_ratio(self) -> float:
        return self.source_items / max(1, self.packed_items)

    def assert_exact_coverage(self) -> None:
        flattened = [item for pack in self.packs for item in pack]
        expected = list(range(self.source_items))
        if sorted(flattened) != expected:
            raise ValueError("packing plan must contain every source item exactly once")
        for pack in self.packs:
            occupied = sum(self.lengths[item] for item in pack)
            occupied += max(0, len(pack) - 1) * self.separator_tokens
            if occupied > self.sequence_length:
                raise ValueError("packing plan contains an over-capacity sequence")


@dataclass(frozen=True)
class EpochSchedule:
    items: int
    world_size: int
    batch_size_per_rank: int
    epochs: int
    steps_per_epoch: int
    total_steps: int

    def epoch_steps(self) -> tuple[int, ...]:
        return tuple(
            self.steps_per_epoch * epoch
            for epoch in range(1, self.epochs + 1)
        )


def pack_example_lengths(
    lengths: Iterable[int],
    *,
    sequence_length: int,
    separator_tokens: int = 1,
) -> SequencePackingPlan:
    """Create stable next-fit packs without splitting or dropping examples."""

    sequence_length = int(sequence_length)
    separator_tokens = int(separator_tokens)
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    if separator_tokens < 0:
        raise ValueError("separator_tokens cannot be negative")
    normalized = tuple(int(length) for length in lengths)
    if not normalized:
        raise ValueError("cannot pack an empty example set")
    if any(length < 1 or length > sequence_length for length in normalized):
        raise ValueError("every example length must be within the sequence window")

    packs: list[tuple[int, ...]] = []
    current: list[int] = []
    occupied = 0
    for item, length in enumerate(normalized):
        separator = separator_tokens if current else 0
        if current and occupied + separator + length > sequence_length:
            packs.append(tuple(current))
            current = []
            occupied = 0
            separator = 0
        current.append(item)
        occupied += separator + length
    if current:
        packs.append(tuple(current))

    plan = SequencePackingPlan(
        packs=tuple(packs),
        lengths=normalized,
        sequence_length=sequence_length,
        separator_tokens=separator_tokens,
    )
    plan.assert_exact_coverage()
    if plan.packed_items > plan.source_items:
        raise ValueError("packing cannot create more windows than source examples")
    return plan


def resolve_epoch_schedule(
    *,
    items: int,
    world_size: int,
    batch_size_per_rank: int,
    epochs: int,
) -> EpochSchedule:
    """Resolve exact DDP epoch boundaries from realized training items."""

    values = {
        "items": int(items),
        "world_size": int(world_size),
        "batch_size_per_rank": int(batch_size_per_rank),
        "epochs": int(epochs),
    }
    if any(value < 1 for value in values.values()):
        raise ValueError("epoch schedule values must all be positive")
    items_per_rank = math.ceil(values["items"] / values["world_size"])
    steps_per_epoch = math.ceil(items_per_rank / values["batch_size_per_rank"])
    return EpochSchedule(
        **values,
        steps_per_epoch=steps_per_epoch,
        total_steps=steps_per_epoch * values["epochs"],
    )
