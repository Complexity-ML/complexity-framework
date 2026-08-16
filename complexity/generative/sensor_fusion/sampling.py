"""Deterministic epoch sampling with an exact mid-epoch cursor."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import BatchSampler, Dataset, Sampler


class EpochRandomSampler(Sampler[int]):
    def __init__(self, dataset: Dataset, seed: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.dataset)


class EpochWeightedSampler(Sampler[int]):
    """Deterministic replacement sampling, sharded identically across ranks."""

    def __init__(
        self,
        weights: torch.Tensor,
        seed: int,
        *,
        rank: int = 0,
        world_size: int = 1,
    ):
        weights = torch.as_tensor(weights, dtype=torch.float64, device="cpu")
        if weights.ndim != 1 or weights.numel() == 0:
            raise ValueError("sampling weights must be a non-empty vector")
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("sampling weights must be finite and non-negative")
        if float(weights.sum()) <= 0.0:
            raise ValueError("at least one sampling weight must be positive")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("invalid distributed sampler rank or world size")
        self.weights = weights
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.num_samples = (weights.numel() + world_size - 1) // world_size
        self.total_size = self.num_samples * world_size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=True,
            generator=generator,
        )
        return iter(indices[self.rank : self.total_size : self.world_size].tolist())

    def __len__(self) -> int:
        return self.num_samples


class EpochAnnotatedSampler(Sampler[tuple[int, int]]):
    """Attach the epoch to every index for deterministic sample augmentation."""

    def __init__(self, sampler: Sampler[int]):
        self.sampler = sampler
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        return iter((int(index), self.epoch) for index in self.sampler)

    def __len__(self) -> int:
        return len(self.sampler)


class ResumableBatchSampler(Sampler[list[int]]):
    def __init__(self, batch_sampler: BatchSampler):
        self.batch_sampler = batch_sampler
        self.start_batch = 0

    def set_start_batch(self, start_batch: int) -> None:
        if not 0 <= start_batch <= len(self.batch_sampler):
            raise ValueError("start_batch is outside the epoch")
        self.start_batch = int(start_batch)

    def __iter__(self) -> Iterator[list[int]]:
        iterator = iter(self.batch_sampler)
        for _ in range(self.start_batch):
            next(iterator)
        yield from iterator

    def __len__(self) -> int:
        return len(self.batch_sampler)
