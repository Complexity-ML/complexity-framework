"""Small DDP helpers for detector training and exact checkpoint resume."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DistributedSampler, Sampler


class DistributedEvalSampler(Sampler[int]):
    """Shard validation without padding or duplicating samples."""

    def __init__(self, dataset: Dataset, rank: int, world_size: int):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        return max((remaining + self.world_size - 1) // self.world_size, 0)


@dataclass(frozen=True)
class DistributedContext:
    device: torch.device
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @classmethod
    def initialize(cls, device: torch.device) -> "DistributedContext":
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size == 1:
            return cls(device=device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("detector DDP currently requires CUDA")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return cls(
            device=torch.device("cuda", local_rank),
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            local_rank=local_rank,
        )

    def train_sampler(self, dataset: Dataset, seed: int) -> DistributedSampler | None:
        if not self.enabled:
            return None
        return DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            seed=seed,
            drop_last=False,
        )

    def eval_sampler(self, dataset: Dataset) -> DistributedEvalSampler | None:
        if not self.enabled:
            return None
        return DistributedEvalSampler(dataset, self.rank, self.world_size)

    def wrap(self, model: torch.nn.Module) -> torch.nn.Module:
        if not self.enabled:
            return model
        return DistributedDataParallel(
            model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            broadcast_buffers=False,
        )

    def all_gather_objects(self, value: Any) -> list[Any]:
        if not self.enabled:
            return [value]
        values: list[Any] = [None] * self.world_size
        dist.all_gather_object(values, value)
        return values

    def mean_scalars(self, values: Mapping[str, float]) -> dict[str, float]:
        if not self.enabled:
            return dict(values)
        names = sorted(values)
        tensor = torch.tensor(
            [values[name] for name in names],
            dtype=torch.float64,
            device=self.device,
        )
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= self.world_size
        return dict(zip(names, tensor.cpu().tolist()))

    def gather_rng_states(self) -> Sequence[Mapping[str, torch.Tensor]] | None:
        if not self.enabled:
            return None
        local_state = {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(self.device),
        }
        return self.all_gather_objects(local_state)

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier(device_ids=[self.local_rank])

    def close(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()
