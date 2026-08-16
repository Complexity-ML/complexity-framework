"""
Distributed DataLoader for LLM training.

Features:
- Distributed sampling across GPUs
- Efficient batching and padding
- Support for streaming datasets
- Automatic sequence packing
"""

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from typing import Optional, Dict, Any, Iterator, List, Callable
from dataclasses import dataclass
import math


@dataclass
class DataConfig:
    """Configuration for data loading."""
    batch_size: int = 32
    seq_length: int = 2048
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True
    shuffle: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True


class DistributedDataLoader:
    """
    DataLoader optimized for distributed LLM training.

    Features:
    - Automatic distributed sampling
    - Efficient batching with padding
    - Support for sequence packing
    - Resumable iteration
    """

    def __init__(
        self,
        dataset: Dataset,
        config: DataConfig,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        collate_fn: Optional[Callable] = None,
    ):
        """
        Args:
            dataset: PyTorch dataset
            config: DataLoader configuration
            rank: Process rank for distributed training
            world_size: Total number of processes
            seed: Random seed for reproducibility
            collate_fn: Custom collation function
        """
        self.dataset = dataset
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0

        # Create distributed sampler if needed
        if world_size > 1:
            self.sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=config.shuffle,
                seed=seed,
                drop_last=config.drop_last,
            )
        else:
            self.sampler = None

        # Default collate function
        if collate_fn is None:
            collate_fn = self._default_collate

        # Create dataloader
        self.dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=self.sampler,
            shuffle=(config.shuffle and self.sampler is None),
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=config.drop_last,
            collate_fn=collate_fn,
            prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
            persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
        )

    def _default_collate(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Default collation with padding."""
        # Get max length in batch
        max_len = max(item['input_ids'].size(0) for item in batch)

        # Pad to max length
        input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

        for i, item in enumerate(batch):
            seq_len = item['input_ids'].size(0)
            input_ids[i, :seq_len] = item['input_ids']
            attention_mask[i, :seq_len] = 1

            if 'labels' in item:
                labels[i, :seq_len] = item['labels']
            else:
                labels[i, :seq_len] = item['input_ids']

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over batches."""
        if self.sampler is not None:
            self.sampler.set_epoch(self.epoch)
        return iter(self.dataloader)

    def __len__(self) -> int:
        """Number of batches."""
        return len(self.dataloader)

    def set_epoch(self, epoch: int):
        """Set epoch for distributed sampler."""
        self.epoch = epoch
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)

    @property
    def batch_size(self) -> int:
        """Effective batch size (per GPU)."""
        return self.config.batch_size

    @property
    def global_batch_size(self) -> int:
        """Global batch size across all GPUs."""
        return self.config.batch_size * self.world_size


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    seq_length: int = 2048,
    num_workers: int = 4,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    **kwargs,
) -> DistributedDataLoader:
    """
    Create a distributed dataloader.

    Args:
        dataset: Dataset to load from
        batch_size: Batch size per GPU
        seq_length: Maximum sequence length
        num_workers: Number of data loading workers
        distributed: Whether using distributed training
        rank: Process rank
        world_size: Total processes

    Returns:
        Configured DistributedDataLoader
    """
    config = DataConfig(
        batch_size=batch_size,
        seq_length=seq_length,
        num_workers=num_workers,
        **kwargs,
    )

    if distributed and world_size == 1:
        import torch.distributed as dist
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()

    return DistributedDataLoader(
        dataset=dataset,
        config=config,
        rank=rank,
        world_size=world_size,
    )


class InfiniteDataLoader:
    """
    DataLoader that loops infinitely.

    Useful for training with step-based (not epoch-based) schedules.
    """

    def __init__(self, dataloader: DistributedDataLoader):
        self.dataloader = dataloader
        self.iterator = None
        self.epoch = 0

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        if self.iterator is None:
            self.dataloader.set_epoch(self.epoch)
            self.iterator = iter(self.dataloader)

        try:
            return next(self.iterator)
        except StopIteration:
            self.epoch += 1
            self.dataloader.set_epoch(self.epoch)
            self.iterator = iter(self.dataloader)
            return next(self.iterator)
