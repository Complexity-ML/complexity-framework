"""
Streaming Datasets for Large-Scale Training.

Handles datasets that don't fit in memory:
- Streaming from disk
- Sharded datasets across files
- Interleaved multi-source datasets
- Resume from checkpoint
"""

import torch
from torch.utils.data import IterableDataset, Dataset
from typing import Optional, Dict, Any, Iterator, List, Callable, Union
from pathlib import Path
import json
import random
import os


class StreamingDataset(IterableDataset):
    """
    Streaming dataset for large text corpora.

    Reads data lazily from disk, supporting:
    - JSONL files
    - Multiple file formats
    - On-the-fly tokenization
    - Sequence packing

    Example:
        dataset = StreamingDataset(
            data_path="train.jsonl",
            tokenizer=tokenizer,
            seq_length=2048,
        )
    """

    def __init__(
        self,
        data_path: Union[str, List[str]],
        tokenizer: Any,
        seq_length: int = 2048,
        text_field: str = "text",
        shuffle_buffer: int = 10000,
        seed: int = 42,
        pack_sequences: bool = True,
    ):
        """
        Args:
            data_path: Path(s) to data file(s)
            tokenizer: Tokenizer with encode() method
            seq_length: Maximum sequence length
            text_field: Field name containing text in JSONL
            shuffle_buffer: Buffer size for shuffling
            seed: Random seed
            pack_sequences: Whether to pack multiple sequences
        """
        if isinstance(data_path, str):
            data_path = [data_path]

        self.data_paths = [Path(p) for p in data_path]
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.text_field = text_field
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.pack_sequences = pack_sequences

        # Validate paths exist
        for path in self.data_paths:
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {path}")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over tokenized sequences."""
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            # Split files across workers
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            paths = self.data_paths[worker_id::num_workers]
        else:
            paths = self.data_paths

        # Create generator
        if self.pack_sequences:
            yield from self._packed_iterator(paths)
        else:
            yield from self._simple_iterator(paths)

    def _simple_iterator(self, paths: List[Path]) -> Iterator[Dict[str, torch.Tensor]]:
        """Simple iteration without packing."""
        buffer = []
        rng = random.Random(self.seed)

        for path in paths:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        text = data.get(self.text_field, "")

                        if not text:
                            continue

                        # Tokenize
                        tokens = self.tokenizer.encode(text)

                        # Truncate if needed
                        if len(tokens) > self.seq_length:
                            tokens = tokens[:self.seq_length]

                        buffer.append({
                            'input_ids': torch.tensor(tokens, dtype=torch.long),
                        })

                        # Yield from shuffle buffer
                        if len(buffer) >= self.shuffle_buffer:
                            rng.shuffle(buffer)
                            yield from buffer
                            buffer = []

                    except json.JSONDecodeError:
                        continue

        # Yield remaining
        if buffer:
            rng.shuffle(buffer)
            yield from buffer

    def _packed_iterator(self, paths: List[Path]) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Packed sequence iteration.

        Concatenates multiple documents into single sequences
        to maximize GPU utilization.
        """
        token_buffer = []
        buffer = []
        rng = random.Random(self.seed)

        # EOS token for separating documents
        eos_token = self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else 2

        for path in paths:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        text = data.get(self.text_field, "")

                        if not text:
                            continue

                        # Tokenize
                        tokens = self.tokenizer.encode(text)
                        tokens.append(eos_token)

                        # Add to buffer
                        token_buffer.extend(tokens)

                        # Create packed sequences
                        while len(token_buffer) >= self.seq_length:
                            seq = token_buffer[:self.seq_length]
                            token_buffer = token_buffer[self.seq_length:]

                            buffer.append({
                                'input_ids': torch.tensor(seq, dtype=torch.long),
                            })

                            # Yield from shuffle buffer
                            if len(buffer) >= self.shuffle_buffer:
                                rng.shuffle(buffer)
                                yield from buffer
                                buffer = []

                    except json.JSONDecodeError:
                        continue

        # Yield remaining full sequences
        if buffer:
            rng.shuffle(buffer)
            yield from buffer


class ShardedDataset(IterableDataset):
    """
    Dataset sharded across multiple files.

    Useful for very large datasets split into chunks.
    Supports automatic worker assignment and distributed training.
    """

    def __init__(
        self,
        shard_pattern: str,
        tokenizer: Any,
        seq_length: int = 2048,
        text_field: str = "text",
        shuffle_shards: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            shard_pattern: Glob pattern for shard files (e.g., "data/shard_*.jsonl")
            tokenizer: Tokenizer
            seq_length: Maximum sequence length
            text_field: Text field name
            shuffle_shards: Whether to shuffle shard order
            seed: Random seed
        """
        from glob import glob

        self.shard_paths = sorted(glob(shard_pattern))
        if not self.shard_paths:
            raise ValueError(f"No files found matching: {shard_pattern}")

        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.text_field = text_field
        self.shuffle_shards = shuffle_shards
        self.seed = seed

        print(f"Found {len(self.shard_paths)} shards")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over all shards."""
        worker_info = torch.utils.data.get_worker_info()

        # Determine which shards this worker handles
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            shards = self.shard_paths[worker_id::num_workers]
        else:
            shards = self.shard_paths

        # Optionally shuffle shard order
        if self.shuffle_shards:
            rng = random.Random(self.seed)
            shards = shards.copy()
            rng.shuffle(shards)

        # Create streaming dataset for shards
        for shard_path in shards:
            shard_dataset = StreamingDataset(
                data_path=shard_path,
                tokenizer=self.tokenizer,
                seq_length=self.seq_length,
                text_field=self.text_field,
                shuffle_buffer=1000,
                seed=self.seed,
            )
            yield from shard_dataset


class InterleavedDataset(IterableDataset):
    """
    Interleaved multi-source dataset.

    Mixes data from multiple sources according to specified weights.
    Useful for combining different training corpora.
    """

    def __init__(
        self,
        datasets: List[IterableDataset],
        weights: Optional[List[float]] = None,
        seed: int = 42,
        stopping_strategy: str = "all_exhausted",  # or "first_exhausted"
    ):
        """
        Args:
            datasets: List of datasets to interleave
            weights: Sampling weights (default: uniform)
            seed: Random seed
            stopping_strategy: When to stop ("all_exhausted" or "first_exhausted")
        """
        self.datasets = datasets

        if weights is None:
            weights = [1.0] * len(datasets)

        # Normalize weights
        total = sum(weights)
        self.weights = [w / total for w in weights]

        self.seed = seed
        self.stopping_strategy = stopping_strategy

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Interleaved iteration."""
        rng = random.Random(self.seed)

        # Create iterators
        iterators = [iter(ds) for ds in self.datasets]
        exhausted = [False] * len(self.datasets)

        while True:
            # Check stopping condition
            if self.stopping_strategy == "first_exhausted":
                if any(exhausted):
                    break
            else:  # all_exhausted
                if all(exhausted):
                    break

            # Sample from datasets according to weights
            active_indices = [i for i, e in enumerate(exhausted) if not e]
            if not active_indices:
                break

            active_weights = [self.weights[i] for i in active_indices]
            total_weight = sum(active_weights)
            probs = [w / total_weight for w in active_weights]

            # Sample index
            r = rng.random()
            cumsum = 0
            chosen_idx = active_indices[-1]
            for idx, p in zip(active_indices, probs):
                cumsum += p
                if r < cumsum:
                    chosen_idx = idx
                    break

            # Get next item from chosen dataset
            try:
                item = next(iterators[chosen_idx])
                yield item
            except StopIteration:
                exhausted[chosen_idx] = True

                # Restart if not stopping
                if self.stopping_strategy == "all_exhausted":
                    iterators[chosen_idx] = iter(self.datasets[chosen_idx])


class CachedDataset(Dataset):
    """
    Cached dataset for smaller datasets that fit in memory.

    Pre-tokenizes and caches all examples for fast access.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        seq_length: int = 2048,
        text_field: str = "text",
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            data_path: Path to data file
            tokenizer: Tokenizer
            seq_length: Maximum sequence length
            text_field: Text field name
            cache_dir: Directory to cache processed data
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.data = []

        # Check for cached version
        if cache_dir:
            cache_path = Path(cache_dir) / f"{Path(data_path).stem}_cached.pt"
            if cache_path.exists():
                print(f"Loading cached dataset from {cache_path}")
                self.data = torch.load(cache_path)
                return

        # Process data
        print(f"Processing dataset: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    text = item.get(text_field, "")

                    if not text:
                        continue

                    tokens = tokenizer.encode(text)
                    if len(tokens) > seq_length:
                        tokens = tokens[:seq_length]

                    self.data.append({
                        'input_ids': torch.tensor(tokens, dtype=torch.long),
                    })
                except json.JSONDecodeError:
                    continue

        # Cache processed data
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            torch.save(self.data, cache_path)
            print(f"Cached dataset to {cache_path}")

        print(f"Loaded {len(self.data)} examples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]
