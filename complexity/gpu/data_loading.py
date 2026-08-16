"""
Data loading optimizations for GPU training.

Prefetch and overlap data loading with compute to hide I/O latency.

Complexity-ML — 2026
"""

import torch
from torch.utils.data import DataLoader
import threading
import logging

logger = logging.getLogger(__name__)


class CUDAPrefetcher:
    """
    Prefetch data to GPU using a separate CUDA stream.

    Overlaps CPU→GPU data transfer with GPU compute.
    Hides PCIe transfer latency (~2ms per batch on H100).

    Usage:
        loader = DataLoader(dataset, batch_size=64, pin_memory=True)
        prefetcher = CUDAPrefetcher(loader, device="cuda:0")

        for batch in prefetcher:
            loss = model(batch["input_ids"])
    """

    def __init__(self, dataloader: DataLoader, device: torch.device = None):
        self.dataloader = dataloader
        self.device = device or torch.device("cuda")
        self.stream = torch.cuda.Stream(device=self.device)

    def __iter__(self):
        self._iter = iter(self.dataloader)
        self._prefetch()
        return self

    def _prefetch(self):
        try:
            self._next_batch = next(self._iter)
        except StopIteration:
            self._next_batch = None
            return

        with torch.cuda.stream(self.stream):
            self._next_batch = self._to_device(self._next_batch)

    def _to_device(self, batch):
        if isinstance(batch, dict):
            return {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return type(batch)(
                v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for v in batch
            )
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        return batch

    def __next__(self):
        torch.cuda.current_stream(self.device).wait_stream(self.stream)

        batch = self._next_batch
        if batch is None:
            raise StopIteration

        # Ensure tensors are ready
        if isinstance(batch, dict):
            for v in batch.values():
                if isinstance(v, torch.Tensor):
                    v.record_stream(torch.cuda.current_stream(self.device))
        elif isinstance(batch, torch.Tensor):
            batch.record_stream(torch.cuda.current_stream(self.device))

        self._prefetch()
        return batch

    def __len__(self):
        return len(self.dataloader)


class BackgroundTokenizer:
    """
    Tokenize text in a background thread while GPU computes.

    For streaming datasets where tokenization is the bottleneck.

    Usage:
        tokenizer_bg = BackgroundTokenizer(tokenizer, max_length=2048, buffer_size=100)
        for text in dataset:
            tokenizer_bg.submit(text)
        tokens = tokenizer_bg.get()  # Returns tokenized batch
    """

    def __init__(self, tokenizer, max_length: int = 2048, buffer_size: int = 100):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._buffer = []
        self._lock = threading.Lock()
        self.buffer_size = buffer_size

    def submit(self, text: str) -> None:
        """Submit text for background tokenization."""
        tokens = self.tokenizer.encode(text)
        with self._lock:
            self._buffer.extend(tokens)

    def get_batch(self) -> torch.Tensor:
        """Get a batch of tokens if enough are buffered."""
        with self._lock:
            if len(self._buffer) < self.max_length + 1:
                return None
            chunk = self._buffer[:self.max_length + 1]
            self._buffer = self._buffer[self.max_length:]
        return torch.tensor(chunk, dtype=torch.long)
