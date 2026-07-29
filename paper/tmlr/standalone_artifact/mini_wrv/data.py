"""Minimal deterministic data pipeline matching the reported FineWeb protocol."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import torch
from torch.utils.data import IterableDataset


class FineWebTokenStream(IterableDataset[torch.Tensor]):
    """Tokenize a parquet/streaming document iterator with the 1-in-20 eval split."""

    def __init__(
        self,
        documents: Iterable[dict[str, Any]],
        tokenizer: Any,
        sequence_length: int,
        *,
        split: str,
        eval_stride: int = 20,
    ) -> None:
        if split not in {"train", "eval"}:
            raise ValueError("split must be 'train' or 'eval'")
        self.documents = documents
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.split = split
        self.eval_stride = eval_stride

    def __iter__(self) -> Iterator[torch.Tensor]:
        buffer: list[int] = []
        for index, example in enumerate(self.documents):
            is_eval = index % self.eval_stride == 0
            if (self.split == "eval") != is_eval:
                continue
            text = example.get("text", "")
            if not text:
                continue
            buffer.extend(self.tokenizer.encode(text))
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_token_id is not None:
                buffer.append(int(eos_token_id))
            while len(buffer) >= self.sequence_length + 1:
                yield torch.tensor(buffer[: self.sequence_length + 1], dtype=torch.long)
                buffer = buffer[self.sequence_length :]


def load_fineweb_parquet(path: str) -> Iterable[dict[str, Any]]:
    """Load the pinned local parquet shard lazily through Hugging Face datasets."""
    from datasets import load_dataset

    return load_dataset(
        "parquet",
        data_files={"train": path},
        split="train",
        streaming=True,
    )


class TiktokenO200k:
    """Small tokenizer adapter for o200k_base-compatible experiments."""

    def __init__(self) -> None:
        import tiktoken

        self.encoding = tiktoken.get_encoding("o200k_base")
        self.eos_token_id = self.encoding.eot_token

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode_ordinary(text)
