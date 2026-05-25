"""Memory-mapped token shards for from-scratch pretraining.

The format is intentionally small:

```
dataset/
  tokens.bin       # flat little-endian uint16 or uint32 token ids
  tokens.idx.json  # metadata: dtype, num_tokens, vocab_size, tokenizer, ...
```

It gives the training runner a fast, already-tokenized path without tying the
framework to a specific external data format.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset


TOKEN_BIN = "tokens.bin"
TOKEN_INDEX = "tokens.idx.json"


def _dtype_for_vocab(vocab_size: int | None) -> np.dtype:
    if vocab_size is not None and vocab_size <= np.iinfo(np.uint16).max:
        return np.dtype("<u2")
    return np.dtype("<u4")


def _normalize_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        return path
    raise ValueError(f"Token shard path must be a directory: {path}")


def _iter_token_arrays(tokens: Iterable[int] | Iterable[Iterable[int]], dtype: np.dtype) -> Iterator[np.ndarray]:
    """Yield 1D numpy arrays from a flat or chunked token iterable."""

    for item in tokens:
        if isinstance(item, (list, tuple, np.ndarray)):
            arr = np.asarray(item, dtype=dtype)
        else:
            arr = np.asarray([item], dtype=dtype)
        if arr.ndim != 1:
            raise ValueError("token chunks must be flat")
        if arr.size:
            yield arr


def _file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def write_token_shard(
    output_dir: str | Path,
    tokens: Iterable[int] | Iterable[Iterable[int]],
    *,
    vocab_size: int | None = None,
    tokenizer: str | None = None,
    dtype: str | np.dtype | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a flat token stream to ``output_dir``.

    Args:
        output_dir: Destination directory.
        tokens: Iterable of token ids.
        vocab_size: Optional vocab size, used for dtype selection and metadata.
        tokenizer: Optional tokenizer identifier/path for reproducibility.
        dtype: Optional numpy dtype override. Defaults to uint16 when possible,
            otherwise uint32.
        extra_metadata: Additional JSON-serializable metadata.

    Returns:
        Path to the written index file.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np_dtype = np.dtype(dtype) if dtype is not None else _dtype_for_vocab(vocab_size)
    bin_path = out / TOKEN_BIN
    num_tokens = 0
    max_token_id = -1
    with bin_path.open("wb") as f:
        for arr in _iter_token_arrays(tokens, np_dtype):
            if arr.size:
                max_token_id = max(max_token_id, int(arr.max()))
            arr.tofile(f)
            num_tokens += int(arr.size)
    if num_tokens == 0:
        raise ValueError("cannot write an empty token shard")
    metadata = {
        "format": "complexity-token-shard-v1",
        "bin": TOKEN_BIN,
        "dtype": np_dtype.str,
        "num_tokens": num_tokens,
        "max_token_id": max_token_id,
        "vocab_size": vocab_size,
        "tokenizer": tokenizer,
        "sha256": _file_sha256(bin_path),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    index_path = out / TOKEN_INDEX
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")
    return index_path


def iter_texts_from_files(
    paths: Iterable[str | Path],
    *,
    input_format: str = "jsonl",
    text_field: str = "text",
    limit: int = 0,
) -> Iterator[str]:
    """Yield text records from JSONL or plain text files."""

    seen = 0
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as f:
            if input_format == "jsonl":
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    text = row.get(text_field, "")
                    if text:
                        yield str(text)
                        seen += 1
                    if limit > 0 and seen >= limit:
                        return
            elif input_format == "text":
                text = f.read()
                if text:
                    yield text
                    seen += 1
                if limit > 0 and seen >= limit:
                    return
            else:
                raise ValueError("input_format must be one of: jsonl, text")


def build_token_shard_from_texts(
    texts: Iterable[str],
    tokenizer: Any,
    output_dir: str | Path,
    *,
    vocab_size: int | None = None,
    tokenizer_name: str | None = None,
    add_eos: bool = True,
    dtype: str | np.dtype | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Tokenize text records and write a streaming token shard."""

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        eos_id = getattr(getattr(tokenizer, "_config", None), "eos_token_id", None)
    if eos_id is None and hasattr(tokenizer, "_get_special_id"):
        eos_id = tokenizer._get_special_id("eos")

    def token_chunks() -> Iterator[list[int]]:
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if add_eos and eos_id is not None:
                ids = list(ids) + [int(eos_id)]
            yield ids

    return write_token_shard(
        output_dir,
        token_chunks(),
        vocab_size=vocab_size,
        tokenizer=tokenizer_name,
        dtype=dtype,
        extra_metadata=extra_metadata,
    )


def load_token_shard(path: str | Path, *, mmap_mode: str = "r") -> tuple[np.memmap, dict[str, Any]]:
    """Load a token shard as a numpy memmap plus metadata."""

    root = _normalize_path(path)
    index_path = root / TOKEN_INDEX
    if not index_path.exists():
        raise FileNotFoundError(f"Token shard index not found: {index_path}")
    with index_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    if metadata.get("format") != "complexity-token-shard-v1":
        raise ValueError(f"Unsupported token shard format: {metadata.get('format')!r}")
    dtype = np.dtype(metadata["dtype"])
    bin_path = root / metadata.get("bin", TOKEN_BIN)
    if not bin_path.exists():
        raise FileNotFoundError(f"Token shard data not found: {bin_path}")
    tokens = np.memmap(bin_path, dtype=dtype, mode=mmap_mode, shape=(int(metadata["num_tokens"]),))
    return tokens, metadata


def token_shard_frequencies(path: str | Path, vocab_size: int) -> torch.Tensor:
    """Count token frequencies from a memory-mapped token shard."""

    tokens, _ = load_token_shard(path)
    freqs = torch.zeros(vocab_size, dtype=torch.float32)
    chunk_size = 8_000_000
    for start in range(0, int(tokens.shape[0]), chunk_size):
        chunk = np.asarray(tokens[start:start + chunk_size], dtype=np.int64)
        valid = chunk[(chunk >= 0) & (chunk < vocab_size)]
        if valid.size == 0:
            continue
        ids = torch.from_numpy(valid)
        freqs.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))
    return freqs


class TokenShardDataset(IterableDataset):
    """Infinite dataset of fixed-length LM chunks from a memmapped token shard."""

    def __init__(
        self,
        path: str | Path,
        seq_len: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        split: str = "train",
        eval_ratio: float = 0.001,
    ):
        self.path = str(path)
        self.seq_len = int(seq_len)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.split = split
        self.eval_ratio = float(eval_ratio)

        tokens, metadata = load_token_shard(path)
        self.num_tokens = int(tokens.shape[0])
        self.metadata = metadata
        if self.num_tokens < self.seq_len + 2:
            raise ValueError(f"Need at least {self.seq_len + 2} tokens, got {self.num_tokens}")

        eval_tokens = max(self.seq_len + 1, int(self.num_tokens * self.eval_ratio))
        eval_tokens = min(eval_tokens, max(self.seq_len + 1, self.num_tokens // 10))
        if split == "train":
            self.start = 0
            self.end = self.num_tokens - eval_tokens
        elif split in {"eval", "val", "validation"}:
            self.start = self.num_tokens - eval_tokens
            self.end = self.num_tokens
        else:
            raise ValueError("split must be one of: train, eval, val, validation")
        if self.end - self.start < self.seq_len + 1:
            raise ValueError(f"Token shard split {split!r} is too small for seq_len={self.seq_len}")

    def __iter__(self):
        tokens, _ = load_token_shard(self.path)
        worker = torch.utils.data.get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        stream_id = self.rank * num_workers + worker_id
        stream_count = max(1, self.world_size * num_workers)
        gen = torch.Generator().manual_seed(self.seed + 1_000_003 * stream_id)

        high = self.end - self.seq_len - 1
        while True:
            # Shard the start positions across rank/worker streams while keeping
            # randomness. This avoids identical batches between workers.
            span = max(1, (high - self.start + 1) // stream_count)
            base = self.start + stream_id * span
            limit = min(high, base + span - 1)
            if base > high:
                base = self.start
                limit = high
            offset = torch.randint(0, limit - base + 1, (1,), generator=gen).item()
            pos = base + offset
            chunk = np.asarray(tokens[pos:pos + self.seq_len + 1], dtype=np.int64)
            ids = torch.from_numpy(chunk.copy()).long()
            yield {"input_ids": ids[:-1], "labels": ids[1:]}
