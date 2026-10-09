"""Prepare the uint16 corpus-mixture format consumed by the training loader.

These reusable primitives preserve the shard/manifest contract previously
implemented in the removed 200M tokenization script. They have no fixed corpus,
training budget or implicit upload destination.
"""

from __future__ import annotations

import hashlib
import json
import logging
import operator
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .corpus_mixture import TextCorpusSource, allocate_weighted_counts, validate_corpus_mixture

logger = logging.getLogger(__name__)
FORMAT = "tr-hash-token-mixture-v1"


def _positive_integer(value, name: str) -> int:
    try:
        integer = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if isinstance(value, bool) or integer < 1:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TokenShardWriter:
    """Write one continuous token stream into independently mmap-able shards."""

    def __init__(
        self,
        root: Path,
        *,
        seq_len: int,
        total_rows: int,
        rows_per_shard: int,
    ) -> None:
        self.seq_len = _positive_integer(seq_len, "seq_len")
        self.total_rows = _positive_integer(total_rows, "total_rows")
        self.rows_per_shard = _positive_integer(rows_per_shard, "rows_per_shard")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if any(self.root.iterdir()):
            raise FileExistsError(
                f"partial output already exists for {self.root.name}: {self.root}. "
                "Keep completed corpora, but clear this incomplete corpus directory before retrying."
            )
        self.rows_written = 0
        self.source_tokens_written = 0
        self.shards: list[dict[str, int | str]] = []
        self._mapping: np.memmap | None = None
        self._position = 0
        self._current_rows = 0
        self._partial_path: Path | None = None
        self._final_path: Path | None = None
        self._carry_token: int | None = None

    @property
    def complete(self) -> bool:
        return self.rows_written == self.total_rows and self._mapping is None

    @property
    def required_source_tokens(self) -> int:
        return self.total_rows * self.seq_len + 1

    def _open_shard(self) -> None:
        remaining_rows = self.total_rows - self.rows_written
        if remaining_rows <= 0:
            return
        self._current_rows = min(self.rows_per_shard, remaining_rows)
        index = len(self.shards)
        self._final_path = self.root / f"tokens-{index:05d}.bin"
        self._partial_path = self.root / f"tokens-{index:05d}.bin.partial"
        token_count = self._current_rows * self.seq_len + 1
        self._mapping = np.memmap(
            self._partial_path,
            mode="w+",
            dtype=np.uint16,
            shape=(token_count,),
        )
        self._position = 0
        if self._carry_token is not None:
            self._mapping[0] = self._carry_token
            self._position = 1

    def _close_full_shard(self) -> None:
        if self._mapping is None or self._position != self._mapping.size:
            raise RuntimeError("cannot finalize an incomplete token shard")
        self._carry_token = int(self._mapping[-1])
        token_count = int(self._mapping.size)
        self._mapping.flush()
        del self._mapping
        self._mapping = None
        assert self._partial_path is not None and self._final_path is not None
        os.replace(self._partial_path, self._final_path)
        sha256 = _sha256_file(self._final_path)
        self.shards.append(
            {
                "file": self._final_path.name,
                "rows": self._current_rows,
                "tokens": token_count,
                "bytes": self._final_path.stat().st_size,
                "sha256": sha256,
            }
        )
        self.rows_written += self._current_rows
        self._position = 0

    def feed(self, tokens: np.ndarray) -> int:
        """Consume new source tokens; duplicated shard-boundary tokens are internal."""

        tokens = np.asarray(tokens)
        if tokens.ndim != 1 or tokens.dtype.kind not in {"i", "u"}:
            raise ValueError("token ids must be a flat integer array")
        if tokens.size and (tokens.min() < 0 or tokens.max() > 65535):
            raise ValueError("token ids must fit uint16 without wrapping")
        if tokens.dtype != np.uint16:
            tokens = tokens.astype(np.uint16, copy=False)
        consumed = 0
        while consumed < tokens.size and self.rows_written < self.total_rows:
            if self._mapping is None:
                self._open_shard()
            assert self._mapping is not None
            available = self._mapping.size - self._position
            take = min(available, tokens.size - consumed)
            self._mapping[self._position : self._position + take] = tokens[
                consumed : consumed + take
            ]
            self._position += take
            consumed += take
            self.source_tokens_written += take
            if self._position == self._mapping.size:
                self._close_full_shard()
        return consumed

    def write_manifest(self, *, source: TextCorpusSource) -> Path:
        if not self.complete:
            raise RuntimeError(
                f"source {source.name} incomplete: {self.rows_written}/{self.total_rows} rows"
            )
        manifest = {
            "format": FORMAT,
            "source": source.name,
            "weight": source.weight,
            "seq_len": self.seq_len,
            "dtype": "uint16",
            "rows": self.total_rows,
            "trained_tokens": self.total_rows * self.seq_len,
            "source_tokens_consumed": self.source_tokens_written,
            "shards": self.shards,
        }
        path = self.root / "manifest.json"
        temporary = path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        return path


def resolve_layout(
    *,
    target_tokens: int,
    seq_len: int,
    global_batch_sequences: int,
    sources: Sequence[TextCorpusSource],
) -> tuple[int, int, dict[str, int]]:
    """Round once to complete optimizer updates, then allocate exact quotas."""

    target_tokens = _positive_integer(target_tokens, "target_tokens")
    seq_len = _positive_integer(seq_len, "seq_len")
    global_batch_sequences = _positive_integer(global_batch_sequences, "global_batch_sequences")
    tokens_per_update = seq_len * global_batch_sequences
    total_rows = (target_tokens + tokens_per_update - 1) // tokens_per_update
    total_rows *= global_batch_sequences
    rows_by_source = allocate_weighted_counts(total_rows, sources)
    return total_rows, total_rows * seq_len, rows_by_source


def write_mixture_manifest(
    *,
    output_root: Path,
    sources: Sequence[TextCorpusSource],
    seq_len: int,
    requested_tokens: int,
    actual_tokens: int,
    global_batch_sequences: int,
    rows_by_source: Mapping[str, int],
) -> Path:
    entries = []
    for source in sources:
        relative = Path("corpora") / source.name / "manifest.json"
        path = output_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"corpus manifest missing: {path}")
        entries.append(
            {
                "name": source.name,
                "weight": source.weight,
                "rows": rows_by_source[source.name],
                "trained_tokens": rows_by_source[source.name] * seq_len,
                "manifest": str(relative),
            }
        )
    manifest = {
        "format": FORMAT,
        "dtype": "uint16",
        "seq_len": seq_len,
        "requested_tokens": requested_tokens,
        "actual_tokens": actual_tokens,
        "global_batch_sequences": global_batch_sequences,
        "sources": entries,
    }
    path = output_root / "mixture_manifest.json"
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def upload_dataset_subset(
    *,
    output_root: Path,
    repo_id: str,
    allow_patterns: Sequence[str],
    token: str | None,
    workers: int,
    api=None,
) -> None:
    """Upload one completed subset with the Hub's resumable large-folder path."""

    workers = _positive_integer(workers, "Hugging Face upload workers")
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
    logger.info(
        "Uploading %s to hf://datasets/%s",
        ", ".join(allow_patterns),
        repo_id,
    )
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=output_root,
        allow_patterns=list(allow_patterns),
        num_workers=workers,
        print_report=True,
        print_report_every=60,
    )


def prepare_corpus_mixture(
    *,
    output_root: str | Path,
    tokenizer,
    sources: Sequence[TextCorpusSource],
    target_tokens: int,
    seq_len: int,
    global_batch_sequences: int,
    rows_per_shard: int,
    streams: Mapping[str, Iterable[Mapping]] | None = None,
) -> Path:
    """Tokenize complete source streams into a quota-controlled local mixture.

    Completed outputs are never overwritten. An exhausted source leaves its
    partial output for inspection and fails before the mixture is published.
    Hub uploads are a separate explicit operation.
    """
    validate_corpus_mixture(sources)
    rows_per_shard = _positive_integer(rows_per_shard, "rows_per_shard")
    for source in sources:
        if Path(source.name).name != source.name or source.name in {".", ".."}:
            raise ValueError("corpus names must be single safe path components")
    _, actual_tokens, quotas = resolve_layout(
        target_tokens=target_tokens,
        seq_len=seq_len,
        global_batch_sequences=global_batch_sequences,
        sources=sources,
    )
    if any(rows < 1 for rows in quotas.values()):
        raise ValueError("token budget is too small to allocate a row to every source")
    root = Path(output_root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"output directory is not empty: {root}")
    if streams is not None and set(streams) != {source.name for source in sources}:
        raise ValueError("provided streams must name every corpus exactly once")
    eos = getattr(tokenizer, "eos_token_id", None)
    for source in sources:
        if streams is None:
            from datasets import load_dataset

            stream = (
                load_dataset(
                    "json", data_files=source.data_files, split=source.split, streaming=True
                )
                if source.data_files is not None
                else load_dataset(
                    source.dataset_id, source.config_name, split=source.split, streaming=True
                )
            )
        else:
            stream = streams[source.name]
        writer = TokenShardWriter(
            root / "corpora" / source.name,
            seq_len=seq_len,
            total_rows=quotas[source.name],
            rows_per_shard=rows_per_shard,
        )
        try:
            for example in stream:
                text = example.get(source.text_field, "")
                if not isinstance(text, str) or not text:
                    continue
                tokens = list(tokenizer.encode(text, add_special_tokens=False))
                if eos is not None:
                    tokens.append(eos)
                writer.feed(np.asarray(tokens, dtype=np.int64))
                if writer.complete:
                    break
            writer.write_manifest(source=source)
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()
    return write_mixture_manifest(
        output_root=root,
        sources=sources,
        seq_len=seq_len,
        requested_tokens=target_tokens,
        actual_tokens=actual_tokens,
        global_batch_sequences=global_batch_sequences,
        rows_by_source=quotas,
    )
