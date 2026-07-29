#!/usr/bin/env python3
"""Freeze FineWeb-Edu into disjoint o200k train/eval token shards.

The script is intended to run on the training server before ``torchrun``. It
downloads one pinned Parquet file at a time to local NVMe, tokenizes it with the
repository's cached o200k tokenizer, appends EOS between documents, writes
memory-mapped uint32 token streams, and deletes the raw Parquet file. Training
therefore performs no network I/O.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pyarrow.parquet as pq
import requests
from huggingface_hub import HfApi, hf_hub_url

from complexity.tokenizer import Tokenizer

REPO_ID = "HuggingFaceFW/fineweb-edu"
DATASET_PREFIX = "sample/10BT/"
# This is the FineWeb-Edu revision cached before the historical 306.5M runs.
# Keeping it explicit prevents a later dataset update from silently changing
# the new matched pair.
DEFAULT_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
FORMAT = "complexity-token-shard-v1"
DTYPE = np.dtype("<u4")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("./tokenizer-o200k"))
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--train-tokens",
        type=int,
        default=3_999_793_153,
        help="Stored train tokens. Default is 7,629 x 4 x 64 x 2,048 targets plus one.",
    )
    parser.add_argument(
        "--eval-tokens",
        type=int,
        default=16_777_217,
        help="Stored held-out tokens. Default covers 32 eval batches on 4 GPUs plus one.",
    )
    parser.add_argument(
        "--eval-stride",
        type=int,
        default=200,
        help="Every Nth source document is reserved for evaluation.",
    )
    parser.add_argument("--document-batch-size", type=int, default=1024)
    parser.add_argument(
        "--tokenizer-threads",
        type=int,
        default=min(96, os.cpu_count() or 8),
        help="CPU threads used by tiktoken for each document batch.",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> tuple[str, int]:
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.unlink(missing_ok=True)
    digest = hashlib.sha256()
    total_bytes = 0
    started = time.monotonic()
    last_report = started
    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        expected_bytes = int(response.headers.get("content-length") or 0)
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                digest.update(chunk)
                total_bytes += len(chunk)
                now = time.monotonic()
                if now - last_report >= 5.0:
                    elapsed = max(now - started, 1e-9)
                    rate_mib = total_bytes / elapsed / 1024**2
                    progress = (
                        f" · {100.0 * total_bytes / expected_bytes:.1f}%"
                        if expected_bytes > 0
                        else ""
                    )
                    print(
                        f"  download {destination.name}: "
                        f"{total_bytes / 1e9:.2f} GB{progress} · {rate_mib:.1f} MiB/s",
                        flush=True,
                    )
                    last_report = now
            handle.flush()
            os.fsync(handle.fileno())
    partial.replace(destination)
    elapsed = max(time.monotonic() - started, 1e-9)
    print(
        f"  downloaded {destination.name}: {total_bytes / 1e9:.2f} GB "
        f"in {elapsed:.1f}s · {total_bytes / elapsed / 1024**2:.1f} MiB/s",
        flush=True,
    )
    return digest.hexdigest(), total_bytes


@dataclass
class TokenWriter:
    name: str
    root: Path
    target_tokens: int
    vocab_size: int
    tokenizer_label: str
    tokenizer_sha256: str

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.partial_path = self.root / "tokens.bin.partial"
        self.final_path = self.root / "tokens.bin"
        self.handle = self.partial_path.open("wb")
        self.digest = hashlib.sha256()
        self.num_tokens = 0
        self.max_token_id = -1
        self.documents = 0

    @property
    def full(self) -> bool:
        return self.num_tokens >= self.target_tokens

    def append(self, token_ids: Iterable[int]) -> None:
        if self.full:
            return
        remaining = self.target_tokens - self.num_tokens
        array64 = np.asarray(list(token_ids), dtype=np.int64)
        if array64.size == 0:
            return
        array64 = array64[:remaining]
        minimum = int(array64.min())
        maximum = int(array64.max())
        if minimum < 0 or maximum >= self.vocab_size:
            raise ValueError(
                f"{self.name}: token id outside vocabulary: min={minimum}, max={maximum}, "
                f"vocab={self.vocab_size}"
            )
        if maximum > np.iinfo(np.uint32).max:
            raise ValueError(f"{self.name}: token id {maximum} does not fit uint32")
        array = array64.astype(DTYPE, copy=False)
        payload = array.tobytes()
        self.handle.write(payload)
        self.digest.update(payload)
        self.num_tokens += int(array.size)
        self.max_token_id = max(self.max_token_id, maximum)
        self.documents += 1

    def finish(self, common_metadata: dict) -> dict:
        if not self.full:
            raise RuntimeError(
                f"{self.name} ended at {self.num_tokens:,}/{self.target_tokens:,} tokens"
            )
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        self.partial_path.replace(self.final_path)
        metadata = {
            "format": FORMAT,
            "bin": "tokens.bin",
            "dtype": DTYPE.str,
            "num_tokens": self.num_tokens,
            "max_token_id": self.max_token_id,
            "vocab_size": self.vocab_size,
            "tokenizer": self.tokenizer_label,
            "tokenizer_sha256": self.tokenizer_sha256,
            "sha256": self.digest.hexdigest(),
            "partition": self.name,
            "documents": self.documents,
            **common_metadata,
        }
        index_path = self.root / "tokens.idx.json"
        temporary = index_path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        temporary.replace(index_path)
        return metadata

    def abort(self) -> None:
        if not self.handle.closed:
            self.handle.close()
        self.partial_path.unlink(missing_ok=True)


def encode_batch(
    tokenizer: Tokenizer,
    texts: list[str],
    *,
    num_threads: Optional[int] = None,
) -> list[list[int]]:
    backend = getattr(tokenizer, "_tokenizer", None)
    tiktoken_encoding = getattr(backend, "encoding", None)
    if tiktoken_encoding is not None and hasattr(
        tiktoken_encoding, "encode_ordinary_batch"
    ):
        return tiktoken_encoding.encode_ordinary_batch(
            texts,
            num_threads=num_threads or min(96, os.cpu_count() or 8),
        )
    if backend is not None and hasattr(backend, "encode_batch"):
        encoded = backend.encode_batch(texts, add_special_tokens=False)
        return [list(item.ids) for item in encoded]
    return [
        tokenizer.encode(text, add_special_tokens=False)
        for text in texts
    ]


def main() -> None:
    args = build_parser().parse_args()
    if args.train_tokens <= 1 or args.eval_tokens <= 1:
        raise ValueError("train-tokens and eval-tokens must both be greater than one")
    if args.eval_stride < 2:
        raise ValueError("eval-stride must be at least two")
    if args.document_batch_size <= 0:
        raise ValueError("document-batch-size must be positive")
    if args.tokenizer_threads <= 0:
        raise ValueError("tokenizer-threads must be positive")

    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.force:
            raise FileExistsError(
                f"Output already exists: {output_root}. Use --force to rebuild it."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    download_root = output_root / ".download"
    download_root.mkdir()

    required_bytes = (args.train_tokens + args.eval_tokens) * DTYPE.itemsize
    free_bytes = shutil.disk_usage(output_root).free
    if free_bytes < required_bytes + 10 * 1024**3:
        raise OSError(
            f"Insufficient free space: need at least {(required_bytes + 10 * 1024**3) / 1e9:.1f} GB, "
            f"have {free_bytes / 1e9:.1f} GB"
        )

    tokenizer = Tokenizer.load(str(args.tokenizer))
    if tokenizer.vocab_size > np.iinfo(np.uint32).max:
        raise ValueError(
            f"Tokenizer vocabulary {tokenizer.vocab_size:,} does not fit uint32"
        )
    tokenizer_files = sorted(path for path in args.tokenizer.rglob("*") if path.is_file())
    if not tokenizer_files:
        raise FileNotFoundError(f"No tokenizer files found under {args.tokenizer}")
    tokenizer_hasher = hashlib.sha256()
    for tokenizer_file in tokenizer_files:
        relative_path = tokenizer_file.relative_to(args.tokenizer)
        tokenizer_hasher.update(str(relative_path).encode("utf-8"))
        tokenizer_hasher.update(b"\0")
        tokenizer_hasher.update(bytes.fromhex(file_sha256(tokenizer_file)))
    tokenizer_digest = tokenizer_hasher.hexdigest()
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer must expose an EOS token")
    print(
        f"tokenization workers={args.tokenizer_threads} "
        f"document_batch_size={args.document_batch_size}",
        flush=True,
    )

    api = HfApi()
    files = sorted(
        filename
        for filename in api.list_repo_files(
            REPO_ID,
            repo_type="dataset",
            revision=args.revision,
        )
        if filename.startswith(DATASET_PREFIX) and filename.endswith(".parquet")
    )
    if not files:
        raise RuntimeError(
            f"No Parquet files found under {DATASET_PREFIX} at {args.revision}"
        )

    train_writer = TokenWriter(
        "train",
        output_root / "train",
        args.train_tokens,
        tokenizer.vocab_size,
        str(args.tokenizer),
        tokenizer_digest,
    )
    eval_writer = TokenWriter(
        "eval",
        output_root / "eval",
        args.eval_tokens,
        tokenizer.vocab_size,
        str(args.tokenizer),
        tokenizer_digest,
    )
    source_files: list[dict] = []
    source_documents = 0

    try:
        for filename in files:
            if train_writer.full and eval_writer.full:
                break
            local_path = download_root / Path(filename).name
            url = hf_hub_url(
                REPO_ID,
                filename=filename,
                repo_type="dataset",
                revision=args.revision,
            )
            print(f"download {filename}", flush=True)
            source_sha256, source_bytes = download_file(url, local_path)
            source_files.append(
                {
                    "path": filename,
                    "sha256": source_sha256,
                    "bytes": source_bytes,
                }
            )

            parquet = pq.ParquetFile(local_path)
            file_documents = 0
            file_total_documents = int(parquet.metadata.num_rows)
            file_started = time.monotonic()
            file_last_report = file_started
            train_tokens_before = train_writer.num_tokens
            eval_tokens_before = eval_writer.num_tokens
            for record_batch in parquet.iter_batches(
                batch_size=args.document_batch_size,
                columns=["text"],
            ):
                texts = [str(text or "") for text in record_batch.column(0).to_pylist()]
                encoded_documents = encode_batch(
                    tokenizer,
                    texts,
                    num_threads=args.tokenizer_threads,
                )
                file_documents += len(texts)
                for text, token_ids in zip(texts, encoded_documents):
                    document_index = source_documents
                    source_documents += 1
                    if not text:
                        continue
                    target = (
                        eval_writer
                        if document_index % args.eval_stride == 0
                        else train_writer
                    )
                    if target.full:
                        continue
                    target.append([*token_ids, int(eos_id)])
                    if train_writer.full and eval_writer.full:
                        break
                now = time.monotonic()
                if now - file_last_report >= 5.0:
                    elapsed = max(now - file_started, 1e-9)
                    produced_tokens = (
                        train_writer.num_tokens
                        - train_tokens_before
                        + eval_writer.num_tokens
                        - eval_tokens_before
                    )
                    print(
                        f"  tokenize {Path(filename).name}: "
                        f"{file_documents:,}/{file_total_documents:,} docs "
                        f"({100.0 * file_documents / max(file_total_documents, 1):.1f}%) · "
                        f"{produced_tokens / elapsed:,.0f} tok/s · "
                        f"train={train_writer.num_tokens:,}/{args.train_tokens:,} · "
                        f"eval={eval_writer.num_tokens:,}/{args.eval_tokens:,}",
                        flush=True,
                    )
                    file_last_report = now
                if train_writer.full and eval_writer.full:
                    break
            local_path.unlink(missing_ok=True)
            print(
                f"progress train={train_writer.num_tokens:,}/{args.train_tokens:,} "
                f"eval={eval_writer.num_tokens:,}/{args.eval_tokens:,} "
                f"documents={source_documents:,}",
                flush=True,
            )
    except Exception:
        train_writer.abort()
        eval_writer.abort()
        raise
    finally:
        shutil.rmtree(download_root, ignore_errors=True)

    common_metadata = {
        "source_repo": REPO_ID,
        "source_revision": args.revision,
        "source_subset": "sample-10BT",
        "source_files": [entry["path"] for entry in source_files],
        "eval_document_rule": f"source_document_index % {args.eval_stride} == 0",
        "eos_token_id": int(eos_id),
    }
    train_metadata = train_writer.finish(common_metadata)
    eval_metadata = eval_writer.finish(common_metadata)
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_repo": REPO_ID,
        "source_revision": args.revision,
        "source_subset": "sample-10BT",
        "source_files": source_files,
        "source_documents_scanned": source_documents,
        "tokenizer": str(args.tokenizer),
        "tokenizer_sha256": tokenizer_digest,
        "vocab_size": tokenizer.vocab_size,
        "dtype": DTYPE.str,
        "eval_stride": args.eval_stride,
        "train": {
            "path": "train",
            "num_tokens": train_metadata["num_tokens"],
            "sha256": train_metadata["sha256"],
        },
        "eval": {
            "path": "eval",
            "num_tokens": eval_metadata["num_tokens"],
            "sha256": eval_metadata["sha256"],
        },
    }
    manifest_path = output_root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
