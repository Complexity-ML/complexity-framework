#!/usr/bin/env python3
"""Verify frozen token-shard sizes and SHA-256 digests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path, chunk_size: int = 64 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_partition(root: Path) -> dict:
    index_path = root / "tokens.idx.json"
    metadata = json.loads(index_path.read_text())
    token_path = root / metadata["bin"]
    expected_bytes = int(metadata["num_tokens"]) * np.dtype(metadata["dtype"]).itemsize
    actual_bytes = token_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{root}: size mismatch, expected {expected_bytes:,}, got {actual_bytes:,}"
        )
    actual_sha = sha256(token_path)
    if actual_sha != metadata["sha256"]:
        raise ValueError(
            f"{root}: SHA-256 mismatch, expected {metadata['sha256']}, got {actual_sha}"
        )
    return {
        "path": str(root),
        "num_tokens": int(metadata["num_tokens"]),
        "bytes": actual_bytes,
        "sha256": actual_sha,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    results = [
        verify_partition(root / "train"),
        verify_partition(root / "eval"),
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
