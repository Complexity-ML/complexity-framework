#!/usr/bin/env python3
"""Materialize pinned full ARC test splits for auditable generative evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset

ARC_REVISION = "210d026faf9955653af8916fad021475a3f00453"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for task, config in (
        ("arc_easy", "ARC-Easy"),
        ("arc_challenge", "ARC-Challenge"),
    ):
        dataset = load_dataset(
            "allenai/ai2_arc",
            config,
            split="test",
            revision=ARC_REVISION,
        )
        path = output / f"samples_{task}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for doc_id, row in enumerate(dataset):
                handle.write(
                    json.dumps(
                        {"doc_id": doc_id, "doc": dict(row)},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        files[task] = {
            "examples": len(dataset),
            "sha256": _sha256(path),
            "path": str(path),
        }
    manifest = {
        "schema_version": 1,
        "dataset": "allenai/ai2_arc",
        "revision": ARC_REVISION,
        "split": "test",
        "files": files,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
