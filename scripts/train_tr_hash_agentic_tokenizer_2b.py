#!/usr/bin/env python3
"""Train the 32K agentic tokenizer from a balanced 2B-token partial corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from scripts.train_tr_hash_agentic_tokenizer import train_tokenizer_from_iterator

LOGGER = logging.getLogger("complexity.agentic_tokenizer_2b")
SOURCE_QUOTAS = (
    ("00-dclm_general/00-dclm_general.jsonl", "general", 400_000_000),
    ("01-fineweb_edu_general/00-fineweb_edu_general.jsonl", "general", 500_000_000),
    ("02-cosmopedia_general/00-cosmopedia_general.jsonl", "general", 300_000_000),
    ("03-stack_edu_agentic/00-stack_edu_agentic.jsonl", "agentic", 400_000_000),
    ("04-fineweb_edu_agentic/00-fineweb_edu_agentic.jsonl", "agentic", 200_000_000),
    ("05-finemath_agentic/00-finemath_agentic.jsonl", "agentic", 100_000_000),
    ("06-infiwebmath_agentic/00-infiwebmath_agentic.jsonl", "agentic", 100_000_000),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class BalancedCorpus:
    def __init__(self, parts_dir: Path, reference_tokenizer: Path, batch_size: int = 512):
        from tokenizers import Tokenizer

        self.parts_dir = parts_dir
        self.batch_size = batch_size
        tokenizer_json = (
            reference_tokenizer / "tokenizer.json"
            if reference_tokenizer.is_dir()
            else reference_tokenizer
        )
        self.reference_tokenizer = Tokenizer.from_file(str(tokenizer_json))
        self.reference_tokenizer_sha256 = _sha256(tokenizer_json)
        self.sources: list[dict[str, Any]] = []

    def _source_texts(self, relative_path: str, bucket: str, quota: int) -> Iterator[str]:
        path = self.parts_dir / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        retained_tokens = 0
        retained_records = 0
        scanned_records = 0
        next_progress = quota // 10
        batch: list[str] = []

        def emit_batch() -> Iterator[str]:
            nonlocal retained_tokens, retained_records, next_progress
            encodings = self.reference_tokenizer.encode_batch(
                batch,
                add_special_tokens=False,
            )
            for text, encoding in zip(batch, encodings, strict=True):
                token_count = len(encoding.ids)
                if retained_tokens + token_count > quota:
                    continue
                retained_tokens += token_count
                retained_records += 1
                if retained_tokens >= next_progress:
                    LOGGER.info(
                        "source=%s selection=%.1f%% tokens=%s/%s records=%s",
                        path.parent.name,
                        100.0 * retained_tokens / quota,
                        f"{retained_tokens:,}",
                        f"{quota:,}",
                        f"{retained_records:,}",
                    )
                    next_progress += quota // 10
                yield text

        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if retained_tokens >= quota:
                    break
                scanned_records += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    LOGGER.warning("ignoring truncated JSONL tail: %s:%s", path, line_number)
                    break
                text = row.get("text", "")
                if not isinstance(text, str) or not text:
                    continue
                batch.append(text)
                if len(batch) >= self.batch_size:
                    yield from emit_batch()
                    batch.clear()
            if batch and retained_tokens < quota:
                yield from emit_batch()
                batch.clear()

        minimum = quota - 100_000
        if retained_tokens < minimum:
            raise RuntimeError(
                f"source {path.parent.name} reached {retained_tokens:,}/{quota:,} tokens"
            )
        self.sources.append(
            {
                "path": str(path),
                "bucket": bucket,
                "target_tokens": quota,
                "retained_tokens": retained_tokens,
                "retained_records": retained_records,
                "scanned_records": scanned_records,
                "source_file_sha256": _sha256(path),
            }
        )

    def __iter__(self) -> Iterator[str]:
        for relative_path, bucket, quota in SOURCE_QUOTAS:
            yield from self._source_texts(relative_path, bucket, quota)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-dir", required=True)
    parser.add_argument("--reference-tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    corpus = BalancedCorpus(
        Path(args.parts_dir),
        Path(args.reference_tokenizer),
        batch_size=args.batch_size,
    )
    output_dir = Path(args.output_dir)
    manifest = train_tokenizer_from_iterator(
        corpus,
        output_dir,
        vocab_size=32_000,
        manifest_extra={
            "selection_schema": "tr-hash-agentic-tokenizer-balanced-2b-v1",
            "target_tokens": 2_000_000_000,
            "target_general_tokens": 1_200_000_000,
            "target_agentic_tokens": 800_000_000,
            "reference_tokenizer_sha256": corpus.reference_tokenizer_sha256,
        },
    )
    selection = {
        "schema": "tr-hash-agentic-tokenizer-balanced-2b-v1",
        "target_tokens": 2_000_000_000,
        "retained_tokens": sum(source["retained_tokens"] for source in corpus.sources),
        "general_tokens": sum(
            source["retained_tokens"] for source in corpus.sources if source["bucket"] == "general"
        ),
        "agentic_tokens": sum(
            source["retained_tokens"] for source in corpus.sources if source["bucket"] == "agentic"
        ),
        "reference_tokenizer_sha256": corpus.reference_tokenizer_sha256,
        "sources": corpus.sources,
    }
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(selection, indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "tokenizer ready: vocab=%s retained=%s general=%s agentic=%s",
        f"{manifest['vocab_size']:,}",
        f"{selection['retained_tokens']:,}",
        f"{selection['general_tokens']:,}",
        f"{selection['agentic_tokens']:,}",
    )


if __name__ == "__main__":
    main()
