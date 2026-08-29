#!/usr/bin/env python3
"""Tokenize an audited agentic raw corpus into the TR-HASH mmap format."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from complexity.training import TextCorpusSource
from scripts.tokenize_tr_hash_200m_200b import (
    TokenShardWriter,
    resolve_layout,
    write_mixture_manifest,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_texts(paths: Sequence[Path]) -> Iterator[str]:
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("text", "")
                if not isinstance(text, str) or not text:
                    raise ValueError(f"missing text at {path}:{line_number}")
                yield text


def _text_batches(texts: Iterator[str], batch_size: int) -> Iterator[list[str]]:
    while batch := list(itertools.islice(texts, batch_size)):
        yield batch


def resolve_bucket_sources(
    corpus_dir: Path,
) -> tuple[list[TextCorpusSource], dict[str, list[Path]]]:
    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"agentic corpus manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "tr-hash-agentic-pretraining-corpus-v1":
        raise ValueError("unsupported agentic corpus manifest")

    weights: defaultdict[str, float] = defaultdict(float)
    paths: defaultdict[str, list[Path]] = defaultdict(list)
    for source in manifest["sources"]:
        bucket = source["bucket"]
        weights[bucket] += float(source["weight"])
        path = corpus_dir / source["output"]
        if not path.is_file():
            raise FileNotFoundError(f"raw corpus shard missing: {path}")
        if _sha256_file(path) != source["output_sha256"]:
            raise ValueError(f"raw corpus shard checksum mismatch: {path}")
        paths[bucket].append(path)

    sources = [
        TextCorpusSource(
            name=bucket,
            weight=weights[bucket],
            data_files=[str(path) for path in paths[bucket]],
        )
        for bucket in ("general", "agentic")
        if paths[bucket]
    ]
    return sources, dict(paths)


def build_pretraining_plan(
    *,
    output_root: Path,
    sources: Sequence[TextCorpusSource],
    seq_len: int,
    actual_tokens: int,
    global_batch_sequences: int,
) -> Path:
    phase_sources: dict[str, list[dict[str, int | str]]] = {}
    source_unique_tokens: dict[str, int] = {}
    for source in sources:
        manifest = json.loads(
            (output_root / "corpora" / source.name / "manifest.json").read_text(encoding="utf-8")
        )
        phase_sources[source.name] = [
            {"file": shard["file"], "rows": int(shard["rows"])} for shard in manifest["shards"]
        ]
        source_unique_tokens[source.name] = int(manifest["trained_tokens"])

    plan = {
        "format": "tr-hash-token-replay-plan-v1",
        "dataset": str(output_root.resolve()),
        "revision": "local-audited",
        "seq_len": seq_len,
        "selection_mode": "manifest_order",
        "row_alignment": global_batch_sequences,
        "unique_tokens": actual_tokens,
        "trained_tokens": actual_tokens,
        "source_unique_tokens": source_unique_tokens,
        "source_passes": {source.name: 1 for source in sources},
        "phases": [{"name": "unique_core", "passes": 1, "sources": phase_sources}],
    }
    path = output_root / "pretrain_plan.json"
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return path


def tokenize_corpus(
    *,
    corpus_dir: Path,
    tokenizer_path: Path,
    output_root: Path,
    target_tokens: int,
    seq_len: int,
    global_batch_sequences: int,
    shard_trained_tokens: int,
    document_batch_size: int,
) -> dict[str, object]:
    from transformers import PreTrainedTokenizerFast

    sources, bucket_paths = resolve_bucket_sources(corpus_dir)
    total_rows, actual_tokens, rows_by_source = resolve_layout(
        target_tokens=target_tokens,
        seq_len=seq_len,
        global_batch_sequences=global_batch_sequences,
        sources=sources,
    )
    del total_rows
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    if len(tokenizer) != 32_000:
        raise ValueError(f"agentic tokenizer must contain exactly 32,000 IDs, got {len(tokenizer)}")
    rows_per_shard = max(1, shard_trained_tokens // seq_len)

    for source in sources:
        writer = TokenShardWriter(
            output_root / "corpora" / source.name,
            seq_len=seq_len,
            total_rows=rows_by_source[source.name],
            rows_per_shard=rows_per_shard,
        )
        for texts in _text_batches(
            iter(_iter_texts(bucket_paths[source.name])), document_batch_size
        ):
            encoded = tokenizer(
                texts,
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )["input_ids"]
            eos = () if tokenizer.eos_token_id is None else (int(tokenizer.eos_token_id),)
            token_count = sum(len(tokens) + len(eos) for tokens in encoded)
            flat = np.fromiter(
                itertools.chain.from_iterable(itertools.chain(tokens, eos) for tokens in encoded),
                dtype=np.uint16,
                count=token_count,
            )
            writer.feed(flat)
            if writer.complete:
                writer.write_manifest(source=source)
                break
        if not writer.complete:
            raise RuntimeError(
                f"bucket {source.name} exhausted at {writer.source_tokens_written:,} tokens; "
                f"need {writer.required_source_tokens:,}"
            )

    mixture_path = write_mixture_manifest(
        output_root=output_root,
        sources=sources,
        seq_len=seq_len,
        requested_tokens=target_tokens,
        actual_tokens=actual_tokens,
        global_batch_sequences=global_batch_sequences,
        rows_by_source=rows_by_source,
    )
    plan_path = build_pretraining_plan(
        output_root=output_root,
        sources=sources,
        seq_len=seq_len,
        actual_tokens=actual_tokens,
        global_batch_sequences=global_batch_sequences,
    )
    lineage = {
        "schema": "tr-hash-agentic-token-lineage-v1",
        "raw_corpus_manifest": str((corpus_dir / "manifest.json").resolve()),
        "raw_corpus_manifest_sha256": _sha256_file(corpus_dir / "manifest.json"),
        "tokenizer": str(tokenizer_path.resolve()),
        "tokenizer_manifest_sha256": _sha256_file(
            tokenizer_path / "agentic_tokenizer_manifest.json"
        ),
        "requested_tokens": target_tokens,
        "actual_tokens": actual_tokens,
        "mixture_manifest_sha256": _sha256_file(mixture_path),
        "pretrain_plan_sha256": _sha256_file(plan_path),
    }
    (output_root / "lineage.json").write_text(
        json.dumps(lineage, indent=2) + "\n",
        encoding="utf-8",
    )
    return lineage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-tokens", type=int, default=50_000_000)
    parser.add_argument("--seq-len", type=int, default=1_024)
    parser.add_argument("--global-batch-sequences", type=int, default=8)
    parser.add_argument("--shard-trained-tokens", type=int, default=10_000_000)
    parser.add_argument("--document-batch-size", type=int, default=256)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    lineage = tokenize_corpus(
        corpus_dir=Path(args.corpus_dir),
        tokenizer_path=Path(args.tokenizer),
        output_root=Path(args.output),
        target_tokens=args.target_tokens,
        seq_len=args.seq_len,
        global_batch_sequences=args.global_batch_sequences,
        shard_trained_tokens=args.shard_trained_tokens,
        document_batch_size=args.document_batch_size,
    )
    print(f"Agentic tokens ready: {lineage['actual_tokens']:,}")
    print(f"Lineage: {Path(args.output) / 'lineage.json'}")


if __name__ == "__main__":
    main()
