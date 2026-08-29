#!/usr/bin/env python3
"""Restart-safe orchestrator for the TR-HASH agentic 50M data laboratory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from scripts.build_agentic_pretraining_corpus import build_corpus, validate_config
from scripts.tokenize_agentic_pretraining_corpus import tokenize_corpus
from scripts.train_tr_hash_agentic_tokenizer import train_tokenizer


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config_sha256(config: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def prepare(
    *,
    config_path: Path,
    raw_dir: Path,
    tokenizer_dir: Path,
    token_dir: Path,
    target_tokens: int,
    seq_len: int,
    global_batch_sequences: int,
) -> None:
    config = _load(config_path)
    validate_config(config)
    expected_config_sha = _config_sha256(config)

    raw_manifest_path = raw_dir / "manifest.json"
    if raw_manifest_path.is_file():
        raw_manifest = _load(raw_manifest_path)
        if raw_manifest.get("config_sha256") != expected_config_sha:
            raise ValueError("existing raw corpus was built from a different config")
        print(f"[resume] raw corpus verified: {raw_manifest_path}", flush=True)
    else:
        print("[stage 1/3] building audited general/agentic raw corpus", flush=True)
        build_corpus(config, raw_dir)

    tokenizer_manifest_path = tokenizer_dir / "agentic_tokenizer_manifest.json"
    raw_manifest_sha = _sha256(raw_manifest_path)
    if tokenizer_manifest_path.is_file():
        tokenizer_manifest = _load(tokenizer_manifest_path)
        if tokenizer_manifest.get("corpus_manifest_sha256") != raw_manifest_sha:
            raise ValueError("existing tokenizer belongs to a different raw corpus")
        if int(tokenizer_manifest.get("vocab_size", 0)) != 32_000:
            raise ValueError("existing agentic tokenizer is not exactly 32K")
        print(f"[resume] tokenizer verified: {tokenizer_manifest_path}", flush=True)
    else:
        print("[stage 2/3] training fixed 32K agentic tokenizer", flush=True)
        train_tokenizer(raw_dir, tokenizer_dir)

    lineage_path = token_dir / "lineage.json"
    if lineage_path.is_file():
        lineage = _load(lineage_path)
        if lineage.get("raw_corpus_manifest_sha256") != raw_manifest_sha:
            raise ValueError("existing token shards belong to a different raw corpus")
        if lineage.get("tokenizer_manifest_sha256") != _sha256(tokenizer_manifest_path):
            raise ValueError("existing token shards belong to a different tokenizer")
        print(f"[resume] token shards verified: {lineage_path}", flush=True)
        return

    if token_dir.exists():
        print("[resume] removing incomplete generated token shards", flush=True)
        shutil.rmtree(token_dir)
    print("[stage 3/3] materializing exact 50M-token mmap mixture", flush=True)
    lineage = tokenize_corpus(
        corpus_dir=raw_dir,
        tokenizer_path=tokenizer_dir,
        output_root=token_dir,
        target_tokens=target_tokens,
        seq_len=seq_len,
        global_batch_sequences=global_batch_sequences,
        shard_trained_tokens=10_000_000,
        document_batch_size=256,
    )
    print(f"[done] agentic lab tokens={lineage['actual_tokens']:,}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/agentic_pretraining/tr_hash_small_agentic_50m.json",
    )
    parser.add_argument("--raw-dir", default="artifacts/tr_hash_small_agentic_raw")
    parser.add_argument("--tokenizer-dir", default="artifacts/tr_hash_agentic_tokenizer_32k")
    parser.add_argument("--token-dir", default="artifacts/tr_hash_small_agentic_50m_tokens")
    parser.add_argument("--target-tokens", type=int, default=50_000_000)
    parser.add_argument("--seq-len", type=int, default=1_024)
    parser.add_argument("--global-batch-sequences", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare(
        config_path=Path(args.config),
        raw_dir=Path(args.raw_dir),
        tokenizer_dir=Path(args.tokenizer_dir),
        token_dir=Path(args.token_dir),
        target_tokens=args.target_tokens,
        seq_len=args.seq_len,
        global_batch_sequences=args.global_batch_sequences,
    )


if __name__ == "__main__":
    main()
