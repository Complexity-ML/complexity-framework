#!/usr/bin/env python3
"""Prepare a memory-mapped token shard for o200k pretraining."""

from __future__ import annotations

import argparse
from pathlib import Path

from complexity.data.token_shards import build_token_shard_from_texts, iter_texts_from_files
from complexity.tokenizer import Tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build tokens.bin + tokens.idx.json")
    parser.add_argument("inputs", nargs="+", help="Input JSONL/text files")
    parser.add_argument("--out", required=True, help="Output shard directory")
    parser.add_argument("--tokenizer", default="./tokenizer-o200k")
    parser.add_argument("--format", choices=["jsonl", "text"], default="jsonl")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--limit", type=int, default=0, help="Max records to tokenize; 0 = all")
    parser.add_argument("--no-eos", action="store_true", help="Do not append tokenizer EOS between records")
    parser.add_argument("--dtype", choices=["uint16", "uint32"], default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tokenizer = Tokenizer.load(args.tokenizer)
    texts = iter_texts_from_files(
        [Path(p) for p in args.inputs],
        input_format=args.format,
        text_field=args.text_field,
        limit=args.limit,
    )
    index = build_token_shard_from_texts(
        texts,
        tokenizer,
        args.out,
        vocab_size=tokenizer.vocab_size,
        tokenizer_name=args.tokenizer,
        add_eos=not args.no_eos,
        dtype=args.dtype,
        extra_metadata={
            "input_files": [str(Path(p)) for p in args.inputs],
            "input_format": args.format,
            "text_field": args.text_field,
        },
    )
    print(f"Wrote token shard index: {index}")


if __name__ == "__main__":
    main()
