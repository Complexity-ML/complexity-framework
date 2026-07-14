#!/usr/bin/env python3
"""Count tokenizer IDs in the frozen FineWeb parquet training split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset

from complexity.tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-stride", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = Tokenizer.load(args.tokenizer)
    vocab_size = int(tokenizer.vocab_size)
    counts = torch.zeros(vocab_size, dtype=torch.int64)
    dataset = load_dataset(
        "parquet",
        data_files={"train": args.parquet},
        split="train",
        streaming=True,
    )
    documents = 0
    tokens = 0
    eos_token_id = tokenizer.eos_token_id
    for index, example in enumerate(dataset):
        if index % args.eval_stride == 0:
            continue
        text = example.get("text", "")
        if not text:
            continue
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if eos_token_id is not None:
            token_ids.append(eos_token_id)
        ids = torch.tensor(token_ids, dtype=torch.long)
        ids = ids[(ids >= 0) & (ids < vocab_size)]
        if ids.numel():
            counts += torch.bincount(ids, minlength=vocab_size)
            tokens += int(ids.numel())
        documents += 1
        if args.log_every > 0 and documents % args.log_every == 0:
            print(
                f"documents={documents:,} tokens={tokens:,} "
                f"observed_vocab={int((counts > 0).sum()):,}",
                flush=True,
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "counts": counts,
            "documents": documents,
            "tokens": tokens,
            "vocab_size": vocab_size,
            "eval_stride": args.eval_stride,
            "parquet_name": Path(args.parquet).name,
            "tokenizer": str(args.tokenizer),
        },
        output,
    )
    print(
        f"saved={output} documents={documents:,} tokens={tokens:,} "
        f"observed_vocab={int((counts > 0).sum()):,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
