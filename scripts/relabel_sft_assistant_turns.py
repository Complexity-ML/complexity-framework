#!/usr/bin/env python3
"""Create a new SFT shard by changing labels only, never source content."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from complexity.tokenizer import Tokenizer
from complexity.training.sft_relabel import relabel_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--reuse-content",
        action="store_true",
        help="Required acknowledgement that input tokens and metadata are reused.",
    )
    parser.add_argument(
        "--skip-content-verification",
        action="store_true",
        help="Skip expensive content hashing; causal label checks still run.",
    )
    args = parser.parse_args()
    if not args.reuse_content:
        parser.error("--reuse-content is required for label-only regeneration")

    tokenizer = Tokenizer.load(str(args.tokenizer))
    results = relabel_dataset(
        args.source,
        args.output,
        tokenizer=tokenizer,
        skip_content_verification=args.skip_content_verification,
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
