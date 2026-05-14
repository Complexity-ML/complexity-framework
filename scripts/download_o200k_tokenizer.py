#!/usr/bin/env python3
"""Download a tiktoken encoding into a local tokenizer directory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache a tiktoken encoding locally")
    parser.add_argument("--encoding", default="o200k_base")
    parser.add_argument("--out", default="tokenizer-o200k")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(out.resolve())

    import tiktoken

    encoding = tiktoken.get_encoding(args.encoding)
    eos_token = "<|endoftext|>" if "<|endoftext|>" in encoding._special_tokens else None
    config = {
        "type": "tiktoken",
        "encoding_name": args.encoding,
        "cache_dir": ".",
        "vocab_size": encoding.n_vocab,
        "format": "tiktoken",
        "method": args.encoding,
        "eos_token": eos_token,
    }
    with open(out / "tiktoken_config.json", "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    print(f"Saved {args.encoding} ({encoding.n_vocab} tokens) to {out}")


if __name__ == "__main__":
    main()
