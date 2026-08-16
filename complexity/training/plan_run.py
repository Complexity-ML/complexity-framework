#!/usr/bin/env python3
"""Print token-budget arithmetic for local DDP pretraining runs."""

from __future__ import annotations

import argparse
import math


def parse_tokens(value: str) -> int:
    text = value.strip().lower().replace("_", "")
    multipliers = {
        "k": 1_000,
        "m": 1_000_000,
        "b": 1_000_000_000,
        "t": 1_000_000_000_000,
    }
    if text[-1:] in multipliers:
        return int(float(text[:-1]) * multipliers[text[-1]])
    return int(float(text))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan a token-budget training run")
    parser.add_argument("--tokens", required=True, type=parse_tokens, help="Target tokens, e.g. 30B or 100B")
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True, help="Per-GPU batch size")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--tok-s", type=float, default=None, help="Measured tokens/sec for ETA")
    parser.add_argument("--save-steps", type=int, default=None)
    args = parser.parse_args()

    tokens_per_step = args.gpus * args.batch_size * args.seq_len
    steps = math.ceil(args.tokens / tokens_per_step)
    actual_tokens = steps * tokens_per_step

    print(f"target_tokens     {args.tokens:,}")
    print(f"tokens_per_step  {tokens_per_step:,}")
    print(f"steps            {steps:,}")
    print(f"actual_tokens    {actual_tokens:,}")
    print(f"overshoot        {actual_tokens - args.tokens:,}")
    if args.save_steps:
        save_tokens = args.save_steps * tokens_per_step
        print(f"save_every       {save_tokens:,} tokens")
        if args.tok_s:
            print(f"save_every_time  {save_tokens / args.tok_s / 60:.1f} min")
    if args.tok_s:
        hours = actual_tokens / args.tok_s / 3600
        print(f"eta              {hours:.2f} h ({hours / 24:.2f} days)")


if __name__ == "__main__":
    main()
