#!/usr/bin/env python3
"""Mirror an active TR-HASH metrics CSV into TensorBoard event files."""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Any

SCALARS = {
    "train_loss": "Loss/train",
    "train_ppl": "PPL/train",
    "matched_eval_loss": "Loss/eval",
    "matched_eval_ppl": "PPL/eval",
    "lr": "LR",
    "expert_lr": "LR/expert",
    "tok_s": "Throughput/tokens_per_second",
    "supervised_tokens": "Tokens/supervised_per_batch",
    "matched_eval_tokens": "Tokens/eval",
    "natural_eval_loss": "Loss/natural_eval",
    "natural_eval_ppl": "PPL/natural_eval",
    "natural_eval_tokens": "Tokens/natural_eval",
}


def finite_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def mirror_once(csv_path: Path, writer: Any, seen: set[tuple[int, str]]) -> int:
    written = 0
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            step = int(row["step"])
            for column, tag in SCALARS.items():
                value = finite_float(row.get(column))
                key = (step, tag)
                if value is None or key in seen:
                    continue
                writer.add_scalar(tag, value, step)
                seen.add(key)
                written += 1
    if written:
        writer.flush()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--follow", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--idle-timeout", type=float, default=600.0)
    args = parser.parse_args()

    if args.poll_seconds <= 0 or args.idle_timeout <= 0:
        raise ValueError("poll and idle timeout values must be positive")

    from torch.utils.tensorboard import SummaryWriter

    args.logdir.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[int, str]] = set()
    last_change = time.monotonic()
    with SummaryWriter(log_dir=str(args.logdir), flush_secs=5) as writer:
        while True:
            if args.csv.exists() and mirror_once(args.csv, writer, seen):
                last_change = time.monotonic()
                print(f"mirrored {len(seen):,} scalar points from {args.csv}", flush=True)
            if not args.follow:
                break
            if time.monotonic() - last_change >= args.idle_timeout:
                print(f"idle timeout reached after {args.idle_timeout:g}s", flush=True)
                break
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
