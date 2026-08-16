"""
Write a small FineWeb-Edu sample to a local text file for repeatable local A/Bs.

This intentionally materializes text before training so the local train scripts
do not keep HuggingFace streaming HTTP workers alive after a short run.

Example:
    python3 scripts/prepare_fineweb_local.py --docs 200 --output runs/fineweb_local/sample.txt
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from datasets import load_dataset


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("prepare_fineweb_local")
for noisy_logger in ("httpx", "httpcore", "huggingface_hub", "datasets"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description="Prepare a local FineWeb-Edu text sample")
    parser.add_argument("--docs", type=int, default=200)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--output", type=str, default="runs/fineweb_local/sample.txt")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    kept = 0
    chars = 0
    with out_path.open("w", encoding="utf-8") as f:
        for example in ds:
            text = example.get("text", "")
            if len(text) < args.min_chars:
                continue
            f.write(text.strip())
            f.write("\n\n")
            kept += 1
            chars += len(text)
            if kept >= args.docs:
                break
        f.flush()
        os.fsync(f.fileno())

    logger.info(f"Wrote {kept} docs, {chars:,} chars to {out_path}")

    # Some versions of HF streaming leave background HTTP state alive for a
    # while. We have already flushed/fsynced the artifact, so exit immediately.
    os._exit(0)


if __name__ == "__main__":
    main()
