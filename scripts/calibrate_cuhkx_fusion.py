#!/usr/bin/env python3
"""Sweep CUHK-X late-fusion calibration on a subject-disjoint checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from complexity.generative.sensor_fusion.diagnostics import (
    run_checkpoint_fusion_sweep,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-users", type=int, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_checkpoint_fusion_sweep(
        args.checkpoint,
        data_root=args.data_root,
        manifest=args.manifest,
        validation_users=args.validation_users,
        batch_size=args.batch_size,
        workers=args.workers,
        device=args.device,
    )
    payload = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
