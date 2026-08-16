#!/usr/bin/env python3
"""Print the latest exact-resume checkpoint under a sensor-fusion run."""

from __future__ import annotations

import argparse
from pathlib import Path

from complexity.generative.sensor_fusion.checkpointing import (
    find_latest_resumable_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    checkpoint = find_latest_resumable_checkpoint(args.output)
    if checkpoint is not None:
        print(checkpoint)


if __name__ == "__main__":
    main()
