#!/usr/bin/env python3
"""Remove resumable DDP checkpoints after a verified successful export.

The final compact ``final/model.safetensors`` export is preserved. Only
numbered checkpoint directories created for resumption are eligible.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


CHECKPOINT_NAME = re.compile(
    r"(?:token_pack_\d+|step_\d+|interrupted_\d+|final_\d+|best_\d+)"
)


def cleanup(checkpoint_dir: Path) -> tuple[list[str], int]:
    checkpoint_dir = checkpoint_dir.resolve()
    final_weights = checkpoint_dir / "final" / "model.safetensors"
    if not final_weights.is_file() or final_weights.stat().st_size == 0:
        raise RuntimeError(
            f"refusing cleanup: validated final export is missing: {final_weights}"
        )

    removed: list[str] = []
    reclaimed = 0
    for child in checkpoint_dir.iterdir():
        if not child.is_dir() or CHECKPOINT_NAME.fullmatch(child.name) is None:
            continue
        reclaimed += sum(
            path.stat().st_size for path in child.rglob("*") if path.is_file()
        )
        shutil.rmtree(child)
        removed.append(child.name)
    return sorted(removed), reclaimed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    removed, reclaimed = cleanup(args.checkpoint_dir)
    print(
        f"Post-training checkpoint cleanup: removed={removed} "
        f"reclaimed_gib={reclaimed / 1024**3:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
