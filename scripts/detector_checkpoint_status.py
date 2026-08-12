"""Inspect whether a detector run has reached its requested epoch budget."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

COMPLETE = 0
INCOMPLETE = 10
NOT_FOUND = 20
INCOMPATIBLE = 30


def _step_number(path: Path) -> int:
    try:
        return int(path.name.removeprefix("step_"))
    except ValueError:
        return -1


def latest_resumable_checkpoint(root: Path) -> Path | None:
    candidates = [
        path
        for path in root.glob("step_*")
        if path.is_dir() and (path / "training_state.pt").is_file()
    ]
    return max(candidates, key=_step_number, default=None)


def checkpoint_status(root: Path, expected_epochs: int) -> tuple[int, Path | None]:
    checkpoint = latest_resumable_checkpoint(root)
    if checkpoint is None:
        return NOT_FOUND, None

    try:
        state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
    except Exception as error:
        print(f"invalid training state in {checkpoint}: {error}", file=sys.stderr)
        return INCOMPATIBLE, checkpoint

    saved_epochs = state.get("total_epochs")
    if saved_epochs != expected_epochs:
        print(
            f"epoch budget mismatch in {checkpoint}: "
            f"saved={saved_epochs!r}, requested={expected_epochs}",
            file=sys.stderr,
        )
        return INCOMPATIBLE, checkpoint

    epoch = state.get("epoch")
    batch_in_epoch = state.get("batch_in_epoch")
    if not isinstance(epoch, int) or not isinstance(batch_in_epoch, int):
        print(f"invalid training cursor in {checkpoint}", file=sys.stderr)
        return INCOMPATIBLE, checkpoint
    if epoch == expected_epochs and batch_in_epoch == 0:
        return COMPLETE, checkpoint
    if 0 <= epoch < expected_epochs and batch_in_epoch >= 0:
        return INCOMPLETE, checkpoint

    print(
        f"out-of-range training cursor in {checkpoint}: epoch={epoch}, batch={batch_in_epoch}",
        file=sys.stderr,
    )
    return INCOMPATIBLE, checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-epochs", type=int, required=True)
    args = parser.parse_args()

    status, checkpoint = checkpoint_status(args.root, args.expected_epochs)
    if checkpoint is not None:
        print(checkpoint)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
