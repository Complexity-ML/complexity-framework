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


def _training_step(path: Path) -> int:
    try:
        state = torch.load(
            path / "training_state.pt",
            map_location="cpu",
            weights_only=True,
        )
    except Exception:
        return -1
    step = state.get("step")
    return step if isinstance(step, int) else -1


def latest_resumable_checkpoint(root: Path) -> Path | None:
    step_checkpoints = [
        path
        for path in root.glob("step_*")
        if path.is_dir() and (path / "training_state.pt").is_file()
    ]
    named_checkpoints = [
        root / name
        for name in ("best", "best_nms_free")
        if (root / name / "training_state.pt").is_file()
    ]
    candidates = step_checkpoints + named_checkpoints
    return max(
        candidates,
        key=lambda path: (_training_step(path), _step_number(path)),
        default=None,
    )


def checkpoint_status(
    root: Path,
    expected_epochs: int,
    expected_steps: int | None = None,
) -> tuple[int, Path | None]:
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
        if expected_steps is not None and state.get("step") != expected_steps:
            print(
                f"step budget mismatch in {checkpoint}: "
                f"saved={state.get('step')!r}, requested={expected_steps}",
                file=sys.stderr,
            )
            return INCOMPATIBLE, checkpoint
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
    parser.add_argument("--expected-steps", type=int, default=None)
    args = parser.parse_args()

    status, checkpoint = checkpoint_status(
        args.root,
        args.expected_epochs,
        args.expected_steps,
    )
    if checkpoint is not None:
        print(checkpoint)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
