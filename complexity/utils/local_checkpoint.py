"""Small local checkpoint helpers for single-node research runs."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch


CHECKPOINT_FILE = "checkpoint.pt"
LATEST_FILE = "latest"


def resolve_checkpoint_path(path: str | Path) -> Path:
    """Resolve a checkpoint directory, including ``.../latest`` sentinels."""

    ckpt = Path(path)
    if ckpt.name != LATEST_FILE:
        return ckpt

    parent = ckpt.parent
    latest_file = parent / LATEST_FILE
    if latest_file.is_file():
        target = latest_file.read_text(encoding="utf-8").strip()
        if target:
            resolved = parent / target
            if (resolved / CHECKPOINT_FILE).exists():
                return resolved

    candidates = _checkpoint_dirs(parent)
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {parent}")
    return candidates[-1]


def save_local_checkpoint(
    save_dir: str | Path,
    *,
    step: int,
    state: dict[str, Any],
    total_limit: int = 3,
) -> Path:
    """Atomically write ``checkpoint.pt``, update ``latest``, and rotate old steps."""

    save_root = Path(save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    ckpt_dir = save_root / f"step_{step:06d}"
    tmp_dir = save_root / f".step_{step:06d}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    torch.save(state, tmp_dir / CHECKPOINT_FILE)
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    tmp_dir.rename(ckpt_dir)

    (save_root / LATEST_FILE).write_text(f"{ckpt_dir.name}\n", encoding="utf-8")
    rotate_local_checkpoints(save_root, total_limit=total_limit)
    return ckpt_dir


def load_local_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[Path, dict[str, Any]]:
    """Load a local checkpoint state dict and return its resolved directory."""

    ckpt_dir = resolve_checkpoint_path(path)
    ckpt_file = ckpt_dir / CHECKPOINT_FILE
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_file}")
    state = torch.load(ckpt_file, map_location=map_location)
    return ckpt_dir, state


def rotate_local_checkpoints(save_dir: str | Path, *, total_limit: int = 3) -> None:
    """Keep the newest ``total_limit`` step directories."""

    save_root = Path(save_dir)
    checkpoints = _checkpoint_dirs(save_root)
    excess = len(checkpoints) - max(1, total_limit)
    for old in checkpoints[: max(0, excess)]:
        shutil.rmtree(old)


def _checkpoint_dirs(save_root: Path) -> list[Path]:
    return sorted(path for path in save_root.glob("step_*") if path.is_dir())
