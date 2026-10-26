"""Exact training-state checkpointing for the object detector."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

TRAINING_STATE_FILENAME = "training_state.pt"
TRAINING_STATE_FORMAT = 1


def _mismatched_options(
    saved: Mapping[str, Any], expected: Mapping[str, Any]
) -> list[str]:
    keys = sorted(set(saved) | set(expected))
    return [key for key in keys if saved.get(key) != expected.get(key)]


def save_training_state(
    checkpoint: Path,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    epoch: int,
    batch_in_epoch: int,
    step: int,
    best_map50: float,
    best_end_to_end_map50: float = -1.0,
    running_losses: Mapping[str, float],
    running_loss_steps: int,
    total_epochs: int,
    steps_per_epoch: int,
    training_options: Mapping[str, Any],
    distributed_rng_states: Sequence[Mapping[str, torch.Tensor]] | None = None,
) -> None:
    """Write the state needed to continue at the exact next optimizer step.

    The model itself remains in ``model.safetensors``.  This file is written
    last so its presence also marks a checkpoint as exactly resumable.
    """

    state = {
        "format_version": TRAINING_STATE_FORMAT,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "step": step,
        "best_map50": best_map50,
        "best_end_to_end_map50": best_end_to_end_map50,
        "running_losses": dict(running_losses),
        "running_loss_steps": running_loss_steps,
        "total_epochs": total_epochs,
        "steps_per_epoch": steps_per_epoch,
        "training_options": dict(training_options),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() and distributed_rng_states is None
            else []
        ),
    }
    if distributed_rng_states is not None:
        state["distributed_rng_states"] = list(distributed_rng_states)
        state["world_size"] = len(distributed_rng_states)
    path = checkpoint / TRAINING_STATE_FILENAME
    temporary_path = checkpoint / f".{TRAINING_STATE_FILENAME}.tmp"
    torch.save(state, temporary_path)
    temporary_path.replace(path)


def load_training_state(
    checkpoint: Path,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    total_epochs: int,
    steps_per_epoch: int,
    training_options: Mapping[str, Any],
    rank: int = 0,
    world_size: int = 1,
    device: torch.device | None = None,
) -> Dict[str, Any]:
    """Restore optimizer, scheduler, cursor and RNG state from ``checkpoint``."""

    path = checkpoint / TRAINING_STATE_FILENAME
    if not path.is_file():
        raise ValueError(
            f"{checkpoint} is a weights-only checkpoint and cannot be resumed exactly; "
            "use --detector-checkpoint to fine-tune from its weights"
        )
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format_version") != TRAINING_STATE_FORMAT:
        raise ValueError(
            f"unsupported detector training-state format: {state.get('format_version')!r}"
        )
    if state.get("total_epochs") != total_epochs:
        raise ValueError(
            "--epochs is the total run target and must match the resumed run "
            f"({total_epochs} != {state.get('total_epochs')})"
        )
    if state.get("steps_per_epoch") != steps_per_epoch:
        raise ValueError(
            "training dataset or batch size changed: steps per epoch do not match "
            f"({steps_per_epoch} != {state.get('steps_per_epoch')})"
        )
    mismatches = _mismatched_options(state.get("training_options", {}), training_options)
    if mismatches:
        raise ValueError(
            "exact resume requires unchanged training options; mismatched: "
            + ", ".join(mismatches)
        )

    distributed_rng_states = state.get("distributed_rng_states")
    saved_world_size = int(state.get("world_size", 1))
    if saved_world_size != world_size:
        raise ValueError(
            "distributed world size differs from the resumed run "
            f"({world_size} != {saved_world_size})"
        )
    if not 0 <= rank < world_size:
        raise ValueError(f"distributed rank out of range: {rank}")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if distributed_rng_states is not None:
        rng_state = distributed_rng_states[rank]
        torch.set_rng_state(rng_state["torch"])
        if device is None or device.type != "cuda":
            raise ValueError("distributed CUDA checkpoint requires a CUDA device")
        torch.cuda.set_rng_state(rng_state["cuda"], device=device)
    else:
        torch.set_rng_state(state["torch_rng_state"])
    cuda_rng_states = state.get("cuda_rng_state_all", [])
    if distributed_rng_states is None and cuda_rng_states:
        if not torch.cuda.is_available():
            raise ValueError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_rng_states) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from the resumed run")
        torch.cuda.set_rng_state_all(cuda_rng_states)
    return state
