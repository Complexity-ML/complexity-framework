"""Exact resumable checkpoints for TR-HASH sensor-fusion training."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file

from .config import TRHashSensorFusionConfig
from .preprocessing import CUHKXPreprocessingConfig

TRAINING_STATE_FORMAT = 1


def find_latest_resumable_checkpoint(output: str | Path) -> Path | None:
    """Return the child checkpoint with the greatest persisted training step."""

    output = Path(output)
    if not output.is_dir():
        return None
    candidates: list[tuple[tuple[int, int, int], Path]] = []
    for checkpoint in output.iterdir():
        state_path = checkpoint / "training_state.pt"
        if not checkpoint.is_dir() or not state_path.is_file():
            continue
        try:
            state = torch.load(state_path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError):
            continue
        if state.get("format_version") != TRAINING_STATE_FORMAT:
            continue
        cursor = (
            int(state.get("step", -1)),
            int(state.get("epoch", -1)),
            int(state.get("batch_in_epoch", -1)),
        )
        candidates.append((cursor, checkpoint))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], str(item[1])))[1]


def save_sensor_fusion_checkpoint(
    checkpoint: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    model_config: TRHashSensorFusionConfig,
    preprocessing: CUHKXPreprocessingConfig,
    epoch: int,
    batch_in_epoch: int,
    step: int,
    best_accuracy: float,
    total_epochs: int,
    steps_per_epoch: int,
    training_options: Mapping[str, Any],
    metrics: Mapping[str, Any],
    distributed_rng_states: Sequence[Mapping[str, torch.Tensor]] | None = None,
) -> None:
    """Save weights, configs, metrics, optimizer, scheduler, cursor, and RNG."""

    checkpoint.mkdir(parents=True, exist_ok=True)
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
    }
    save_file(state, str(checkpoint / "model.safetensors"))
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "model": model_config.to_dict(),
                "preprocessing": preprocessing.to_dict(),
            },
            indent=2,
        )
        + "\n"
    )
    (checkpoint / "metrics.json").write_text(json.dumps(dict(metrics), indent=2) + "\n")
    state = {
        "format_version": TRAINING_STATE_FORMAT,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": int(epoch),
        "batch_in_epoch": int(batch_in_epoch),
        "step": int(step),
        "best_accuracy": float(best_accuracy),
        "total_epochs": int(total_epochs),
        "steps_per_epoch": int(steps_per_epoch),
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
    temporary = checkpoint / ".training_state.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(checkpoint / "training_state.pt")


def load_sensor_fusion_checkpoint(
    checkpoint: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    model_config: TRHashSensorFusionConfig,
    preprocessing: CUHKXPreprocessingConfig,
    total_epochs: int,
    steps_per_epoch: int,
    training_options: Mapping[str, Any],
    rank: int = 0,
    world_size: int = 1,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Restore a checkpoint and reject any option that breaks exact resume."""

    metadata = json.loads((checkpoint / "config.json").read_text())
    if TRHashSensorFusionConfig.from_dict(metadata["model"]) != model_config:
        raise ValueError("exact resume requires an unchanged sensor-fusion model config")
    if CUHKXPreprocessingConfig.from_dict(metadata["preprocessing"]) != preprocessing:
        raise ValueError("exact resume requires unchanged preprocessing")
    model.load_state_dict(load_file(str(checkpoint / "model.safetensors"), device="cpu"))

    state_path = checkpoint / "training_state.pt"
    if not state_path.is_file():
        raise ValueError(f"{checkpoint} is weights-only and cannot be resumed exactly")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if state.get("format_version") != TRAINING_STATE_FORMAT:
        raise ValueError("unsupported sensor-fusion training-state format")
    if int(state["total_epochs"]) != total_epochs:
        raise ValueError("--epochs must match the original total run target")
    if int(state["steps_per_epoch"]) != steps_per_epoch:
        raise ValueError("dataset, world size, or batch size changed since checkpoint")
    saved_options = state.get("training_options", {})
    mismatches = sorted(
        key
        for key in set(saved_options) | set(training_options)
        if saved_options.get(key) != training_options.get(key)
    )
    if mismatches:
        raise ValueError("exact resume options changed: " + ", ".join(mismatches))
    saved_world_size = int(state.get("world_size", 1))
    if saved_world_size != world_size:
        raise ValueError("distributed world size changed since checkpoint")

    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    distributed = state.get("distributed_rng_states")
    if distributed is not None:
        if device is None or device.type != "cuda":
            raise ValueError("distributed checkpoint resume requires CUDA")
        torch.set_rng_state(distributed[rank]["torch"])
        torch.cuda.set_rng_state(distributed[rank]["cuda"], device=device)
    else:
        torch.set_rng_state(state["torch_rng_state"])
        cuda_states = state.get("cuda_rng_state_all", [])
        if cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)
    return state
