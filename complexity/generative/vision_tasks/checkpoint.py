"""Strict, task-aware checkpoints for the shared TR-Hash vision family."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from safetensors.torch import load_file, save_file

from ..detection import TRHashDetectorConfig
from .model import SUPPORTED_VISION_TASKS, VisionTask, create_vision_model


def _model_config(model) -> TRHashDetectorConfig:
    config = getattr(model, "config", None) or getattr(model, "detector_config", None)
    if not isinstance(config, TRHashDetectorConfig):
        raise TypeError("vision task model must expose a TRHashDetectorConfig")
    return config


def _task_options(model, task: VisionTask) -> dict[str, Any]:
    if task == "classification":
        return {"num_classes": int(model.head.out_features)}
    if task == "semantic_segmentation":
        return {"num_classes": int(model.num_classes)}
    if task == "pose":
        return {"num_keypoints": int(model.num_keypoints)}
    if task == "instance_segmentation":
        return {"num_prototypes": int(model.num_prototypes)}
    if task == "depth":
        return {"max_depth": model.max_depth}
    return {}


def save_vision_task_checkpoint(
    model: torch.nn.Module,
    output: Path | str,
    *,
    task: VisionTask,
    class_names: Optional[Sequence[str]] = None,
) -> Path:
    """Save architecture, task identity, names, and weights without pickles."""

    if task not in SUPPORTED_VISION_TASKS:
        raise ValueError(f"unsupported vision task: {task}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = _model_config(model)
    options = _task_options(model, task)
    if task == "depth":
        expected_classes = 0
    elif task == "pose":
        expected_classes = int(options["num_keypoints"])
    else:
        expected_classes = int(options.get("num_classes", config.num_classes))
    names = (
        tuple(str(name) for name in class_names)
        if class_names is not None
        else tuple(str(index) for index in range(expected_classes))
    )
    if task in {
        "detection",
        "classification",
        "semantic_segmentation",
        "instance_segmentation",
        "pose",
        "obb",
    }:
        if len(names) != expected_classes:
            raise ValueError("class_names must match the task class count")

    (output / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    manifest = {"format_version": 5, "task": task, "options": options}
    (output / "vision_task.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "class_names.json").write_text(json.dumps(names, indent=2) + "\n")
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
    }
    save_file(state, str(output / "model.safetensors"))
    return output.resolve()


def load_vision_task_checkpoint(
    checkpoint: Path | str,
    *,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Recreate a vision task model and validate every state tensor strictly."""

    checkpoint = Path(checkpoint)
    manifest = json.loads((checkpoint / "vision_task.json").read_text())
    format_version = manifest.get("format_version")
    if format_version != 5:
        raise ValueError("unsupported vision task checkpoint format")
    task = str(manifest.get("task"))
    if task not in SUPPORTED_VISION_TASKS:
        raise ValueError(f"unsupported vision task: {task}")
    options = dict(manifest.get("options", {}))
    allowed = {
        "num_classes",
        "num_keypoints",
        "num_prototypes",
        "max_depth",
    }
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise ValueError(f"unknown vision task options: {unknown}")
    config = TRHashDetectorConfig.from_dict(
        json.loads((checkpoint / "config.json").read_text())
    )
    model = create_vision_model(task, config, **options)
    state = load_file(str(checkpoint / "model.safetensors"), device=str(device))
    model.load_state_dict(state, strict=True)
    model.vision_task = task
    return model.to(device).eval()
