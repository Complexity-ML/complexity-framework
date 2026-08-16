"""Checkpoint diagnostics for CUHK-X cross-subject sensor fusion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from complexity.tr_hash import TRHashEngine

from .config import SENSOR_MODALITIES, TRHashSensorFusionConfig
from .cuhkx_records import CUHKXRecord, load_cuhkx_manifest
from .data import CUHKXSmallTrackDataset, collate_cuhkx
from .model import TRHashSensorFusionClassifier
from .preprocessing import CUHKXPreprocessingConfig


def _classification_summary(
    labels: torch.Tensor, predictions: torch.Tensor, num_classes: int
) -> dict:
    flat = labels.to(torch.int64) * num_classes + predictions.to(torch.int64)
    confusion = torch.bincount(flat, minlength=num_classes**2).reshape(
        num_classes,
        num_classes,
    )
    summary = classification_metrics_from_confusion(confusion)
    return {
        "top1_accuracy": summary["top1_accuracy"],
        "macro_accuracy": summary["macro_accuracy"],
        "examples": summary["examples"],
    }


def resolve_diagnostic_device(override: str | None = None) -> torch.device:
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_diagnostic_model(
    checkpoint: str | Path,
    device: torch.device,
) -> tuple[TRHashSensorFusionClassifier, CUHKXPreprocessingConfig]:
    checkpoint = Path(checkpoint)
    metadata = json.loads((checkpoint / "config.json").read_text())
    config = TRHashSensorFusionConfig.from_dict(metadata["model"])
    preprocessing = CUHKXPreprocessingConfig.from_dict(metadata["preprocessing"])
    model = TRHashSensorFusionClassifier(config)
    model.load_state_dict(load_file(str(checkpoint / "model.safetensors"), device="cpu"))
    model.to(device).eval()
    return model, preprocessing


def classification_metrics_from_confusion(confusion: torch.Tensor) -> dict:
    if confusion.ndim != 2 or confusion.size(0) != confusion.size(1):
        raise ValueError("confusion matrix must be square")
    confusion = confusion.to(dtype=torch.int64, device="cpu")
    examples = confusion.sum(dim=1)
    correct = confusion.diagonal()
    present = examples > 0
    per_class = torch.where(
        present,
        correct.double() / examples.clamp_min(1).double(),
        torch.zeros_like(correct, dtype=torch.float64),
    )
    total = int(examples.sum().item())
    return {
        "top1_accuracy": float(correct.sum().item() / max(total, 1)),
        "macro_accuracy": float(per_class[present].mean().item()) if present.any() else 0.0,
        "examples": total,
        "per_class_accuracy": per_class.tolist(),
        "per_class_examples": examples.tolist(),
        "confusion_matrix": confusion.tolist(),
    }


def hash_route_diagnostics(model: TRHashSensorFusionClassifier) -> dict:
    """Summarize fixed expert assignments, including class-modality routes."""

    layers = {}
    for name, module in model.named_modules():
        if not isinstance(module, TRHashEngine):
            continue
        table = module.route_table.detach().to(dtype=torch.int64, device="cpu")
        num_experts = int(module.config.num_experts)
        assignments = torch.bincount(table.flatten(), minlength=num_experts)
        total = max(int(assignments.sum().item()), 1)
        layers[name] = {
            "routes": int(table.size(1)),
            "top_k": int(table.size(0)),
            "expert_assignments": assignments.tolist(),
            "expert_shares": (assignments.double() / total).tolist(),
        }

    class_gate = model.class_modality_gate
    class_modality = None
    if class_gate is not None:
        table = class_gate.mlp.route_table.detach().to(
            dtype=torch.int64,
            device="cpu",
        )
        reshaped = table.reshape(
            table.size(0),
            len(SENSOR_MODALITIES),
            model.config.num_classes,
        )
        num_experts = model.config.num_experts
        by_modality = {}
        for index, modality in enumerate(SENSOR_MODALITIES):
            counts = torch.bincount(
                reshaped[:, index].flatten(),
                minlength=num_experts,
            )
            by_modality[modality] = counts.tolist()
        by_class = []
        for class_id in range(model.config.num_classes):
            counts = torch.bincount(
                reshaped[:, :, class_id].flatten(),
                minlength=num_experts,
            )
            by_class.append(counts.tolist())
        class_modality = {
            "route_table": table.tolist(),
            "expert_assignments_by_modality": by_modality,
            "expert_assignments_by_class": by_class,
        }
    return {
        "layers": layers,
        "class_modality_gate": class_modality,
    }


def _move_inputs(
    values: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device, non_blocking=True) for name, tensor in values.items()}


@torch.inference_mode()
def evaluate_sensor_mode(
    model: TRHashSensorFusionClassifier,
    loader: DataLoader,
    device: torch.device,
    *,
    modality: str | None,
    precision: str,
) -> dict:
    if modality is not None and modality not in SENSOR_MODALITIES:
        raise ValueError(f"unsupported modality: {modality}")
    confusion = torch.zeros(
        model.config.num_classes,
        model.config.num_classes,
        dtype=torch.int64,
        device=device,
    )
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    example_count = 0
    subject_confusions: dict[int, torch.Tensor] = {}
    autocast_enabled = precision == "bf16" and device.type in {"cuda", "cpu"}
    for batch in loader:
        inputs = _move_inputs(batch["inputs"], device)
        masks = _move_inputs(batch["modality_mask"], device)
        labels = batch["labels"].to(device, non_blocking=True)
        subject_ids = torch.tensor(
            [int(item["user_id"]) for item in batch["metadata"]],
            dtype=torch.int64,
            device=device,
        )
        if modality is not None:
            usable = masks[modality].bool()
            if not usable.any():
                continue
            inputs = {name: values[usable] for name, values in inputs.items()}
            labels = labels[usable]
            subject_ids = subject_ids[usable]
            masks = {
                name: (
                    values[usable]
                    if name == modality
                    else torch.zeros_like(values[usable], dtype=torch.bool)
                )
                for name, values in masks.items()
            }
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            output = model(inputs, labels, modality_mask=masks)
        predictions = output["logits"].argmax(dim=-1)
        flat = labels * model.config.num_classes + predictions
        confusion += torch.bincount(
            flat,
            minlength=model.config.num_classes**2,
        ).reshape_as(confusion)
        for subject_id in subject_ids.unique().tolist():
            selected = subject_ids == subject_id
            subject_flat = labels[selected] * model.config.num_classes + predictions[selected]
            subject_confusion = torch.bincount(
                subject_flat,
                minlength=model.config.num_classes**2,
            ).reshape_as(confusion)
            if subject_id not in subject_confusions:
                subject_confusions[subject_id] = torch.zeros_like(confusion)
            subject_confusions[subject_id] += subject_confusion
        count = int(labels.numel())
        loss_sum += output["loss"].double() * count
        example_count += count
    metrics = classification_metrics_from_confusion(confusion)
    metrics["validation_loss"] = float(loss_sum.item() / max(example_count, 1))
    metrics["mode"] = modality or "all"
    metrics["by_subject"] = {
        str(subject_id): classification_metrics_from_confusion(subject_confusion)
        for subject_id, subject_confusion in sorted(subject_confusions.items())
    }
    return metrics


@torch.inference_mode()
def evaluate_late_fusion_sweep(
    model: TRHashSensorFusionClassifier,
    loader: DataLoader,
    device: torch.device,
    *,
    precision: str,
    late_weights: Iterable[float] = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0),
    gate_temperatures: Iterable[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
    skeleton_boosts: Iterable[float] = (1.0, 2.0, 4.0, 8.0),
) -> dict:
    """Calibrate late fusion without changing checkpoint parameters."""

    late_weights = tuple(float(value) for value in late_weights)
    temperatures = tuple(float(value) for value in gate_temperatures)
    boosts = tuple(float(value) for value in skeleton_boosts)
    if not late_weights or min(late_weights) < 0.0:
        raise ValueError("late weights must be non-empty and non-negative")
    if not temperatures or min(temperatures) <= 0.0:
        raise ValueError("gate temperatures must be positive")
    if not boosts or min(boosts) <= 0.0:
        raise ValueError("skeleton boosts must be positive")

    labels_parts: list[torch.Tensor] = []
    fused_parts: list[torch.Tensor] = []
    modality_parts: list[torch.Tensor] = []
    gate_parts: list[torch.Tensor] = []
    autocast_enabled = precision == "bf16" and device.type in {"cuda", "cpu"}
    for batch in loader:
        inputs = _move_inputs(batch["inputs"], device)
        masks = _move_inputs(batch["modality_mask"], device)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            output = model(inputs, modality_mask=masks)
        labels_parts.append(labels.cpu())
        fused_parts.append(output["fused_logits"].float().cpu())
        modality_parts.append(output["modality_logits"].float().cpu())
        gate_parts.append(output["modality_weights"].float().cpu())

    labels = torch.cat(labels_parts)
    fused_logits = torch.cat(fused_parts)
    modality_logits = torch.cat(modality_parts)
    base_gates = torch.cat(gate_parts).clamp_min(1e-8)
    num_classes = model.config.num_classes
    skeleton_index = SENSOR_MODALITIES.index("skeleton")
    candidates = []
    seen = set()
    for late_weight in late_weights:
        for temperature in temperatures:
            for skeleton_boost in boosts:
                key = (
                    late_weight,
                    1.0 if late_weight == 0.0 else temperature,
                    1.0 if late_weight == 0.0 else skeleton_boost,
                )
                if key in seen:
                    continue
                seen.add(key)
                gates = base_gates.pow(1.0 / key[1])
                gates[:, skeleton_index] *= key[2]
                gates = gates / gates.sum(dim=1, keepdim=True).clamp_min(1e-8)
                if gates.ndim == 2:
                    gates = gates.unsqueeze(-1)
                late_logits = (modality_logits * gates).sum(dim=1)
                predictions = (fused_logits + key[0] * late_logits).argmax(dim=-1)
                metrics = _classification_summary(labels, predictions, num_classes)
                candidates.append(
                    {
                        "late_weight": key[0],
                        "gate_temperature": key[1],
                        "skeleton_boost": key[2],
                        **metrics,
                    }
                )
    candidates.sort(
        key=lambda item: (item["top1_accuracy"], item["macro_accuracy"]),
        reverse=True,
    )
    return {
        "checkpoint_late_weight": model.config.late_fusion_weight,
        "best": candidates[0],
        "candidates": candidates,
    }


def run_checkpoint_fusion_sweep(
    checkpoint: str | Path,
    *,
    data_root: str | Path,
    manifest: str | Path,
    validation_users: Iterable[int],
    batch_size: int = 4,
    workers: int = 4,
    device: str | None = None,
) -> dict:
    resolved_device = resolve_diagnostic_device(device)
    model, preprocessing = load_diagnostic_model(checkpoint, resolved_device)
    records = load_cuhkx_manifest(manifest)
    users = tuple(validation_users)
    dataset = CUHKXSmallTrackDataset(
        data_root,
        split="validation",
        preprocessing=preprocessing,
        validation_users=users,
        records=records,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        pin_memory=resolved_device.type == "cuda",
        persistent_workers=workers > 0,
        collate_fn=collate_cuhkx,
    )
    return {
        "checkpoint": str(Path(checkpoint)),
        "device": str(resolved_device),
        "validation_users": list(users),
        "preprocessing": preprocessing.to_dict(),
        "fusion_sweep": evaluate_late_fusion_sweep(
            model,
            loader,
            resolved_device,
            precision=model.config.precision,
        ),
    }


def run_checkpoint_diagnostics(
    checkpoint: str | Path,
    *,
    data_root: str | Path,
    manifest: str | Path,
    validation_users: Iterable[int],
    batch_size: int = 4,
    workers: int = 4,
    device: str | None = None,
    modalities: Iterable[str] = SENSOR_MODALITIES,
) -> dict:
    resolved_device = resolve_diagnostic_device(device)
    model, preprocessing = load_diagnostic_model(checkpoint, resolved_device)
    records: list[CUHKXRecord] = load_cuhkx_manifest(manifest)
    dataset = CUHKXSmallTrackDataset(
        data_root,
        split="validation",
        preprocessing=preprocessing,
        validation_users=tuple(validation_users),
        records=records,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        pin_memory=resolved_device.type == "cuda",
        persistent_workers=workers > 0,
        collate_fn=collate_cuhkx,
    )
    selected = tuple(modalities)
    unknown = sorted(set(selected) - set(SENSOR_MODALITIES))
    if unknown:
        raise ValueError(f"unsupported modalities: {unknown}")
    return {
        "checkpoint": str(Path(checkpoint)),
        "device": str(resolved_device),
        "validation_users": list(validation_users),
        "preprocessing": preprocessing.to_dict(),
        "hash_routes": hash_route_diagnostics(model),
        "results": {
            mode or "all": evaluate_sensor_mode(
                model,
                loader,
                resolved_device,
                modality=mode,
                precision=model.config.precision,
            )
            for mode in (None, *selected)
        },
    }
