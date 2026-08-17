"""Train TR-HASH Sensor Fusion on the CUHK-X Small Model Track."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader
from tqdm import tqdm

from complexity.training.musgd import MuSGD, build_musgd_parameter_groups, named_learning_rates

from ..detection.distributed import DistributedContext
from .checkpointing import load_sensor_fusion_checkpoint, save_sensor_fusion_checkpoint
from .config import SENSOR_MODALITIES, TRHashSensorFusionConfig
from .cuhkx_records import (
    CUHKX_TRAIN_USERS,
    build_cuhkx_records,
    load_cuhkx_manifest,
    save_cuhkx_manifest,
)
from .data import CUHKXSmallTrackDataset, collate_cuhkx
from .model import TRHashSensorFusionClassifier
from .preprocessing import CUHKXPreprocessingConfig
from .sampling import (
    EpochAnnotatedSampler,
    EpochRandomSampler,
    EpochWeightedSampler,
    ResumableBatchSampler,
)
from .transfer import load_pretrained_visual_tower

LOGGER = logging.getLogger("tr_hash_sensor_fusion")


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Extracted HAR or HAR/data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2, help="Per process/GPU")
    parser.add_argument("--eval-batch-size", type=int, default=2, help="Per process/GPU")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validation-users", type=int, nargs="+", default=(8, 9, 23, 24))
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--preprocessing-version", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument("--sensor-steps", type=int, default=64)
    parser.add_argument(
        "--vision-backbone-checkpoint",
        type=Path,
        default=None,
        help="Optional compatible pretrained TR-Hash vision tower to transfer into a fresh run",
    )
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--num-experts", type=int, choices=(8,), default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--shared-width", type=int, default=256)
    parser.add_argument("--expert-width", type=int, default=128)
    parser.add_argument("--sequence-tokens", type=int, default=32)
    parser.add_argument("--class-hash-shared-width", type=int, default=96)
    parser.add_argument("--class-hash-expert-width", type=int, default=24)
    parser.add_argument("--class-hash-initial-scale", type=float, default=0.05)
    parser.add_argument("--late-fusion-weight", type=float, default=0.5)
    parser.add_argument("--auxiliary-loss-weight", type=float, default=0.0)
    parser.add_argument("--gate-calibration-loss-weight", type=float, default=0.0)
    parser.add_argument("--gate-quality-temperature", type=float, default=1.0)
    parser.add_argument("--gate-target-smoothing", type=float, default=0.1)
    parser.add_argument("--contrastive-loss-weight", type=float, default=0.0)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--subject-adversarial-weight", type=float, default=0.0)
    parser.add_argument("--subject-adversarial-warmup-epochs", type=int, default=10)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--optimizer", choices=("musgd",), required=True)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--expert-lr-multiplier", type=float, default=1.5)
    parser.add_argument(
        "--backbone-lr-multiplier",
        type=float,
        default=None,
        help="Backbone LR multiplier. Defaults to 1.0; lower it when transferring a pretrained tower.",
    )
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--musgd-muon-weight", type=float, default=0.2)
    parser.add_argument("--musgd-sgd-weight", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--log-steps", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--visual-jitter", type=float, default=0.1)
    parser.add_argument(
        "--temporal-jitter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample different source frames deterministically at each epoch",
    )
    parser.add_argument("--visual-horizontal-flip", type=float, default=0.5)
    parser.add_argument("--visual-crop-jitter", type=float, default=0.1)
    parser.add_argument("--sensor-noise", type=float, default=0.01)
    parser.add_argument(
        "--modality-dropout",
        type=float,
        default=0.1,
        help="Probability of hiding each available modality during training",
    )
    parser.add_argument(
        "--clean-finetune-epochs",
        type=int,
        default=0,
        help=(
            "Final N epochs run with batch-level augmentation strength "
            "(mixup, visual jitter, sensor noise, modality dropout) linearly "
            "annealed to zero, mirroring TR-HASH Vision's noisy-pretrain -> "
            "clean full-parameter SFT recipe. 0 disables the anneal."
        ),
    )
    parser.add_argument(
        "--class-balance",
        choices=("none", "inverse-sqrt", "inverse"),
        default="inverse-sqrt",
        help="Training-only cross-entropy weighting for imbalanced HAR actions",
    )
    parser.add_argument(
        "--class-sampling",
        choices=("none", "inverse-sqrt", "inverse"),
        default="none",
        help="Training-only replacement sampling for long-tailed HAR actions",
    )
    parser.add_argument(
        "--sample-loss-weighting",
        choices=("none", "inverse-sqrt-2d", "inverse-2d", "inverse-sqrt-3d"),
        default="none",
        help=(
            "Full-shard per-example loss weighting by action and action-subject "
            "frequency; 3d also balances complete versus incomplete sensor groups"
        ),
    )
    parser.add_argument("--sample-loss-weight-min", type=float, default=0.0)
    parser.add_argument(
        "--sample-loss-weight-max",
        type=float,
        default=0.0,
        help="Zero disables the upper bound",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-drop-last", action="store_true")
    parser.add_argument(
        "--require-fused-cuda",
        action="store_true",
        help="Fail if any TR-HASH block falls back from the fused CUDA backend",
    )
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=0,
        help="Stop after N optimizer steps and save smoke_final (validation still runs)",
    )
    return parser.parse_args()


def resolve_device(override: str | None) -> torch.device:
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_backbone_lr_multiplier(requested: float | None) -> float:
    """Default to full backbone LR; lower it explicitly when transferring a pretrained tower."""

    if requested is not None:
        if requested <= 0.0:
            raise ValueError("--backbone-lr-multiplier must be positive")
        return float(requested)
    return 1.0


def seed_everything(seed: int, rank: int = 0) -> None:
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + rank)


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _autocast(device: torch.device, precision: str):
    if precision == "bf16" and device.type in {"cuda", "cpu"}:
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def _scheduler_lambda(step: int, *, warmup: int, total: int, minimum: float) -> float:
    if warmup > 0 and step < warmup:
        return max((step + 1) / warmup, 1e-8)
    progress = (step - warmup) / max(total - warmup, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return minimum + (1.0 - minimum) * cosine


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                state[name] = value.to(device)


def _training_options(args: argparse.Namespace, world_size: int) -> dict[str, Any]:
    names = (
        "batch_size",
        "lr",
        "expert_lr_multiplier",
        "backbone_lr_multiplier",
        "momentum",
        "weight_decay",
        "musgd_muon_weight",
        "musgd_sgd_weight",
        "warmup_steps",
        "min_lr_ratio",
        "grad_clip",
        "label_smoothing",
        "mixup_alpha",
        "visual_jitter",
        "temporal_jitter",
        "visual_horizontal_flip",
        "visual_crop_jitter",
        "sensor_noise",
        "modality_dropout",
        "clean_finetune_epochs",
        "class_balance",
        "class_sampling",
        "sample_loss_weighting",
        "sample_loss_weight_min",
        "sample_loss_weight_max",
        "late_fusion_weight",
        "class_hash_shared_width",
        "class_hash_expert_width",
        "class_hash_initial_scale",
        "auxiliary_loss_weight",
        "gate_calibration_loss_weight",
        "gate_quality_temperature",
        "gate_target_smoothing",
        "contrastive_loss_weight",
        "contrastive_temperature",
        "subject_adversarial_weight",
        "subject_adversarial_warmup_epochs",
        "seed",
        "no_drop_last",
        "require_fused_cuda",
    )
    result = {name: getattr(args, name) for name in names}
    result["validation_users"] = list(args.validation_users)
    result["world_size"] = world_size
    return result


def augmentation_scale(epoch: int, *, total_epochs: int, clean_finetune_epochs: int) -> float:
    """1.0 through the noisy-pretrain phase, linearly decaying to 0.0 across
    the final ``clean_finetune_epochs`` -- the analogue of TR-HASH Vision
    v8's noisy-pretrain -> clean full-parameter SFT recipe. Used both for
    batch-level augmentation (mixup/jitter/noise/modality-dropout, applied
    directly in the training loop) and, when workers > 0, for dataset-side
    augmentation (flip/crop-jitter/training_augmentation) via a separate
    non-persistent-worker loader (clean_finetune_train_loader) swapped in
    for exactly the epochs inside this window -- persistent workers pickle
    the Dataset once and never see attribute mutations made afterwards."""

    if clean_finetune_epochs <= 0 or total_epochs <= 0:
        return 1.0
    anneal_start = max(total_epochs - clean_finetune_epochs, 0)
    if epoch < anneal_start:
        return 1.0
    progress = (epoch - anneal_start) / max(clean_finetune_epochs, 1)
    return max(0.0, 1.0 - min(progress, 1.0))


def should_build_clean_finetune_train_loader(*, workers: int, clean_finetune_epochs: int) -> bool:
    """A separate non-persistent-worker loader is only needed when there both
    are workers to be stale in the first place and an anneal window that
    needs their per-epoch mutations to land."""

    return workers > 0 and clean_finetune_epochs > 0


def clean_finetune_anneal_start(*, total_epochs: int, clean_finetune_epochs: int) -> int:
    """First epoch (0-indexed) of the clean-finetune anneal window; mirrors
    augmentation_scale's own boundary so the loader swap and the augmentation
    scale agree on exactly when the window begins."""

    if clean_finetune_epochs <= 0 or total_epochs <= 0:
        return total_epochs
    return max(total_epochs - clean_finetune_epochs, 0)


def resolve_epoch_training_augmentation(
    *, base_enabled: bool, aug_scale: float, seed: int, epoch: int
) -> bool:
    """Stochastically gate the boolean training_augmentation switch (which
    drives temporal-jitter frame sampling) by aug_scale: always on during the
    noisy-pretrain phase (aug_scale=1), increasingly likely to fall back to
    deterministic frame sampling as the clean-finetune window anneals
    aug_scale toward 0."""

    if not base_enabled:
        return False
    return random.Random(seed + epoch).random() < aug_scale


def apply_modality_dropout(
    masks: Mapping[str, torch.Tensor],
    probability: float,
) -> dict[str, torch.Tensor]:
    if not 0.0 <= probability < 1.0:
        raise ValueError("modality_dropout must be in [0, 1)")
    output = {name: value.clone() for name, value in masks.items()}
    if probability == 0.0:
        return output
    names = tuple(output)
    present = torch.stack([output[name] for name in names], dim=1)
    keep = torch.rand(present.shape, device=present.device) >= probability
    dropped = present & keep
    empty = ~dropped.any(dim=1)
    if empty.any():
        # Preserve the first genuinely available stream for samples where the
        # random draw hid every modality.
        first = present.to(torch.int64).argmax(dim=1)
        dropped[empty, first[empty]] = True
    return {name: dropped[:, index] for index, name in enumerate(names)}


def augment_sensor_inputs(
    inputs: Mapping[str, torch.Tensor],
    *,
    visual_jitter: float,
    sensor_noise: float,
) -> dict[str, torch.Tensor]:
    if visual_jitter < 0.0 or sensor_noise < 0.0:
        raise ValueError("augmentation strengths must be non-negative")
    output = {}
    for name, values in inputs.items():
        if values.ndim == 5 and visual_jitter > 0.0:
            shape = (values.size(0), 1, 1, 1, 1)
            gain = 1.0 + (2.0 * torch.rand(shape, device=values.device) - 1.0) * visual_jitter
            offset = (2.0 * torch.rand(shape, device=values.device) - 1.0) * (0.5 * visual_jitter)
            output[name] = (values * gain + offset).clamp(-1.0, 1.0)
        elif values.ndim in (3, 4) and sensor_noise > 0.0:
            output[name] = values + sensor_noise * torch.randn_like(values)
        else:
            output[name] = values
    return output


def mixup_sensor_batch(
    inputs: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    alpha: float,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    torch.Tensor,
    float,
    torch.Tensor,
]:
    if alpha < 0.0:
        raise ValueError("mixup_alpha must be non-negative")
    if alpha == 0.0 or labels.numel() < 2:
        identity = torch.arange(labels.numel(), device=labels.device)
        return dict(inputs), dict(masks), labels, 1.0, identity
    beta = torch.distributions.Beta(alpha, alpha)
    weight = float(beta.sample().item())
    permutation = torch.randperm(labels.numel(), device=labels.device)
    mixed_inputs = {
        name: weight * values + (1.0 - weight) * values[permutation]
        for name, values in inputs.items()
    }
    mixed_masks = {name: values | values[permutation] for name, values in masks.items()}
    return mixed_inputs, mixed_masks, labels[permutation], weight, permutation


def _build_manifest(args: argparse.Namespace, context: DistributedContext) -> list:
    manifest = args.manifest or args.output / "cuhkx_manifest.json"
    if context.is_main and not manifest.is_file():
        LOGGER.info("Indexing CUHK-X trials under %s", args.data_root)
        records = build_cuhkx_records(args.data_root)
        save_cuhkx_manifest(records, manifest)
        LOGGER.info("Indexed %d synchronized trials in %s", len(records), manifest)
    context.barrier()
    return load_cuhkx_manifest(manifest)


def build_class_weights(
    records: list,
    *,
    num_classes: int,
    mode: str,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    counts = torch.bincount(
        torch.tensor([record.action_id for record in records], dtype=torch.long),
        minlength=num_classes,
    )
    present = counts > 0
    if not present.any():
        raise ValueError("cannot build class weights from an empty training split")
    if mode == "none":
        return None, counts
    if mode == "inverse-sqrt":
        weights = torch.zeros_like(counts, dtype=torch.float32)
        weights[present] = counts[present].float().rsqrt()
    elif mode == "inverse":
        weights = torch.zeros_like(counts, dtype=torch.float32)
        weights[present] = counts[present].float().reciprocal()
    else:
        raise ValueError(f"unsupported class balance mode: {mode}")
    weights[present] /= weights[present].mean()
    return weights.to(device), counts


def build_sampling_weights(records: list, *, num_classes: int, mode: str) -> torch.Tensor:
    labels = torch.tensor([record.action_id for record in records], dtype=torch.long)
    counts = torch.bincount(labels, minlength=num_classes)
    if mode == "none":
        return torch.ones(labels.numel(), dtype=torch.float64)
    present = counts > 0
    class_weights = torch.zeros_like(counts, dtype=torch.float64)
    if mode == "inverse-sqrt":
        class_weights[present] = counts[present].double().rsqrt()
    elif mode == "inverse":
        class_weights[present] = counts[present].double().reciprocal()
    else:
        raise ValueError(f"unsupported class sampling mode: {mode}")
    return class_weights[labels]


def _normalize_bounded_weights(
    weights: torch.Tensor,
    *,
    minimum: float,
    maximum: float,
) -> torch.Tensor:
    """Return mean-one weights while respecting optional hard bounds."""

    weights = weights / weights.mean()
    if minimum <= 0.0 and maximum <= 0.0:
        return weights
    lower = max(0.0, float(minimum))
    upper = float(maximum) if maximum > 0.0 else float("inf")
    low, high = 0.0, 1.0
    while torch.clamp(weights * high, min=lower, max=upper).mean() < 1.0:
        high *= 2.0
    for _ in range(64):
        middle = 0.5 * (low + high)
        candidate = torch.clamp(weights * middle, min=lower, max=upper)
        if candidate.mean() < 1.0:
            low = middle
        else:
            high = middle
    return torch.clamp(weights * high, min=lower, max=upper)


def build_sample_loss_weights(
    records: list,
    *,
    num_classes: int,
    mode: str,
    minimum: float = 0.0,
    maximum: float = 0.0,
) -> torch.Tensor:
    """Weight every full-shard row by action and nested action-subject frequency."""

    if mode == "none":
        return torch.ones(len(records), dtype=torch.float64)
    if mode not in {"inverse-sqrt-2d", "inverse-2d", "inverse-sqrt-3d"}:
        raise ValueError(f"unsupported sample loss weighting mode: {mode}")
    labels = torch.tensor([record.action_id for record in records], dtype=torch.long)
    class_frequency = torch.bincount(labels, minlength=num_classes).double()[labels]
    pair_counts: dict[tuple[int, int], int] = {}
    for record in records:
        key = (int(record.action_id), int(record.user_id))
        pair_counts[key] = pair_counts.get(key, 0) + 1
    pair_frequency = torch.tensor(
        [pair_counts[(int(record.action_id), int(record.user_id))] for record in records],
        dtype=torch.float64,
    )
    if mode in {"inverse-sqrt-2d", "inverse-sqrt-3d"}:
        weights = class_frequency.rsqrt() * pair_frequency.rsqrt()
    else:
        weights = class_frequency.reciprocal() * pair_frequency.reciprocal()
    if mode == "inverse-sqrt-3d":
        completeness_counts: dict[tuple[int, bool], int] = {}
        completeness = []
        for record in records:
            complete = all(name in record.paths for name in SENSOR_MODALITIES)
            key = (int(record.action_id), complete)
            completeness.append(key)
            completeness_counts[key] = completeness_counts.get(key, 0) + 1
        completeness_frequency = torch.tensor(
            [completeness_counts[key] for key in completeness],
            dtype=torch.float64,
        )
        weights = weights * completeness_frequency.rsqrt()
    return _normalize_bounded_weights(
        weights,
        minimum=minimum,
        maximum=maximum,
    )


def weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor,
    *,
    class_weights: torch.Tensor | None,
    label_smoothing: float,
) -> torch.Tensor:
    losses = F.cross_entropy(
        logits,
        labels,
        weight=class_weights,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    weights = sample_weights.to(device=losses.device, dtype=losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-12)


def auxiliary_modality_loss(
    modality_logits: torch.Tensor,
    modality_mask: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    *,
    class_weights: torch.Tensor | None,
    label_smoothing: float,
) -> torch.Tensor:
    if modality_logits.ndim != 3:
        raise ValueError("modality_logits must have shape [batch, modality, class]")
    available = torch.stack(
        tuple(modality_mask[name] for name in SENSOR_MODALITIES),
        dim=1,
    ).bool()
    if available.shape != modality_logits.shape[:2]:
        raise ValueError("modality mask and logits shapes do not match")
    expanded_labels = labels[:, None].expand_as(available)
    losses = F.cross_entropy(
        modality_logits[available],
        expanded_labels[available],
        weight=class_weights,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    if sample_weights is None:
        return losses.mean()
    expanded_weights = sample_weights[:, None].expand_as(available)[available]
    return (losses * expanded_weights).sum() / expanded_weights.sum().clamp_min(1e-12)


def modality_gate_calibration_loss(
    modality_logits: torch.Tensor,
    modality_weights: torch.Tensor,
    modality_mask: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    *,
    quality_temperature: float,
    target_smoothing: float,
    label_smoothing: float,
) -> torch.Tensor:
    """Calibrate learned gates against detached per-modality prediction quality.

    Each available modality receives a soft target proportional to the inverse
    of its classification loss for the current example. The target is detached
    so modality heads cannot improve the routing objective by manipulating its
    scale, while a small uniform component prevents single-modality collapse.
    """

    if modality_logits.ndim != 3:
        raise ValueError("modality_logits must have shape [batch, modality, class]")
    expected_weight_shapes = {
        modality_logits.shape[:2],
        modality_logits.shape,
    }
    if modality_weights.shape not in expected_weight_shapes:
        raise ValueError("modality weights and logits shapes do not match")
    if labels.shape != (modality_logits.size(0),):
        raise ValueError("labels must have shape [batch]")
    if quality_temperature <= 0.0:
        raise ValueError("gate quality temperature must be positive")
    if not 0.0 <= target_smoothing < 1.0:
        raise ValueError("gate target smoothing must be in [0, 1)")

    available = torch.stack(
        tuple(modality_mask[name] for name in SENSOR_MODALITIES),
        dim=1,
    ).bool()
    if available.shape != modality_logits.shape[:2]:
        raise ValueError("modality mask and logits shapes do not match")
    if not available.any(dim=1).all():
        raise ValueError("every example must expose at least one modality")

    batch_size, num_modalities, num_classes = modality_logits.shape
    expanded_labels = labels[:, None].expand(batch_size, num_modalities)
    quality_losses = F.cross_entropy(
        modality_logits.float().reshape(-1, num_classes),
        expanded_labels.reshape(-1),
        label_smoothing=label_smoothing,
        reduction="none",
    ).reshape(batch_size, num_modalities)
    quality_scores = (-quality_losses / quality_temperature).masked_fill(
        ~available,
        -torch.inf,
    )
    quality_targets = quality_scores.softmax(dim=1).detach()
    if target_smoothing > 0.0:
        uniform_targets = available.float()
        uniform_targets = uniform_targets / uniform_targets.sum(dim=1, keepdim=True)
        quality_targets = (
            1.0 - target_smoothing
        ) * quality_targets + target_smoothing * uniform_targets

    selected_weights = modality_weights
    if modality_weights.ndim == 3:
        selected_weights = modality_weights.gather(
            2,
            labels[:, None, None].expand(-1, num_modalities, 1),
        ).squeeze(-1)
    log_gate_weights = selected_weights.float().clamp_min(1e-8).log()
    per_example = -(quality_targets * log_gate_weights).sum(dim=1)
    if sample_weights is None:
        return per_example.mean()
    weights = sample_weights.to(device=per_example.device, dtype=per_example.dtype)
    return (per_example * weights).sum() / weights.sum().clamp_min(1e-12)


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
    context: DistributedContext | None = None,
) -> torch.Tensor:
    """Pull same-action clips together using all DDP ranks as candidates."""

    if features.ndim != 2:
        raise ValueError("contrastive features must have shape [batch, hidden]")
    if labels.shape != (features.size(0),):
        raise ValueError("contrastive labels must have shape [batch]")
    if temperature <= 0.0:
        raise ValueError("contrastive temperature must be positive")
    anchors = F.normalize(features.float(), dim=-1)
    candidate_features = anchors
    candidate_labels = labels
    rank = 0
    if context is not None and context.enabled:
        gathered_features = dist_nn.all_gather(anchors)
        candidate_features = torch.cat(tuple(gathered_features), dim=0)
        gathered_labels = [torch.empty_like(labels) for _ in range(context.world_size)]
        dist.all_gather(gathered_labels, labels)
        candidate_labels = torch.cat(gathered_labels, dim=0)
        rank = context.rank

    logits = anchors @ candidate_features.transpose(0, 1)
    logits = logits / temperature
    candidate_mask = torch.ones_like(logits, dtype=torch.bool)
    self_indices = rank * anchors.size(0) + torch.arange(
        anchors.size(0),
        device=anchors.device,
    )
    candidate_mask[torch.arange(anchors.size(0), device=anchors.device), self_indices] = False
    positive_mask = labels[:, None].eq(candidate_labels[None, :]) & candidate_mask
    positive_count = positive_mask.sum(dim=1)
    valid_anchors = positive_count > 0
    if not valid_anchors.any():
        # Every DDP rank must traverse the autograd-aware all-gather during
        # backward. Returning a zero connected only to the local features
        # deadlocks the ranks that do have positive pairs in this step.
        return candidate_features.sum() * 0.0

    logits = (
        logits
        - logits.masked_fill(~candidate_mask, -torch.inf)
        .max(
            dim=1,
            keepdim=True,
        )
        .values.detach()
    )
    exp_logits = logits.exp() * candidate_mask
    log_probabilities = logits - exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12).log()
    positive_log_probabilities = torch.where(
        positive_mask,
        log_probabilities,
        torch.zeros_like(log_probabilities),
    )
    mean_positive_log_probability = positive_log_probabilities.sum(
        dim=1
    ) / positive_count.clamp_min(1)
    return -mean_positive_log_probability[valid_anchors].mean()


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    precision: str,
    context: DistributedContext,
) -> dict[str, float]:
    model.eval()
    raw_model = model.module if hasattr(model, "module") else model
    num_classes = raw_model.config.num_classes
    num_modalities = len(SENSOR_MODALITIES)
    base_size = 4 + 2 * num_classes
    totals = torch.zeros(
        base_size + 3 * num_modalities,
        dtype=torch.float64,
        device=device,
    )
    class_gate_totals = torch.zeros(
        num_classes,
        num_modalities,
        dtype=torch.float64,
        device=device,
    )
    for batch in loader:
        inputs = _move(batch["inputs"], device)
        masks = _move(batch["modality_mask"], device)
        labels = batch["labels"].to(device, non_blocking=True)
        with _autocast(device, precision):
            output = model(inputs, labels, modality_mask=masks)
        count = labels.numel()
        predictions = output["logits"].topk(min(5, output["logits"].size(-1)), dim=-1).indices
        totals[0] += output["loss"].detach().double() * count
        totals[1] += predictions[:, 0].eq(labels).sum()
        totals[2] += predictions.eq(labels[:, None]).any(dim=1).sum()
        totals[3] += count
        totals[4 : 4 + num_classes].scatter_add_(
            0,
            labels,
            predictions[:, 0].eq(labels).to(torch.float64),
        )
        totals[4 + num_classes : 4 + 2 * num_classes].scatter_add_(
            0,
            labels,
            torch.ones_like(labels, dtype=torch.float64),
        )
        if num_modalities:
            available = torch.stack(
                tuple(masks[name] for name in SENSOR_MODALITIES),
                dim=1,
            ).bool()
            modality_predictions = output["modality_logits"].argmax(dim=-1)
            gate_weights = output["modality_weights"]
            if gate_weights.ndim == 3:
                gate_weights = gate_weights.gather(
                    2,
                    labels[:, None, None].expand(-1, num_modalities, 1),
                ).squeeze(-1)
                class_gate_totals.index_add_(0, labels, gate_weights.double())
            totals[base_size : base_size + num_modalities] += gate_weights.double().sum(dim=0)
            totals[base_size + num_modalities : base_size + 2 * num_modalities] += (
                modality_predictions.eq(labels[:, None]) & available
            ).sum(dim=0)
            totals[base_size + 2 * num_modalities :] += available.sum(dim=0)
    if context.enabled:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        dist.all_reduce(class_gate_totals, op=dist.ReduceOp.SUM)
    denominator = max(float(totals[3].item()), 1.0)
    class_correct = totals[4 : 4 + num_classes]
    class_count = totals[4 + num_classes : 4 + 2 * num_classes]
    present = class_count > 0
    per_class = torch.where(
        present,
        class_correct / class_count.clamp_min(1.0),
        torch.zeros_like(class_correct),
    )
    metrics = {
        "validation_loss": float(totals[0].item() / denominator),
        "top1_accuracy": float(totals[1].item() / denominator),
        "top5_accuracy": float(totals[2].item() / denominator),
        "macro_accuracy": float(per_class[present].mean().item()),
        "per_class_accuracy": per_class.cpu().tolist(),
        "per_class_examples": class_count.long().cpu().tolist(),
        "validation_examples": int(totals[3].item()),
    }
    if num_modalities:
        modality_counts = totals[base_size + 2 * num_modalities :]
        modality_accuracy = totals[
            base_size + num_modalities : base_size + 2 * num_modalities
        ] / modality_counts.clamp_min(1.0)
        metrics.update(
            {
                "modality_gate_weights": (
                    totals[base_size : base_size + num_modalities] / denominator
                )
                .cpu()
                .tolist(),
                "modality_accuracy": modality_accuracy.cpu().tolist(),
                "modality_examples": modality_counts.long().cpu().tolist(),
            }
        )
        metrics["modality_gate_weights_by_class"] = (
            (class_gate_totals / class_count[:, None].clamp_min(1.0)).cpu().tolist()
        )
    return metrics


def _save(
    path: Path,
    raw_model: TRHashSensorFusionClassifier,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    context: DistributedContext,
    *,
    preprocessing: CUHKXPreprocessingConfig,
    epoch: int,
    batch_in_epoch: int,
    step: int,
    best_accuracy: float,
    total_epochs: int,
    steps_per_epoch: int,
    training_options: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> None:
    rng_states = context.gather_rng_states()
    if context.is_main:
        save_sensor_fusion_checkpoint(
            path,
            raw_model,
            optimizer,
            scheduler,
            model_config=raw_model.config,
            preprocessing=preprocessing,
            epoch=epoch,
            batch_in_epoch=batch_in_epoch,
            step=step,
            best_accuracy=best_accuracy,
            total_epochs=total_epochs,
            steps_per_epoch=steps_per_epoch,
            training_options=training_options,
            metrics=metrics,
            distributed_rng_states=rng_states,
        )
        LOGGER.info("Checkpoint saved: %s", path)
    context.barrier()


def main() -> None:
    args = parse_args()
    args.backbone_lr_multiplier = resolve_backbone_lr_multiplier(args.backbone_lr_multiplier)
    if args.auxiliary_loss_weight < 0.0:
        raise ValueError("--auxiliary-loss-weight must be non-negative")
    if args.gate_calibration_loss_weight < 0.0:
        raise ValueError("--gate-calibration-loss-weight must be non-negative")
    if args.gate_calibration_loss_weight > 0.0 and args.auxiliary_loss_weight <= 0.0:
        raise ValueError(
            "--gate-calibration-loss-weight requires --auxiliary-loss-weight > 0 "
            "so modality quality targets remain meaningful"
        )
    if args.gate_quality_temperature <= 0.0:
        raise ValueError("--gate-quality-temperature must be positive")
    if not 0.0 <= args.gate_target_smoothing < 1.0:
        raise ValueError("--gate-target-smoothing must be in [0, 1)")
    if args.contrastive_loss_weight < 0.0:
        raise ValueError("--contrastive-loss-weight must be non-negative")
    if args.contrastive_temperature <= 0.0:
        raise ValueError("--contrastive-temperature must be positive")
    if args.contrastive_loss_weight > 0.0 and args.mixup_alpha > 0.0:
        raise ValueError(
            "supervised contrastive training requires --mixup-alpha 0 so each "
            "representation has one unambiguous action label"
        )
    if args.subject_adversarial_weight < 0.0:
        raise ValueError("--subject-adversarial-weight must be non-negative")
    if args.subject_adversarial_warmup_epochs < 0:
        raise ValueError("--subject-adversarial-warmup-epochs must be non-negative")
    if args.subject_adversarial_weight > 0.0 and args.mixup_alpha > 0.0:
        raise ValueError("subject-adversarial training requires --mixup-alpha 0")
    if args.clean_finetune_epochs < 0:
        raise ValueError("--clean-finetune-epochs must be non-negative")
    if args.clean_finetune_epochs > args.epochs:
        raise ValueError("--clean-finetune-epochs cannot exceed --epochs")
    if args.sample_loss_weighting != "none" and args.class_sampling != "none":
        raise ValueError("full-shard sample loss weighting cannot use replacement class sampling")
    if args.sample_loss_weighting != "none" and args.class_balance != "none":
        raise ValueError(
            "sample loss weighting already balances classes; --class-balance must be none"
        )
    if not 0.0 <= args.sample_loss_weight_min <= 1.0:
        raise ValueError("--sample-loss-weight-min must be between 0 and 1")
    if args.sample_loss_weight_max != 0.0 and args.sample_loss_weight_max < 1.0:
        raise ValueError("--sample-loss-weight-max must be zero or at least 1")
    if (
        args.sample_loss_weight_max > 0.0
        and args.sample_loss_weight_min > args.sample_loss_weight_max
    ):
        raise ValueError("sample loss weight minimum exceeds its maximum")
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    LOGGER.handlers[:] = [handler]
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False

    context = DistributedContext.initialize(resolve_device(args.device))
    device = context.device
    seed_everything(args.seed, context.rank)
    try:
        records = _build_manifest(args, context)
        preprocessing = CUHKXPreprocessingConfig(
            version=args.preprocessing_version,
            image_size=args.image_size,
            clip_frames=args.clip_frames,
            sensor_steps=args.sensor_steps,
        )
        sample_augmentation = (
            args.temporal_jitter
            or args.visual_horizontal_flip > 0.0
            or args.visual_crop_jitter > 0.0
        )
        train_dataset = CUHKXSmallTrackDataset(
            args.data_root,
            split="train",
            preprocessing=preprocessing,
            validation_users=tuple(args.validation_users),
            records=records,
            training_augmentation=sample_augmentation,
            seed=args.seed,
            visual_horizontal_flip=args.visual_horizontal_flip,
            visual_crop_jitter=args.visual_crop_jitter,
        )
        sample_loss_weights = build_sample_loss_weights(
            train_dataset.records,
            num_classes=40,
            mode=args.sample_loss_weighting,
            minimum=args.sample_loss_weight_min,
            maximum=args.sample_loss_weight_max,
        )
        train_dataset.set_loss_weights(sample_loss_weights)
        validation_dataset = CUHKXSmallTrackDataset(
            args.data_root,
            split="validation",
            preprocessing=preprocessing,
            validation_users=tuple(args.validation_users),
            records=records,
        )
        if not train_dataset or not validation_dataset:
            raise ValueError("cross-subject train and validation splits must both be non-empty")

        if args.class_sampling == "none":
            train_sampler = context.train_sampler(train_dataset, args.seed)
            if train_sampler is None:
                train_sampler = EpochRandomSampler(train_dataset, args.seed)
        else:
            sample_weights = build_sampling_weights(
                train_dataset.records,
                num_classes=40,
                mode=args.class_sampling,
            )
            train_sampler = EpochWeightedSampler(
                sample_weights,
                args.seed,
                rank=context.rank,
                world_size=context.world_size,
            )
        if sample_augmentation:
            train_sampler = EpochAnnotatedSampler(train_sampler)
        base_batch_sampler = BatchSampler(
            train_sampler,
            args.batch_size,
            drop_last=not args.no_drop_last,
        )
        batch_sampler = ResumableBatchSampler(base_batch_sampler)
        loader_options = {
            "num_workers": args.workers,
            "pin_memory": device.type == "cuda",
            "collate_fn": collate_cuhkx,
        }
        if args.workers > 0:
            loader_options["persistent_workers"] = True
            loader_options["prefetch_factor"] = 2
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            **loader_options,
        )
        clean_finetune_train_loader = None
        if should_build_clean_finetune_train_loader(
            workers=args.workers, clean_finetune_epochs=args.clean_finetune_epochs
        ):
            # Persistent workers pickle the Dataset once at pool creation and
            # never see later attribute mutations, so the clean-finetune
            # anneal (which mutates visual_horizontal_flip/visual_crop_jitter/
            # training_augmentation on train_dataset every epoch) needs fresh
            # workers each epoch -- but only for the epochs actually inside
            # the anneal window. Using non-persistent workers for the whole
            # run would also lose data.py's imu/radar/skeleton parse-once
            # cache (_static_sensor_cache), which relies on the worker
            # process staying alive across epochs, and re-parse those
            # deterministic modalities from disk every epoch instead of once.
            clean_loader_options = dict(loader_options)
            clean_loader_options["persistent_workers"] = False
            clean_finetune_train_loader = DataLoader(
                train_dataset,
                batch_sampler=batch_sampler,
                **clean_loader_options,
            )
        clean_finetune_anneal_start_epoch = clean_finetune_anneal_start(
            total_epochs=args.epochs, clean_finetune_epochs=args.clean_finetune_epochs
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.eval_batch_size,
            sampler=context.eval_sampler(validation_dataset),
            shuffle=False,
            drop_last=False,
            **loader_options,
        )
        steps_per_epoch = len(base_batch_sampler)
        if steps_per_epoch == 0:
            raise ValueError("batch size is larger than the per-rank training split")

        model_config = TRHashSensorFusionConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.layers,
            num_attention_heads=args.heads,
            num_experts=args.num_experts,
            top_k=args.top_k,
            shared_width=args.shared_width,
            expert_width=args.expert_width,
            sequence_tokens=args.sequence_tokens,
            class_hash_shared_width=args.class_hash_shared_width,
            class_hash_expert_width=args.class_hash_expert_width,
            class_hash_initial_scale=args.class_hash_initial_scale,
            late_fusion_weight=args.late_fusion_weight,
            precision=args.precision,
        )
        raw_model = TRHashSensorFusionClassifier(model_config).to(device)
        if args.vision_backbone_checkpoint is not None and args.resume is None:
            transferred = load_pretrained_visual_tower(
                raw_model,
                args.vision_backbone_checkpoint,
            )
            if context.is_main:
                LOGGER.info(
                    "Loaded %d compact pretrained visual-tower tensors from %s",
                    transferred,
                    args.vision_backbone_checkpoint,
                )
        class_weights, class_counts = build_class_weights(
            train_dataset.records,
            num_classes=model_config.num_classes,
            mode=args.class_balance,
            device=device,
        )

        def learning_rate(name: str) -> tuple[float, str]:
            is_backbone = name.startswith(
                (
                    "vision_tower.",
                    "resnet_visual_backbone.",
                )
            )
            is_expert = ".mlp.expert_" in name
            multiplier = args.backbone_lr_multiplier if is_backbone else 1.0
            multiplier *= args.expert_lr_multiplier if is_expert else 1.0
            if is_backbone and is_expert:
                return args.lr * multiplier, "backbone_expert"
            if is_backbone:
                return args.lr * multiplier, "backbone"
            if is_expert:
                return args.lr * multiplier, "expert"
            return args.lr, "base"

        groups = build_musgd_parameter_groups(
            raw_model,
            learning_rate=learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        optimizer = MuSGD(
            groups,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
            muon_weight=args.musgd_muon_weight,
            sgd_weight=args.musgd_sgd_weight,
        )
        total_steps = args.epochs * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda current: _scheduler_lambda(
                current,
                warmup=args.warmup_steps,
                total=total_steps,
                minimum=args.min_lr_ratio,
            ),
        )
        options = _training_options(args, context.world_size)
        start_epoch = start_batch = step = 0
        best_accuracy = -1.0
        if args.resume is not None:
            state = load_sensor_fusion_checkpoint(
                args.resume,
                raw_model,
                optimizer,
                scheduler,
                model_config=model_config,
                preprocessing=preprocessing,
                total_epochs=args.epochs,
                steps_per_epoch=steps_per_epoch,
                training_options=options,
                rank=context.rank,
                world_size=context.world_size,
                device=device,
            )
            _optimizer_to(optimizer, device)
            start_epoch = int(state["epoch"])
            start_batch = int(state["batch_in_epoch"])
            step = int(state["step"])
            best_accuracy = float(state["best_accuracy"])
            if context.is_main:
                LOGGER.info(
                    "Resumed exactly from %s: epoch=%d batch=%d step=%d",
                    args.resume,
                    start_epoch,
                    start_batch,
                    step,
                )
        model = context.wrap(raw_model)

        if context.is_main:
            LOGGER.info(
                "CUHK-X: %d train, %d validation, users=%s",
                len(train_dataset),
                len(validation_dataset),
                ",".join(map(str, args.validation_users)),
            )
            LOGGER.info(
                "Model: %.2fM parameters, 8 experts top-%d, device=%s, world=%d",
                raw_model.num_parameters() / 1e6,
                args.top_k,
                device,
                context.world_size,
            )
            LOGGER.info("Optimizer: MuSGD lrs=%s", named_learning_rates(optimizer))
            LOGGER.info(
                "Class balance: loss=%s sampling=%s sample_loss=%s "
                "(class_count=%d..%d sample_weight=%.3f..%.3f)",
                args.class_balance,
                args.class_sampling,
                args.sample_loss_weighting,
                int(class_counts.min()),
                int(class_counts.max()),
                float(sample_loss_weights.min()),
                float(sample_loss_weights.max()),
            )

        last_metrics: dict[str, Any] = {}
        user_to_subject_index = {user: index for index, user in enumerate(CUHKX_TRAIN_USERS)}
        started = time.monotonic()
        stop = False
        backend_logged = False
        cursor_epoch, cursor_batch = start_epoch, start_batch
        for epoch in range(start_epoch, args.epochs):
            aug_scale = augmentation_scale(
                epoch,
                total_epochs=args.epochs,
                clean_finetune_epochs=args.clean_finetune_epochs,
            )
            epoch_train_loader = train_loader
            in_clean_finetune_window = epoch >= clean_finetune_anneal_start_epoch
            if in_clean_finetune_window:
                # Dataset-side augmentation (flip/crop-jitter/temporal-jitter)
                # can only be annealed by mutating the Dataset before this
                # epoch's non-persistent workers spawn (clean_finetune_train_loader
                # above) -- there is no batch-level equivalent for temporal
                # jitter, since it decides which frames get read off disk in
                # the first place, and crop jitter is applied against the
                # source image before resize. Outside this window, train_loader
                # keeps its persistent workers (and data.py's parse-once
                # sensor cache) untouched.
                train_dataset.visual_horizontal_flip = args.visual_horizontal_flip * aug_scale
                train_dataset.visual_crop_jitter = args.visual_crop_jitter * aug_scale
                train_dataset.training_augmentation = resolve_epoch_training_augmentation(
                    base_enabled=sample_augmentation,
                    aug_scale=aug_scale,
                    seed=args.seed,
                    epoch=epoch,
                )
                if clean_finetune_train_loader is not None:
                    epoch_train_loader = clean_finetune_train_loader
            if context.is_main and args.clean_finetune_epochs > 0:
                LOGGER.info(
                    "Epoch %d augmentation scale: %.3f%s",
                    epoch + 1,
                    aug_scale,
                    f" (dataset-side augmentation={train_dataset.training_augmentation})"
                    if in_clean_finetune_window
                    else "",
                )
            if hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)
            batch_sampler.set_start_batch(start_batch if epoch == start_epoch else 0)
            model.train()
            running_loss = 0.0
            running_count = 0
            running_correct = 0
            running_examples = 0
            running_contrastive = 0.0
            running_gate_calibration = 0.0
            running_subject = 0.0
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_examples = 0
            epoch_contrastive = 0.0
            epoch_gate_calibration = 0.0
            epoch_subject = 0.0
            progress = tqdm(
                epoch_train_loader,
                total=steps_per_epoch,
                initial=batch_sampler.start_batch,
                desc=f"sensor fusion {epoch + 1}/{args.epochs}",
                unit="batch",
                disable=not context.is_main,
                dynamic_ncols=True,
            )
            log_started = time.monotonic()
            for batch_index, batch in enumerate(progress, start=batch_sampler.start_batch):
                inputs = _move(batch["inputs"], device)
                masks = _move(batch["modality_mask"], device)
                masks = apply_modality_dropout(masks, args.modality_dropout * aug_scale)
                labels = batch["labels"].to(device, non_blocking=True)
                sample_weights = batch["loss_weights"].to(device, non_blocking=True)
                inputs = augment_sensor_inputs(
                    inputs,
                    visual_jitter=args.visual_jitter * aug_scale,
                    sensor_noise=args.sensor_noise * aug_scale,
                )
                inputs, masks, mixed_labels, mixup_weight, mixup_permutation = mixup_sensor_batch(
                    inputs,
                    masks,
                    labels,
                    args.mixup_alpha * aug_scale,
                )
                optimizer.zero_grad(set_to_none=True)
                contrastive = labels.new_zeros((), dtype=torch.float32)
                gate_calibration = labels.new_zeros((), dtype=torch.float32)
                subject_loss = labels.new_zeros((), dtype=torch.float32)
                subject_weight = 0.0
                if args.subject_adversarial_weight > 0.0:
                    warmup_steps = args.subject_adversarial_warmup_epochs * steps_per_epoch
                    subject_progress = min(step / max(warmup_steps, 1), 1.0)
                    subject_weight = args.subject_adversarial_weight * subject_progress
                with _autocast(device, args.precision):
                    output = model(
                        inputs,
                        modality_mask=masks,
                        subject_adversarial_scale=1.0,
                    )
                    primary_loss = weighted_cross_entropy(
                        output["logits"],
                        labels,
                        sample_weights,
                        class_weights=class_weights,
                        label_smoothing=args.label_smoothing,
                    )
                    mixed_loss = weighted_cross_entropy(
                        output["logits"],
                        mixed_labels,
                        sample_weights[mixup_permutation],
                        class_weights=class_weights,
                        label_smoothing=args.label_smoothing,
                    )
                    loss = mixup_weight * primary_loss + (1.0 - mixup_weight) * mixed_loss
                    if args.auxiliary_loss_weight > 0.0:
                        if "modality_logits" not in output:
                            raise ValueError("--auxiliary-loss-weight requires architecture v4")
                        auxiliary_primary = auxiliary_modality_loss(
                            output["modality_logits"],
                            masks,
                            labels,
                            sample_weights,
                            class_weights=class_weights,
                            label_smoothing=args.label_smoothing,
                        )
                        auxiliary_mixed = auxiliary_modality_loss(
                            output["modality_logits"],
                            masks,
                            mixed_labels,
                            sample_weights[mixup_permutation],
                            class_weights=class_weights,
                            label_smoothing=args.label_smoothing,
                        )
                        auxiliary = (
                            mixup_weight * auxiliary_primary
                            + (1.0 - mixup_weight) * auxiliary_mixed
                        )
                        loss = loss + args.auxiliary_loss_weight * auxiliary
                    if args.gate_calibration_loss_weight > 0.0:
                        if "modality_weights" not in output or "modality_logits" not in output:
                            raise ValueError(
                                "--gate-calibration-loss-weight requires architecture v4"
                            )
                        gate_primary = modality_gate_calibration_loss(
                            output["modality_logits"],
                            output["modality_weights"],
                            masks,
                            labels,
                            sample_weights,
                            quality_temperature=args.gate_quality_temperature,
                            target_smoothing=args.gate_target_smoothing,
                            label_smoothing=args.label_smoothing,
                        )
                        gate_mixed = modality_gate_calibration_loss(
                            output["modality_logits"],
                            output["modality_weights"],
                            masks,
                            mixed_labels,
                            sample_weights[mixup_permutation],
                            quality_temperature=args.gate_quality_temperature,
                            target_smoothing=args.gate_target_smoothing,
                            label_smoothing=args.label_smoothing,
                        )
                        gate_calibration = (
                            mixup_weight * gate_primary + (1.0 - mixup_weight) * gate_mixed
                        )
                        loss = loss + args.gate_calibration_loss_weight * gate_calibration
                    if args.contrastive_loss_weight > 0.0:
                        contrastive = supervised_contrastive_loss(
                            output["pooled_features"],
                            labels,
                            temperature=args.contrastive_temperature,
                            context=context,
                        )
                        loss = loss + args.contrastive_loss_weight * contrastive
                    if args.subject_adversarial_weight > 0.0:
                        if "subject_logits" not in output:
                            raise ValueError(
                                "--subject-adversarial-weight requires architecture v8"
                            )
                        subject_labels = torch.tensor(
                            [
                                user_to_subject_index[int(item["user_id"])]
                                for item in batch["metadata"]
                            ],
                            dtype=torch.long,
                            device=device,
                        )
                        subject_loss = F.cross_entropy(
                            output["subject_logits"],
                            subject_labels,
                        )
                        loss = loss + subject_weight * subject_loss
                if not backend_logged:
                    selected_backends = {
                        block.mlp.last_backend.selected.value
                        for block in raw_model.tr_hash_blocks
                        if block.mlp.last_backend is not None
                    }
                    if args.require_fused_cuda and selected_backends != {"fused_cuda"}:
                        raise RuntimeError(
                            "--require-fused-cuda expected every TR-HASH block to use "
                            f"fused_cuda, got {sorted(selected_backends)}"
                        )
                    if context.is_main:
                        LOGGER.info("TR-HASH backends: %s", ", ".join(sorted(selected_backends)))
                    backend_logged = True
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                step += 1
                running_loss += float(loss.detach())
                running_count += 1
                predictions = output["logits"].argmax(dim=-1)
                correct = mixup_weight * float(predictions.eq(labels).sum())
                correct += (1.0 - mixup_weight) * float(predictions.eq(mixed_labels).sum())
                examples = int(labels.numel())
                running_correct += correct
                running_examples += examples
                running_contrastive += float(contrastive.detach())
                running_gate_calibration += float(gate_calibration.detach())
                running_subject += float(subject_loss.detach())
                epoch_loss += float(loss.detach()) * examples
                epoch_correct += correct
                epoch_examples += examples
                epoch_contrastive += float(contrastive.detach()) * examples
                epoch_gate_calibration += float(gate_calibration.detach()) * examples
                epoch_subject += float(subject_loss.detach()) * examples
                next_epoch = epoch
                next_batch = batch_index + 1
                if next_batch == steps_per_epoch:
                    next_epoch, next_batch = epoch + 1, 0
                cursor_epoch, cursor_batch = next_epoch, next_batch

                if step % args.log_steps == 0:
                    log_finished = time.monotonic()
                    scalars = context.mean_scalars(
                        {
                            "loss": running_loss / max(running_count, 1),
                            "top1": running_correct / max(running_examples, 1),
                            "contrastive": running_contrastive / max(running_count, 1),
                            "gate_calibration": running_gate_calibration / max(running_count, 1),
                            "subject": running_subject / max(running_count, 1),
                            "subject_weight": subject_weight,
                            "local_samples_per_second": running_examples
                            / max(log_finished - log_started, 1e-9),
                        }
                    )
                    if context.is_main:
                        lrs = named_learning_rates(optimizer)
                        progress.set_postfix(
                            loss=f"{scalars['loss']:.4f}",
                            top1=f"{scalars['top1']:.3f}",
                            lr=f"{lrs['base']:.2e}",
                            expert_lr=f"{lrs['expert']:.2e}",
                            contra=f"{scalars['contrastive']:.3f}",
                            gate=f"{scalars['gate_calibration']:.3f}",
                            subject=f"{scalars['subject']:.3f}",
                            sample_s=f"{scalars['local_samples_per_second'] * context.world_size:.1f}",
                        )
                    running_loss = 0.0
                    running_count = 0
                    running_correct = 0
                    running_examples = 0
                    running_contrastive = 0.0
                    running_gate_calibration = 0.0
                    running_subject = 0.0
                    log_started = log_finished

                if args.save_steps > 0 and step % args.save_steps == 0:
                    _save(
                        args.output / f"step_{step:07d}",
                        raw_model,
                        optimizer,
                        scheduler,
                        context,
                        preprocessing=preprocessing,
                        epoch=next_epoch,
                        batch_in_epoch=next_batch,
                        step=step,
                        best_accuracy=best_accuracy,
                        total_epochs=args.epochs,
                        steps_per_epoch=steps_per_epoch,
                        training_options=options,
                        metrics=last_metrics,
                    )
                if args.smoke_steps and step >= args.smoke_steps:
                    stop = True
                    break
            progress.close()
            start_batch = 0

            should_evaluate = stop or (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs
            if should_evaluate:
                # Evaluate the synchronized local replica directly. With the
                # non-padding eval sampler some ranks may have zero batches;
                # invoking DDP.forward on only a subset of ranks can deadlock.
                last_metrics = evaluate(
                    raw_model,
                    validation_loader,
                    device,
                    args.precision,
                    context,
                )
                train_metrics = context.mean_scalars(
                    {
                        "train_loss": epoch_loss / max(epoch_examples, 1),
                        "train_top1_accuracy": epoch_correct / max(epoch_examples, 1),
                        "train_contrastive_loss": epoch_contrastive / max(epoch_examples, 1),
                        "train_gate_calibration_loss": epoch_gate_calibration
                        / max(epoch_examples, 1),
                        "train_subject_loss": epoch_subject / max(epoch_examples, 1),
                    }
                )
                last_metrics.update(train_metrics)
                last_metrics.update(
                    {
                        "epoch": epoch + 1,
                        "step": step,
                        "elapsed_seconds": time.monotonic() - started,
                        "validation_users": list(args.validation_users),
                    }
                )
                last_metrics["selection_composite"] = 0.5 * (
                    last_metrics["top1_accuracy"] + last_metrics["macro_accuracy"]
                )
                if context.is_main:
                    LOGGER.info(
                        "validation epoch=%d top1=%.4f top5=%.4f macro=%.4f "
                        "loss=%.4f train_top1=%.4f train_loss=%.4f examples=%d",
                        epoch,
                        last_metrics["top1_accuracy"],
                        last_metrics["top5_accuracy"],
                        last_metrics["macro_accuracy"],
                        last_metrics["validation_loss"],
                        last_metrics["train_top1_accuracy"],
                        last_metrics["train_loss"],
                        last_metrics["validation_examples"],
                    )
                    if "modality_accuracy" in last_metrics:
                        LOGGER.info(
                            "validation modalities epoch=%d %s",
                            epoch,
                            " ".join(
                                f"{name}=acc:{accuracy:.3f}/gate:{gate:.3f}"
                                for name, accuracy, gate in zip(
                                    SENSOR_MODALITIES,
                                    last_metrics["modality_accuracy"],
                                    last_metrics["modality_gate_weights"],
                                )
                            ),
                        )
                    args.output.mkdir(parents=True, exist_ok=True)
                    with (args.output / "metrics.jsonl").open("a") as handle:
                        handle.write(json.dumps(last_metrics) + "\n")
                if last_metrics["top1_accuracy"] > best_accuracy:
                    best_accuracy = last_metrics["top1_accuracy"]
                    _save(
                        args.output / "best",
                        raw_model,
                        optimizer,
                        scheduler,
                        context,
                        preprocessing=preprocessing,
                        epoch=cursor_epoch,
                        batch_in_epoch=cursor_batch,
                        step=step,
                        best_accuracy=best_accuracy,
                        total_epochs=args.epochs,
                        steps_per_epoch=steps_per_epoch,
                        training_options=options,
                        metrics=last_metrics,
                    )
                for checkpoint_name, metric_name in (
                    ("best_macro", "macro_accuracy"),
                    ("best_composite", "selection_composite"),
                ):
                    metrics_path = args.output / checkpoint_name / "metrics.json"
                    previous = -1.0
                    if metrics_path.is_file():
                        previous = float(json.loads(metrics_path.read_text())[metric_name])
                    if last_metrics[metric_name] > previous:
                        _save(
                            args.output / checkpoint_name,
                            raw_model,
                            optimizer,
                            scheduler,
                            context,
                            preprocessing=preprocessing,
                            epoch=cursor_epoch,
                            batch_in_epoch=cursor_batch,
                            step=step,
                            best_accuracy=best_accuracy,
                            total_epochs=args.epochs,
                            steps_per_epoch=steps_per_epoch,
                            training_options=options,
                            metrics=last_metrics,
                        )
            if stop:
                _save(
                    args.output / "smoke_final",
                    raw_model,
                    optimizer,
                    scheduler,
                    context,
                    preprocessing=preprocessing,
                    epoch=cursor_epoch,
                    batch_in_epoch=cursor_batch,
                    step=step,
                    best_accuracy=best_accuracy,
                    total_epochs=args.epochs,
                    steps_per_epoch=steps_per_epoch,
                    training_options=options,
                    metrics=last_metrics,
                )
                break
        if context.is_main:
            # A smoke limit can coincide exactly with the final batch. Treat
            # that as complete when the persisted cursor reached all epochs.
            if not stop or cursor_epoch >= args.epochs:
                completion = {
                    "completed": True,
                    "epochs": args.epochs,
                    "step": step,
                    "best_top1_accuracy": best_accuracy,
                    "validation_users": list(args.validation_users),
                    "best_checkpoint": str(args.output / "best"),
                    "best_macro_checkpoint": str(args.output / "best_macro"),
                    "best_composite_checkpoint": str(args.output / "best_composite"),
                }
                args.output.mkdir(parents=True, exist_ok=True)
                temporary = args.output / ".training_complete.json.tmp"
                temporary.write_text(json.dumps(completion, indent=2) + "\n")
                temporary.replace(args.output / "training_complete.json")
            LOGGER.info("Training finished at step=%d best_top1=%.4f", step, best_accuracy)
    finally:
        context.close()


if __name__ == "__main__":
    main()
