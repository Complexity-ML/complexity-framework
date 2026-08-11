"""Single-device or DDP training loop for ``TRHashObjectDetector``."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .checkpointing import load_training_state, save_training_state
from .config import TRHashDetectorConfig
from .data import (
    CocoDetectionDataset,
    SyntheticShapesDataset,
    YoloDetectionDataset,
    collate_detection,
)
from .distributed import DistributedContext
from .ema import ModelEMA
from .metrics import DetectionMetricsAccumulator
from .model import TRHashObjectDetector

LOGGER = logging.getLogger("tr_hash_detector")
TRITON_BACKENDS = frozenset({"fused_cuda", "cggr"})


class TqdmLoggingHandler(logging.StreamHandler):
    """Write log records without corrupting an active tqdm progress bar."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=self.stream)
        except Exception:
            self.handleError(record)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=None, help="COCO-format JSON")
    parser.add_argument(
        "--images", type=Path, default=None, help="Directory of images for --annotations"
    )
    parser.add_argument("--validation-annotations", type=Path, default=None)
    parser.add_argument("--validation-images", type=Path, default=None)
    parser.add_argument("--yolo-images", type=Path, default=None)
    parser.add_argument("--yolo-labels", type=Path, default=None)
    parser.add_argument("--validation-yolo-images", type=Path, default=None)
    parser.add_argument("--validation-yolo-labels", type=Path, default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--synthetic-samples", type=int, default=512, help="Used when --annotations is omitted"
    )
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--validation-seed", type=int, default=1_000_000)
    parser.add_argument(
        "--fixed-synthetic-data",
        action="store_true",
        help="Reuse identical synthetic samples each epoch instead of resampling",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Exactly resume model, optimizer, scheduler and data cursor from a checkpoint",
    )
    parser.add_argument(
        "--backbone-checkpoint",
        type=Path,
        default=None,
        help="Directory containing a pretrained tower.safetensors",
    )
    parser.add_argument(
        "--detector-checkpoint",
        type=Path,
        default=None,
        help="Transfer a v6 detector checkpoint, optionally onto new classes",
    )
    parser.add_argument(
        "--class-map",
        type=Path,
        default=None,
        help="JSON object mapping target class IDs to source class IDs",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--architecture-version",
        type=int,
        choices=(6,),
        default=6,
        help="detector architecture version (v6 only)",
    )
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--vision-hidden-size", type=int, default=192)
    parser.add_argument("--vision-layers", type=int, default=4)
    parser.add_argument("--vision-heads", type=int, default=6)
    parser.add_argument("--vision-num-experts", type=int, default=4)
    parser.add_argument("--vision-top-k", type=int, default=2)
    parser.add_argument("--vision-expert-width", type=int, default=48)
    parser.add_argument(
        "--vision-stage-depths",
        type=int,
        nargs="+",
        default=(1, 1, 2),
        help="v6 TR-Hash block counts for the hierarchical stages",
    )
    parser.add_argument("--vision-window-size", type=int, default=8)
    parser.add_argument("--single-scale", action="store_true")
    parser.add_argument(
        "--neck-mode",
        choices=("baseline", "fpn", "pan"),
        default="pan",
        help="cross-scale feature fusion used before the prediction heads",
    )
    parser.add_argument(
        "--p2-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable the stride-8-equivalent fine prediction level",
    )
    parser.add_argument("--static-assignment", action="store_true")
    parser.add_argument("--assignment-top-k", type=int, default=5)
    parser.add_argument(
        "--vision-precision",
        choices=("auto", "fp32", "bf16", "fp16"),
        default="auto",
        help="auto selects fp32 on MPS and bf16 elsewhere",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="training batch size per device",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=0,
        help="validation batch size per device; 0 reuses --batch-size",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--expert-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="EMA decay used for validation/export; 0 disables EMA",
    )
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--eval-confidence", type=float, default=0.20)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="validate every N epochs and always after the final epoch",
    )
    parser.add_argument(
        "--eval-max-detections",
        type=int,
        default=100,
        help="maximum detections retained per validation image",
    )
    parser.add_argument("--box-loss-weight", type=float, default=5.0)
    parser.add_argument("--quality-loss-weight", type=float, default=1.0)
    parser.add_argument("--box-l1-weight", type=float, default=0.25)
    parser.add_argument("--box-iou-weight", type=float, default=1.0)
    parser.add_argument("--reg-max", type=int, default=16)
    parser.add_argument(
        "--head-hidden-size",
        type=int,
        default=0,
        help="0 uses half the backbone width, with a minimum of 32",
    )
    parser.add_argument("--dfl-loss-weight", type=float, default=0.5)
    parser.add_argument("--quality-focal-beta", type=float, default=2.0)
    parser.add_argument(
        "--end-to-end",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="train a one-to-one branch for NMS-free inference",
    )
    parser.add_argument("--one-to-one-loss-weight", type=float, default=0.5)
    parser.add_argument(
        "--augmentation",
        choices=("light", "strong"),
        default="strong",
    )
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--copy-paste", type=float, default=0.0)
    parser.add_argument("--random-erasing", type=float, default=0.0)
    parser.add_argument(
        "--close-mosaic-epochs",
        type=int,
        default=0,
        help="disable Mosaic for the final N epochs",
    )
    parser.add_argument(
        "--multi-scale-min",
        type=int,
        default=0,
        help="minimum square training resolution; 0 disables input resizing",
    )
    parser.add_argument(
        "--multi-scale-max",
        type=int,
        default=0,
        help="maximum square training resolution; 0 uses --image-size",
    )
    parser.add_argument(
        "--multi-scale-step",
        type=int,
        default=0,
        help="resolution interval; 0 uses --patch-size",
    )
    parser.add_argument("--no-stal", action="store_true")
    parser.add_argument("--no-progressive-loss", action="store_true")
    parser.add_argument("--log-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=0, help="0 disables periodic checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Override auto-detected device")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA BF16 autocast")
    parser.add_argument(
        "--require-triton",
        action="store_true",
        help="Fail at startup instead of silently using the PyTorch TR-Hash fallback",
    )
    return parser.parse_args()


def vision_backend_summary(
    model: TRHashObjectDetector,
    device_type: str,
    *,
    require_triton: bool = False,
) -> dict:
    """Resolve and optionally enforce the execution backend for every vision block."""

    summaries = [
        block.mlp.capability_summary(device_type)
        for block in model.tower.blocks
    ]
    selected = {summary["selected_backend"] for summary in summaries}
    if len(selected) != 1:
        raise RuntimeError(
            "vision blocks selected inconsistent TR-Hash backends: "
            + ", ".join(sorted(selected))
        )
    summary = summaries[0]
    if require_triton and summary["selected_backend"] not in TRITON_BACKENDS:
        reasons = "; ".join(summary["backend_reasons"]) or "no backend reason reported"
        raise RuntimeError(
            "Triton is required for this run, but the vision tower selected "
            f"{summary['selected_backend']}: {reasons}"
        )
    return summary


def resolve_device(override: str | None) -> torch.device:
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(
    output: Path,
    model: TRHashObjectDetector,
    config: TRHashDetectorConfig,
    step: int,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    batch_in_epoch: int,
    best_map50: float,
    running_losses: Dict[str, float],
    running_loss_steps: int,
    total_epochs: int,
    steps_per_epoch: int,
    training_options: Dict[str, object],
    name: str | None = None,
    validation_metrics: Dict[str, float] | None = None,
    distributed_rng_states: Sequence[Mapping[str, torch.Tensor]] | None = None,
    ema_model: TRHashObjectDetector | None = None,
) -> None:
    target = output / (name or f"step_{step:06d}")
    target.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    save_file(state, str(target / "model.safetensors"))
    tower_state = {
        name: value.detach().cpu().contiguous() for name, value in model.tower.state_dict().items()
    }
    save_file(tower_state, str(target / "tower.safetensors"))
    if ema_model is not None:
        ema_state = {
            name: value.detach().cpu().contiguous()
            for name, value in ema_model.state_dict().items()
        }
        save_file(ema_state, str(target / "ema.safetensors"))
        ema_tower_state = {
            name: value.detach().cpu().contiguous()
            for name, value in ema_model.tower.state_dict().items()
        }
        save_file(ema_tower_state, str(target / "ema_tower.safetensors"))
    (target / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    if validation_metrics is not None:
        (target / "validation.json").write_text(json.dumps(validation_metrics, indent=2) + "\n")
    save_training_state(
        target,
        optimizer,
        scheduler,
        epoch=epoch,
        batch_in_epoch=batch_in_epoch,
        step=step,
        best_map50=best_map50,
        running_losses=running_losses,
        running_loss_steps=running_loss_steps,
        total_epochs=total_epochs,
        steps_per_epoch=steps_per_epoch,
        training_options=training_options,
        distributed_rng_states=distributed_rng_states,
    )
    LOGGER.info("Checkpoint saved: %s", target)


def load_pretrained_tower(model: TRHashObjectDetector, checkpoint: Path) -> None:
    """Load compatible tower parameters and resize learned patch positions.

    Routing tables are deterministic buffers derived from the target image
    grid, so they are deliberately regenerated instead of copied.
    """

    state = load_file(str(checkpoint / "tower.safetensors"))
    parameters = dict(model.tower.named_parameters())
    position_name = "position_embedding"
    if position_name in state and state[position_name].shape != parameters[position_name].shape:
        old_positions = state[position_name]
        old_grid = math.isqrt(old_positions.shape[1])
        new_grid = math.isqrt(parameters[position_name].shape[1])
        if (
            old_grid**2 != old_positions.shape[1]
            or new_grid**2 != parameters[position_name].shape[1]
        ):
            raise ValueError("vision position embeddings must form square patch grids")
        state[position_name] = (
            F.interpolate(
                old_positions.reshape(1, old_grid, old_grid, old_positions.shape[-1])
                .permute(0, 3, 1, 2)
                .float(),
                size=(new_grid, new_grid),
                mode="bicubic",
                align_corners=False,
            )
            .permute(0, 2, 3, 1)
            .reshape(1, new_grid**2, old_positions.shape[-1])
            .to(old_positions.dtype)
        )
    for name, parameter in parameters.items():
        source = state.get(name)
        if (
            source is not None
            and name.startswith(("position_rows.", "position_cols."))
            and source.shape != parameter.shape
        ):
            if source.ndim != 3 or source.shape[:2] != parameter.shape[:2]:
                continue
            state[name] = F.interpolate(
                source.float(),
                size=parameter.shape[-1],
                mode="linear",
                align_corners=False,
            ).to(source.dtype)

    compatible = {
        name: state[name]
        for name, parameter in parameters.items()
        if name in state and state[name].shape == parameter.shape
    }
    skipped = sorted(set(parameters) - set(compatible))
    model.tower.load_state_dict(compatible, strict=False)
    LOGGER.info(
        "Loaded %d pretrained tower parameters from %s%s",
        len(compatible),
        checkpoint,
        f" (skipped incompatible: {', '.join(skipped)})" if skipped else "",
    )


def load_class_mapping(path: Path | None) -> Dict[int, int] | None:
    if path is None:
        return None
    values = json.loads(path.read_text())
    if not isinstance(values, dict):
        raise ValueError("class map must be a JSON object of target_id: source_id")
    try:
        return {int(target): int(source) for target, source in values.items()}
    except (TypeError, ValueError) as error:
        raise ValueError("class map IDs must be integers") from error


def load_pretrained_detector(
    model: TRHashObjectDetector,
    checkpoint: Path,
    *,
    class_mapping: Dict[int, int] | None = None,
) -> None:
    """Transfer a v6 detector and optionally remap classification rows."""

    source_config = TRHashDetectorConfig.from_dict(
        json.loads((checkpoint / "config.json").read_text())
    )
    source_weights = (
        checkpoint / "ema.safetensors"
        if (checkpoint / "ema.safetensors").is_file()
        else checkpoint / "model.safetensors"
    )
    source_state = load_file(str(source_weights))
    parameters = dict(model.named_parameters())

    position_name = "tower.position_embedding"
    if (
        position_name in source_state
        and position_name in parameters
        and source_state[position_name].shape != parameters[position_name].shape
    ):
        old_positions = source_state[position_name]
        old_grid = math.isqrt(old_positions.shape[1])
        new_grid = math.isqrt(parameters[position_name].shape[1])
        if (
            old_grid**2 != old_positions.shape[1]
            or new_grid**2 != parameters[position_name].shape[1]
        ):
            raise ValueError("vision position embeddings must form square patch grids")
        source_state[position_name] = (
            F.interpolate(
                old_positions.reshape(1, old_grid, old_grid, old_positions.shape[-1])
                .permute(0, 3, 1, 2)
                .float(),
                size=(new_grid, new_grid),
                mode="bicubic",
                align_corners=False,
            )
            .permute(0, 2, 3, 1)
            .reshape(1, new_grid**2, old_positions.shape[-1])
            .to(old_positions.dtype)
        )
    for name, parameter in parameters.items():
        source = source_state.get(name)
        if (
            source is not None
            and name.startswith(("tower.position_rows.", "tower.position_cols."))
            and source.shape != parameter.shape
        ):
            if source.ndim != 3 or source.shape[:2] != parameter.shape[:2]:
                continue
            source_state[name] = F.interpolate(
                source.float(),
                size=parameter.shape[-1],
                mode="linear",
                align_corners=False,
            ).to(source.dtype)

    output_parameter_names = {
        f"head.classification_heads.{level}.3.{suffix}"
        for level in range(len(model.config.grid_sizes))
        for suffix in ("weight", "bias")
    }
    pyramid_compatible = source_config.grid_sizes == model.config.grid_sizes
    compatible = {}
    adapted_outputs = []
    for name, parameter in parameters.items():
        source = source_state.get(name)
        if source is None:
            continue
        if not pyramid_compatible and name.startswith("head."):
            continue
        is_output = name in output_parameter_names
        if not is_output and source.shape == parameter.shape:
            compatible[name] = source
            continue
        if (
            not is_output
            or source.ndim != parameter.ndim
            or source.shape[1:] != parameter.shape[1:]
        ):
            continue
        adapted = parameter.detach().cpu().clone()
        if class_mapping is None and source_config.num_classes == model.config.num_classes:
            adapted.copy_(source)
        elif class_mapping is not None:
            for target_class, source_class in class_mapping.items():
                if not 0 <= target_class < model.config.num_classes:
                    raise ValueError(f"target class ID out of range: {target_class}")
                if not 0 <= source_class < source_config.num_classes:
                    raise ValueError(f"source class ID out of range: {source_class}")
                adapted[target_class].copy_(source[source_class])
        compatible[name] = adapted
        adapted_outputs.append(name)

    model.load_state_dict(compatible, strict=False)
    skipped = sorted(set(parameters) - set(compatible))
    LOGGER.info(
        "Transferred detector from %s: %d parameter tensors, %d adapted output tensors%s",
        checkpoint,
        len(compatible),
        len(adapted_outputs),
        f" (left initialized: {', '.join(skipped)})" if skipped else "",
    )


def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    centers, sizes = boxes[:, :2], boxes[:, 2:]
    return torch.cat((centers - sizes / 2, centers + sizes / 2), dim=-1)


def _match_image_detections(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    target_boxes: torch.Tensor,
    target_labels: torch.Tensor,
    num_classes: int,
) -> Dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Greedily match one image's detections after computing IoUs once.

    Only predictions that overlap any target by at least 0.5 enter the small
    greedy loop. All other predictions are known false positives immediately.
    """

    matches_by_class = {}
    for class_id in range(num_classes):
        prediction_mask = labels == class_id
        class_scores = scores[prediction_mask]
        class_boxes = boxes[prediction_mask]
        order = torch.argsort(class_scores, descending=True)
        class_scores = class_scores[order]
        class_boxes = class_boxes[order]
        matches = torch.zeros(len(class_scores), dtype=torch.float32)
        class_targets = target_boxes[target_labels == class_id]

        if len(class_boxes) and len(class_targets):
            top_left = torch.maximum(class_boxes[:, None, :2], class_targets[None, :, :2])
            bottom_right = torch.minimum(class_boxes[:, None, 2:], class_targets[None, :, 2:])
            intersections = (bottom_right - top_left).clamp_min(0).prod(-1)
            prediction_areas = (class_boxes[:, 2:] - class_boxes[:, :2]).clamp_min(0).prod(-1)
            target_areas = (class_targets[:, 2:] - class_targets[:, :2]).clamp_min(0).prod(-1)
            ious = intersections / (
                prediction_areas[:, None] + target_areas[None, :] - intersections
            ).clamp_min(1e-9)
            candidate_predictions = torch.nonzero(
                (ious >= 0.5).any(dim=1), as_tuple=False
            ).flatten()
            used_targets = torch.zeros(len(class_targets), dtype=torch.bool)
            for prediction_index in candidate_predictions.tolist():
                available_ious = ious[prediction_index].masked_fill(used_targets, -1.0)
                best_iou, target_index = available_ious.max(dim=0)
                if float(best_iou) >= 0.5:
                    matches[prediction_index] = 1.0
                    used_targets[int(target_index)] = True

        matches_by_class[class_id] = (class_scores, matches)
    return matches_by_class


def _average_precision_from_matches(
    scores: torch.Tensor,
    true_positives: torch.Tensor,
    total_ground_truth: int,
) -> float:
    if total_ground_truth == 0:
        return float("nan")
    if not len(scores):
        return 0.0
    order = torch.argsort(scores, descending=True)
    true_positive_cumulative = true_positives[order].cumsum(0)
    false_positive_cumulative = (1.0 - true_positives[order]).cumsum(0)
    recall = true_positive_cumulative / total_ground_truth
    precision = true_positive_cumulative / (
        true_positive_cumulative + false_positive_cumulative
    ).clamp_min(1e-9)
    recall = torch.cat((torch.tensor([0.0]), recall, torch.tensor([1.0])))
    precision = torch.cat((torch.tensor([1.0]), precision, torch.tensor([0.0])))
    precision = torch.cummax(precision.flip(0), dim=0).values.flip(0)
    changing = torch.nonzero(recall[1:] != recall[:-1], as_tuple=False).flatten()
    return float(((recall[changing + 1] - recall[changing]) * precision[changing + 1]).sum())


@torch.inference_mode()
def evaluate_detector(
    model: TRHashObjectDetector,
    loader: DataLoader,
    device: torch.device,
    *,
    confidence_threshold: float,
    use_amp: bool = False,
    show_progress: bool = False,
    max_detections: int = 100,
    distributed: DistributedContext | None = None,
) -> Dict[str, float]:
    model.eval()
    metrics = DetectionMetricsAccumulator(
        model.config.num_classes,
        model.config.image_size,
    )

    progress = tqdm(
        loader,
        desc="detector validation",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
        disable=False if show_progress else True,
    )
    for pixel_values, targets in progress:
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
        with autocast:
            detections = model.predict(
                pixel_values.to(device, non_blocking=device.type == "cuda"),
                confidence_threshold=0.001,
                iou_threshold=0.5,
                postprocess_on_cpu=device.type == "mps",
                max_detections=max_detections,
            )
        for detection, image_targets in zip(detections, targets):
            metrics.update(
                detection["boxes"],
                detection["scores"],
                detection["labels"],
                image_targets,
            )
    if distributed is not None and distributed.enabled:
        states = distributed.all_gather_objects(metrics.state_dict())
        metrics = DetectionMetricsAccumulator(
            model.config.num_classes,
            model.config.image_size,
        )
        for state in states:
            metrics.merge_state_dict(state)
    model.train()
    return metrics.compute(confidence_threshold)


def cosine_schedule(step: int, *, warmup_steps: int, total_steps: int, min_ratio: float) -> float:
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def should_validate_epoch(epoch: int, total_epochs: int, eval_every: int) -> bool:
    """Return true on the configured cadence and unconditionally at the end."""

    completed_epochs = epoch + 1
    return completed_epochs % eval_every == 0 or completed_epochs == total_epochs


def main() -> None:
    args = parse_args()
    distributed = DistributedContext.initialize(resolve_device(args.device))
    logging.basicConfig(
        level=logging.INFO if distributed.is_main else logging.ERROR,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[TqdmLoggingHandler()],
        force=True,
    )
    torch.manual_seed(args.seed + distributed.rank)
    device = distributed.device
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    validation_dataset = None
    epoch_dataset = None
    if args.yolo_images is not None:
        if args.yolo_labels is None:
            raise ValueError("--yolo-labels is required alongside --yolo-images")
        train_full = YoloDetectionDataset(
            args.yolo_images,
            args.yolo_labels,
            image_size=args.image_size,
            augmentation=args.augmentation,
            seed=args.seed,
            mosaic_probability=args.mosaic,
            mixup_probability=args.mixup,
            copy_paste_probability=args.copy_paste,
            random_erasing_probability=args.random_erasing,
            total_epochs=args.epochs,
            close_mosaic_epochs=args.close_mosaic_epochs,
        )
        validation_paths = (
            args.validation_yolo_images,
            args.validation_yolo_labels,
        )
        if all(value is not None for value in validation_paths):
            dataset = train_full
            validation_dataset = YoloDetectionDataset(
                args.validation_yolo_images,
                args.validation_yolo_labels,
                image_size=args.image_size,
            )
        elif any(value is not None for value in validation_paths):
            raise ValueError("both validation YOLO image and label dirs are required")
        else:
            if not 0.0 < args.validation_fraction < 1.0:
                raise ValueError("--validation-fraction must be between 0 and 1")
            if len(train_full) < 2:
                raise ValueError("at least two YOLO images are required for an automatic split")
            validation_full = YoloDetectionDataset(
                args.yolo_images,
                args.yolo_labels,
                image_size=args.image_size,
            )
            generator = torch.Generator().manual_seed(args.validation_seed)
            indices = torch.randperm(len(train_full), generator=generator).tolist()
            validation_count = min(
                max(round(len(indices) * args.validation_fraction), 1),
                len(indices) - 1,
            )
            dataset = Subset(train_full, indices[validation_count:])
            validation_dataset = Subset(validation_full, indices[:validation_count])
        epoch_dataset = train_full
        num_classes = train_full.num_classes
        LOGGER.info(
            "YOLO dataset: %d train, %d validation, %d classes",
            len(dataset),
            len(validation_dataset),
            num_classes,
        )
    elif args.annotations is not None:
        if args.images is None:
            raise ValueError("--images is required alongside --annotations")
        dataset = CocoDetectionDataset(
            args.annotations,
            args.images,
            image_size=args.image_size,
            augmentation=args.augmentation,
            seed=args.seed,
            mosaic_probability=args.mosaic,
            mixup_probability=args.mixup,
            copy_paste_probability=args.copy_paste,
            random_erasing_probability=args.random_erasing,
            total_epochs=args.epochs,
            close_mosaic_epochs=args.close_mosaic_epochs,
        )
        epoch_dataset = dataset
        num_classes = dataset.num_classes
        validation_coco_paths = (args.validation_annotations, args.validation_images)
        if all(path is not None for path in validation_coco_paths):
            validation_dataset = CocoDetectionDataset(
                args.validation_annotations,
                args.validation_images,
                image_size=args.image_size,
            )
            if validation_dataset.num_classes != num_classes:
                raise ValueError("COCO train/validation class counts differ")
        elif any(path is not None for path in validation_coco_paths):
            raise ValueError(
                "both validation COCO annotations and image dirs are required"
            )
        LOGGER.info("COCO dataset: %d images, %d classes", len(dataset), num_classes)
    else:
        dataset = SyntheticShapesDataset(
            args.synthetic_samples,
            image_size=args.image_size,
            seed=args.seed,
            resample_each_epoch=not args.fixed_synthetic_data,
        )
        validation_dataset = SyntheticShapesDataset(
            args.validation_samples,
            image_size=args.image_size,
            seed=args.validation_seed,
        )
        num_classes = args.num_classes
        epoch_dataset = dataset
        LOGGER.info("Synthetic dataset: %d images, %d classes", len(dataset), num_classes)

    if args.eval_every <= 0:
        raise ValueError("--eval-every must be positive")
    if args.eval_max_detections <= 0:
        raise ValueError("--eval-max-detections must be positive")
    if args.eval_batch_size < 0:
        raise ValueError("--eval-batch-size cannot be negative")
    multi_scale_max = args.multi_scale_max or args.image_size
    multi_scale_step = args.multi_scale_step or args.patch_size
    if args.multi_scale_min:
        if not args.multi_scale_min <= multi_scale_max <= args.image_size:
            raise ValueError("multi-scale resolutions must satisfy min <= max <= image-size")
        if (
            args.multi_scale_min % args.patch_size
            or multi_scale_max % args.patch_size
            or multi_scale_step % args.patch_size
        ):
            raise ValueError("multi-scale resolutions must be divisible by patch-size")
    shuffle_generator = torch.Generator()
    train_sampler = distributed.train_sampler(dataset, args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        generator=shuffle_generator if train_sampler is None else None,
        num_workers=args.workers,
        # Recreate workers each epoch so dataset.set_epoch() reaches worker copies.
        persistent_workers=False,
        pin_memory=device.type == "cuda",
        collate_fn=collate_detection,
    )
    validation_loader = (
        DataLoader(
            validation_dataset,
            batch_size=args.eval_batch_size or args.batch_size,
            sampler=distributed.eval_sampler(validation_dataset),
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
            pin_memory=device.type == "cuda",
            collate_fn=collate_detection,
        )
        if validation_dataset is not None
        else None
    )

    vision_precision = args.vision_precision
    if vision_precision == "auto":
        vision_precision = "fp32" if device.type == "mps" else "bf16"
    config = TRHashDetectorConfig(
        architecture_version=args.architecture_version,
        image_size=args.image_size,
        patch_size=args.patch_size,
        vision_hidden_size=args.vision_hidden_size,
        vision_layers=args.vision_layers,
        vision_heads=args.vision_heads,
        vision_num_experts=args.vision_num_experts,
        vision_top_k=args.vision_top_k,
        vision_expert_width=args.vision_expert_width,
        vision_stage_depths=tuple(args.vision_stage_depths),
        vision_window_size=args.vision_window_size,
        vision_precision=vision_precision,
        num_classes=num_classes,
        multi_scale=not args.single_scale,
        p2_head=args.p2_head,
        neck_mode=args.neck_mode,
        dynamic_assignment=not args.static_assignment,
        assignment_top_k=args.assignment_top_k,
        stal_enabled=not args.no_stal,
        progressive_loss_enabled=not args.no_progressive_loss,
        reg_max=args.reg_max,
        head_hidden_size=args.head_hidden_size,
        dfl_loss_weight=args.dfl_loss_weight,
        quality_focal_beta=args.quality_focal_beta,
        end_to_end=args.end_to_end,
        one_to_one_loss_weight=args.one_to_one_loss_weight,
        box_loss_weight=args.box_loss_weight,
        quality_loss_weight=args.quality_loss_weight,
        box_l1_weight=args.box_l1_weight,
        box_iou_weight=args.box_iou_weight,
    )
    model = TRHashObjectDetector(config).to(device)
    initialization_modes = (
        args.resume,
        args.backbone_checkpoint,
        args.detector_checkpoint,
    )
    if sum(value is not None for value in initialization_modes) > 1:
        raise ValueError(
            "--resume, --backbone-checkpoint, and --detector-checkpoint are mutually exclusive"
        )
    if args.class_map is not None and args.detector_checkpoint is None:
        raise ValueError("--class-map requires --detector-checkpoint")
    if args.resume is not None:
        saved_config = TRHashDetectorConfig.from_dict(
            json.loads((args.resume / "config.json").read_text())
        )
        config_mismatches = sorted(
            key
            for key in set(saved_config.to_dict()) | set(config.to_dict())
            if saved_config.to_dict().get(key) != config.to_dict().get(key)
        )
        if config_mismatches:
            raise ValueError(
                "exact resume requires unchanged model/loss config; mismatched: "
                + ", ".join(config_mismatches)
            )
        model.load_state_dict(load_file(str(args.resume / "model.safetensors")))
    elif args.detector_checkpoint is not None:
        load_pretrained_detector(
            model,
            args.detector_checkpoint,
            class_mapping=load_class_mapping(args.class_map),
        )
    elif args.backbone_checkpoint is not None:
        load_pretrained_tower(model, args.backbone_checkpoint)
    LOGGER.info("Model: %.2fM parameters", model.num_parameters() / 1e6)
    backend_summary = vision_backend_summary(
        model,
        device.type,
        require_triton=args.require_triton,
    )
    LOGGER.info(
        "TR-Hash vision backend: %s (requested=%s experts=%d top_k=%d precision=%s)",
        backend_summary["selected_backend"],
        backend_summary["requested_backend"],
        backend_summary["experts"],
        backend_summary["top_k"],
        backend_summary["precision"],
    )
    if args.expert_lr_multiplier <= 0.0:
        raise ValueError("--expert-lr-multiplier must be positive")
    expert_parameters = []
    base_parameters = []
    for name, parameter in model.named_parameters():
        target = expert_parameters if ".mlp.expert_" in name else base_parameters
        target.append(parameter)
    parameter_groups = (
        {"params": base_parameters, "lr": args.lr, "group_name": "base"},
        {
            "params": expert_parameters,
            "lr": args.lr * args.expert_lr_multiplier,
            "group_name": "experts",
        },
    )
    optimizer = torch.optim.SGD(
        parameter_groups,
        momentum=args.momentum,
        nesterov=True,
        weight_decay=args.weight_decay,
        foreach=False if device.type == "mps" else None,
    )
    LOGGER.info(
        "Optimizer: %s (base_lr=%.2e expert_lr=%.2e)",
        "sgd",
        args.lr,
        args.lr * args.expert_lr_multiplier,
    )
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay else None
    if ema is not None:
        LOGGER.info("EMA enabled: decay=%.6f", args.ema_decay)
    total_steps = len(loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_schedule(
            step,
            warmup_steps=min(args.warmup_steps, total_steps),
            total_steps=total_steps,
            min_ratio=args.min_lr_ratio,
        ),
    )
    training_model = distributed.wrap(model)

    if distributed.is_main:
        args.output.mkdir(parents=True, exist_ok=True)
    distributed.barrier()
    metrics_path = args.output / "metrics.jsonl"

    training_options: Dict[str, object] = {
        "optimizer": "sgd",
        "batch_size": args.batch_size,
        "dataset_size": len(dataset),
        "seed": args.seed,
        "lr": args.lr,
        "expert_lr_multiplier": args.expert_lr_multiplier,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "min_lr_ratio": args.min_lr_ratio,
        "augmentation": args.augmentation,
        "device_type": device.type,
        "use_amp": use_amp,
    }
    training_options.update(
        {
            "ema_decay": args.ema_decay,
            "mosaic": args.mosaic,
            "mixup": args.mixup,
            "copy_paste": args.copy_paste,
            "random_erasing": args.random_erasing,
            "close_mosaic_epochs": args.close_mosaic_epochs,
            "multi_scale_min": args.multi_scale_min,
            "multi_scale_max": multi_scale_max,
            "multi_scale_step": multi_scale_step,
        }
    )
    if distributed.enabled:
        training_options["world_size"] = distributed.world_size
    start_epoch = 0
    start_batch = 0
    step = 0
    best_map50 = -1.0
    running_losses: Dict[str, float] = {}
    running_loss_steps = 0
    if args.resume is not None:
        resume_state = load_training_state(
            args.resume,
            optimizer,
            scheduler,
            total_epochs=args.epochs,
            steps_per_epoch=len(loader),
            training_options=training_options,
            rank=distributed.rank,
            world_size=distributed.world_size,
            device=device,
        )
        start_epoch = int(resume_state["epoch"])
        start_batch = int(resume_state["batch_in_epoch"])
        step = int(resume_state["step"])
        best_map50 = float(resume_state["best_map50"])
        running_losses = {
            str(name): float(value)
            for name, value in resume_state.get("running_losses", {}).items()
        }
        running_loss_steps = int(resume_state.get("running_loss_steps", 0))
        if not 0 <= start_epoch <= args.epochs:
            raise ValueError(f"invalid resumed epoch cursor: {start_epoch}")
        if not 0 <= start_batch <= len(loader):
            raise ValueError(f"invalid resumed batch cursor: {start_batch}")
        if start_epoch == args.epochs and start_batch:
            raise ValueError("a completed run cannot have a non-zero batch cursor")
        LOGGER.info(
            "Resumed exactly from %s: epoch=%d batch=%d step=%d",
            args.resume,
            start_epoch,
            start_batch,
            step,
        )
        if ema is not None:
            ema_path = args.resume / "ema.safetensors"
            if not ema_path.is_file():
                raise ValueError("EMA-enabled resume requires ema.safetensors")
            ema.load_state_dict(load_file(str(ema_path)), updates=step)

    def write_checkpoint(
        *,
        epoch: int,
        batch_in_epoch: int,
        name: str | None = None,
        validation_metrics: Dict[str, float] | None = None,
    ) -> None:
        distributed_rng_states = distributed.gather_rng_states()
        if distributed.is_main:
            save_checkpoint(
                args.output,
                model,
                config,
                step,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                batch_in_epoch=batch_in_epoch,
                best_map50=best_map50,
                running_losses=running_losses,
                running_loss_steps=running_loss_steps,
                total_epochs=args.epochs,
                steps_per_epoch=len(loader),
                training_options=training_options,
                name=name,
                validation_metrics=validation_metrics,
                distributed_rng_states=distributed_rng_states,
                ema_model=ema.module if ema is not None else None,
            )
        distributed.barrier()

    for epoch in range(start_epoch, args.epochs):
        if epoch_dataset is not None and hasattr(epoch_dataset, "set_epoch"):
            epoch_dataset.set_epoch(epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        else:
            shuffle_generator.manual_seed(args.seed + epoch)
        loader_iterator = iter(loader)
        batches_to_skip = start_batch if epoch == start_epoch else 0
        for _ in range(batches_to_skip):
            next(loader_iterator)
        progress = tqdm(
            loader_iterator,
            desc=f"detector train {epoch + 1}/{args.epochs}",
            unit="batch",
            total=len(loader),
            initial=batches_to_skip,
            dynamic_ncols=True,
            leave=False,
            disable=not distributed.is_main,
        )
        for batch_index, (pixel_values, targets) in enumerate(
            progress, start=batches_to_skip
        ):
            pixel_values = pixel_values.to(device, non_blocking=device.type == "cuda")
            if args.multi_scale_min:
                choices = range(
                    args.multi_scale_min,
                    multi_scale_max + 1,
                    multi_scale_step,
                )
                resize_rng = random.Random(
                    args.seed + epoch * len(loader) + batch_index
                )
                runtime_size = resize_rng.choice(tuple(choices))
                if tuple(pixel_values.shape[-2:]) != (runtime_size, runtime_size):
                    pixel_values = F.interpolate(
                        pixel_values,
                        size=(runtime_size, runtime_size),
                        mode="bilinear",
                        align_corners=False,
                    )
            targets = [target.to(device, non_blocking=device.type == "cuda") for target in targets]

            autocast = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
            with autocast:
                raw = training_model(
                    pixel_values,
                    return_branches=config.end_to_end,
                )
                losses = model.compute_loss(
                    raw,
                    targets,
                    training_progress=step / max(total_steps - 1, 1),
                )
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                foreach=False if device.type == "mps" else None,
            )
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(model)

            step += 1
            for name, value in losses.items():
                running_losses[name] = running_losses.get(name, 0.0) + float(value.detach())
            running_loss_steps += 1
            if step % args.log_steps == 0:
                averages = {
                    name: value / running_loss_steps for name, value in running_losses.items()
                }
                averages = distributed.mean_scalars(averages)
                if distributed.is_main:
                    with metrics_path.open("a") as handle:
                        handle.write(
                            json.dumps(
                                {
                                    "step": step,
                                    "epoch": epoch,
                                    "lr": scheduler.get_last_lr()[0],
                                    "expert_lr": scheduler.get_last_lr()[1],
                                    **averages,
                                }
                            )
                            + "\n"
                        )
                    progress.set_postfix(
                        loss=f"{averages['loss']:.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                        expert_lr=f"{scheduler.get_last_lr()[1]:.2e}",
                    )
                running_losses.clear()
                running_loss_steps = 0
            if args.save_steps and step % args.save_steps == 0:
                write_checkpoint(epoch=epoch, batch_in_epoch=batch_index + 1)

        should_validate = validation_loader is not None and should_validate_epoch(
            epoch,
            args.epochs,
            args.eval_every,
        )
        if should_validate:
            validation_metrics = evaluate_detector(
                ema.module if ema is not None else model,
                validation_loader,
                device,
                confidence_threshold=args.eval_confidence,
                use_amp=use_amp,
                show_progress=distributed.is_main,
                max_detections=args.eval_max_detections,
                distributed=distributed,
            )
            LOGGER.info(
                "validation epoch=%d mAP50=%.4f mAP50-95=%.4f "
                "APs=%.4f APm=%.4f APl=%.4f precision=%.4f recall=%.4f "
                "f1=%.4f best_f1=%.4f best_conf=%.3f",
                epoch,
                validation_metrics["map50"],
                validation_metrics["map50_95"],
                validation_metrics["ap_small"],
                validation_metrics["ap_medium"],
                validation_metrics["ap_large"],
                validation_metrics["precision"],
                validation_metrics["recall"],
                validation_metrics["f1"],
                validation_metrics["best_f1"],
                validation_metrics["best_confidence"],
            )
            if distributed.is_main:
                with metrics_path.open("a") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "step": step,
                                "epoch": epoch,
                                "validation": validation_metrics,
                            }
                        )
                        + "\n"
                    )
            if validation_metrics["map50"] > best_map50:
                best_map50 = validation_metrics["map50"]
                write_checkpoint(
                    epoch=epoch + 1,
                    batch_in_epoch=0,
                    name="best",
                    validation_metrics=validation_metrics,
                )
        start_batch = 0

    write_checkpoint(epoch=args.epochs, batch_in_epoch=0)
    LOGGER.info("Training complete: %d steps over %d epochs", step, args.epochs)
    distributed.close()


if __name__ == "__main__":
    main()
