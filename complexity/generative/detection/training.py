"""Training loop for ``TRHashObjectDetector``.

Single-device (CPU/MPS/CUDA), non-distributed -- the detector is small
enough that a training farm is not the point here; this is for proving the
model actually learns, and for fine-tuning on a real COCO-format dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Subset

from .config import TRHashDetectorConfig
from .data import (
    CocoDetectionDataset,
    SyntheticShapesDataset,
    YoloDetectionDataset,
    collate_detection,
)
from .model import TRHashObjectDetector

LOGGER = logging.getLogger("tr_hash_detector")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=None, help="COCO-format JSON")
    parser.add_argument("--images", type=Path, default=None, help="Directory of images for --annotations")
    parser.add_argument("--yolo-images", type=Path, default=None)
    parser.add_argument("--yolo-labels", type=Path, default=None)
    parser.add_argument("--validation-yolo-images", type=Path, default=None)
    parser.add_argument("--validation-yolo-labels", type=Path, default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--synthetic-samples", type=int, default=512, help="Used when --annotations is omitted")
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
        help="Load model.safetensors from a checkpoint directory before training",
    )
    parser.add_argument(
        "--backbone-checkpoint",
        type=Path,
        default=None,
        help="Directory containing a pretrained tower.safetensors",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--vision-hidden-size", type=int, default=192)
    parser.add_argument("--vision-layers", type=int, default=4)
    parser.add_argument("--vision-heads", type=int, default=6)
    parser.add_argument("--vision-num-experts", type=int, default=4)
    parser.add_argument("--vision-top-k", type=int, default=2)
    parser.add_argument("--vision-expert-width", type=int, default=48)
    parser.add_argument("--single-scale", action="store_true")
    parser.add_argument("--static-assignment", action="store_true")
    parser.add_argument("--assignment-top-k", type=int, default=5)
    parser.add_argument(
        "--vision-precision",
        choices=("auto", "fp32", "bf16", "fp16"),
        default="auto",
        help="auto selects fp32 on MPS and bf16 elsewhere",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--expert-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--eval-confidence", type=float, default=0.20)
    parser.add_argument("--box-loss-weight", type=float, default=5.0)
    parser.add_argument("--objectness-loss-weight", type=float, default=1.0)
    parser.add_argument("--class-loss-weight", type=float, default=1.0)
    parser.add_argument("--box-l1-weight", type=float, default=0.25)
    parser.add_argument("--box-iou-weight", type=float, default=1.0)
    parser.add_argument("--focal-alpha", type=float, default=0.5)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--objectness-loss-type",
        choices=("focal", "varifocal"),
        default="varifocal",
    )
    parser.add_argument("--varifocal-alpha", type=float, default=0.75)
    parser.add_argument("--varifocal-gamma", type=float, default=2.0)
    parser.add_argument("--log-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=0, help="0 disables periodic checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Override auto-detected device")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA BF16 autocast")
    return parser.parse_args()


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
    name: str | None = None,
    validation_metrics: Dict[str, float] | None = None,
) -> None:
    target = output / (name or f"step_{step:06d}")
    target.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    save_file(state, str(target / "model.safetensors"))
    tower_state = {
        name: value.detach().cpu().contiguous()
        for name, value in model.tower.state_dict().items()
    }
    save_file(tower_state, str(target / "tower.safetensors"))
    (target / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    if validation_metrics is not None:
        (target / "validation.json").write_text(
            json.dumps(validation_metrics, indent=2) + "\n"
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
        if old_grid**2 != old_positions.shape[1] or new_grid**2 != parameters[position_name].shape[1]:
            raise ValueError("vision position embeddings must form square patch grids")
        state[position_name] = F.interpolate(
            old_positions.reshape(1, old_grid, old_grid, old_positions.shape[-1])
            .permute(0, 3, 1, 2)
            .float(),
            size=(new_grid, new_grid),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).reshape(1, new_grid**2, old_positions.shape[-1]).to(
            old_positions.dtype
        )

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


def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    centers, sizes = boxes[:, :2], boxes[:, 2:]
    return torch.cat((centers - sizes / 2, centers + sizes / 2), dim=-1)


def _box_iou_one_to_many(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(box[:2], boxes[:, :2])
    bottom_right = torch.minimum(box[2:], boxes[:, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(-1)
    box_area = (box[2:] - box[:2]).clamp_min(0).prod()
    boxes_area = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(-1)
    return intersection / (box_area + boxes_area - intersection).clamp_min(1e-9)


def _average_precision(
    predictions: List[tuple[float, int, torch.Tensor]],
    ground_truth: Dict[int, torch.Tensor],
) -> float:
    total_ground_truth = sum(len(boxes) for boxes in ground_truth.values())
    if total_ground_truth == 0:
        return float("nan")

    matched = {image_index: set() for image_index in ground_truth}
    true_positives = []
    false_positives = []
    for _, image_index, predicted_box in sorted(predictions, reverse=True, key=lambda item: item[0]):
        target_boxes = ground_truth.get(image_index)
        if target_boxes is None or not len(target_boxes):
            true_positives.append(0.0)
            false_positives.append(1.0)
            continue
        ious = _box_iou_one_to_many(predicted_box, target_boxes)
        best_iou, best_index = ious.max(dim=0)
        target_index = int(best_index)
        is_match = float(best_iou) >= 0.5 and target_index not in matched[image_index]
        true_positives.append(float(is_match))
        false_positives.append(float(not is_match))
        if is_match:
            matched[image_index].add(target_index)

    if not true_positives:
        return 0.0
    true_positive_cumulative = torch.tensor(true_positives).cumsum(0)
    false_positive_cumulative = torch.tensor(false_positives).cumsum(0)
    recall = true_positive_cumulative / total_ground_truth
    precision = true_positive_cumulative / (
        true_positive_cumulative + false_positive_cumulative
    ).clamp_min(1e-9)
    recall = torch.cat((torch.tensor([0.0]), recall, torch.tensor([1.0])))
    precision = torch.cat((torch.tensor([1.0]), precision, torch.tensor([0.0])))
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = torch.maximum(precision[index], precision[index + 1])
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
) -> Dict[str, float]:
    model.eval()
    predictions_by_class: Dict[int, List[tuple[float, int, torch.Tensor]]] = {
        class_id: [] for class_id in range(model.config.num_classes)
    }
    targets_by_class: Dict[int, Dict[int, torch.Tensor]] = {
        class_id: {} for class_id in range(model.config.num_classes)
    }
    image_index = 0
    scored_matches: List[tuple[float, bool]] = []
    total_targets = 0

    for pixel_values, targets in loader:
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
        )
        with autocast:
            detections = model.predict(
                pixel_values.to(device, non_blocking=device.type == "cuda"),
                objectness_threshold=0.001,
                iou_threshold=0.5,
                postprocess_on_cpu=device.type == "mps",
            )
        for detection, image_targets in zip(detections, targets):
            target_boxes = _xywh_to_xyxy(image_targets[:, :4]).cpu()
            target_labels = image_targets[:, 4].long().cpu()
            total_targets += len(image_targets)
            for class_id in range(model.config.num_classes):
                targets_by_class[class_id][image_index] = target_boxes[
                    target_labels == class_id
                ]

            boxes = detection["boxes"].cpu()
            scores = detection["scores"].cpu()
            labels = detection["labels"].cpu()
            for box, score, label in zip(boxes, scores, labels):
                predictions_by_class[int(label)].append((float(score), image_index, box))

            used_targets = set()
            for prediction_index in torch.argsort(scores, descending=True).tolist():
                is_match = False
                candidates = torch.nonzero(
                    target_labels == labels[prediction_index], as_tuple=False
                ).flatten()
                candidates = [int(index) for index in candidates if int(index) not in used_targets]
                if candidates:
                    candidate_tensor = torch.tensor(candidates, dtype=torch.long)
                    candidate_ious = _box_iou_one_to_many(
                        boxes[prediction_index], target_boxes[candidate_tensor]
                    )
                    best_iou, relative_index = candidate_ious.max(dim=0)
                    if float(best_iou) >= 0.5:
                        used_targets.add(candidates[int(relative_index)])
                        is_match = True
                scored_matches.append((float(scores[prediction_index]), is_match))
            image_index += 1

    average_precisions = [
        _average_precision(predictions_by_class[class_id], targets_by_class[class_id])
        for class_id in range(model.config.num_classes)
    ]
    valid_average_precisions = [value for value in average_precisions if not math.isnan(value)]
    scored_matches.sort(reverse=True, key=lambda item: item[0])
    scores_tensor = torch.tensor([item[0] for item in scored_matches])
    matches_tensor = torch.tensor(
        [float(item[1]) for item in scored_matches], dtype=torch.float32
    )
    fixed_mask = scores_tensor >= confidence_threshold
    fixed_true_positives = float(matches_tensor[fixed_mask].sum())
    fixed_predictions = int(fixed_mask.sum())
    precision = fixed_true_positives / max(fixed_predictions, 1)
    recall = fixed_true_positives / max(total_targets, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    if len(scored_matches):
        cumulative_true_positives = matches_tensor.cumsum(0)
        prediction_counts = torch.arange(1, len(scored_matches) + 1)
        f1_curve = 2.0 * cumulative_true_positives / (
            prediction_counts + total_targets
        ).clamp_min(1)
        best_f1, best_index = f1_curve.max(dim=0)
        best_confidence = float(scores_tensor[int(best_index)])
    else:
        best_f1 = torch.tensor(0.0)
        best_confidence = 1.0
    model.train()
    return {
        "map50": sum(valid_average_precisions) / max(len(valid_average_precisions), 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "best_f1": float(best_f1),
        "best_confidence": best_confidence,
    }


def cosine_schedule(step: int, *, warmup_steps: int, total_steps: int, min_ratio: float) -> float:
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
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
            augment=True,
            seed=args.seed,
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
        dataset = CocoDetectionDataset(args.annotations, args.images, image_size=args.image_size)
        num_classes = dataset.num_classes
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

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_detection,
    )
    validation_loader = (
        DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
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
        image_size=args.image_size,
        patch_size=args.patch_size,
        vision_hidden_size=args.vision_hidden_size,
        vision_layers=args.vision_layers,
        vision_heads=args.vision_heads,
        vision_num_experts=args.vision_num_experts,
        vision_top_k=args.vision_top_k,
        vision_expert_width=args.vision_expert_width,
        vision_precision=vision_precision,
        num_classes=num_classes,
        multi_scale=not args.single_scale,
        dynamic_assignment=not args.static_assignment,
        assignment_top_k=args.assignment_top_k,
        box_loss_weight=args.box_loss_weight,
        objectness_loss_weight=args.objectness_loss_weight,
        class_loss_weight=args.class_loss_weight,
        box_l1_weight=args.box_l1_weight,
        box_iou_weight=args.box_iou_weight,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        objectness_loss_type=args.objectness_loss_type,
        varifocal_alpha=args.varifocal_alpha,
        varifocal_gamma=args.varifocal_gamma,
    )
    model = TRHashObjectDetector(config).to(device)
    if args.resume is not None and args.backbone_checkpoint is not None:
        raise ValueError("--resume and --backbone-checkpoint are mutually exclusive")
    if args.resume is not None:
        model.load_state_dict(load_file(str(args.resume / "model.safetensors")))
        LOGGER.info("Resumed from: %s", args.resume)
    elif args.backbone_checkpoint is not None:
        load_pretrained_tower(model, args.backbone_checkpoint)
    LOGGER.info("Model: %.2fM parameters", model.num_parameters() / 1e6)
    if args.expert_lr_multiplier <= 0.0:
        raise ValueError("--expert-lr-multiplier must be positive")
    expert_parameters = []
    base_parameters = []
    for name, parameter in model.named_parameters():
        target = expert_parameters if ".mlp.expert_" in name else base_parameters
        target.append(parameter)
    optimizer_options = {
        "weight_decay": args.weight_decay,
        "foreach": False if device.type == "mps" else None,
    }
    if device.type == "cuda":
        optimizer_options["fused"] = True
    optimizer = torch.optim.AdamW(
        (
            {"params": base_parameters, "lr": args.lr, "group_name": "base"},
            {
                "params": expert_parameters,
                "lr": args.lr * args.expert_lr_multiplier,
                "group_name": "experts",
            },
        ),
        **optimizer_options,
    )
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

    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.jsonl"

    step = 0
    running_losses: Dict[str, float] = {}
    started = time.monotonic()
    best_map50 = -1.0
    for epoch in range(args.epochs):
        if epoch_dataset is not None and hasattr(epoch_dataset, "set_epoch"):
            epoch_dataset.set_epoch(epoch)
        for pixel_values, targets in loader:
            pixel_values = pixel_values.to(
                device, non_blocking=device.type == "cuda"
            )
            targets = [
                target.to(device, non_blocking=device.type == "cuda")
                for target in targets
            ]

            autocast = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if use_amp
                else nullcontext()
            )
            with autocast:
                raw = model(pixel_values)
                losses = model.compute_loss(raw, targets)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                foreach=False if device.type == "mps" else None,
            )
            optimizer.step()
            scheduler.step()

            step += 1
            for name, value in losses.items():
                running_losses[name] = running_losses.get(name, 0.0) + float(value.detach())
            if step % args.log_steps == 0:
                averages = {
                    name: value / args.log_steps for name, value in running_losses.items()
                }
                elapsed = time.monotonic() - started
                LOGGER.info(
                    "epoch=%d step=%d loss=%.4f obj=%.4f box=%.4f cls=%.4f "
                    "lr=%.2e expert_lr=%.2e elapsed=%.1fs",
                    epoch,
                    step,
                    averages["loss"],
                    averages["objectness_loss"],
                    averages["box_loss"],
                    averages["class_loss"],
                    scheduler.get_last_lr()[0],
                    scheduler.get_last_lr()[1],
                    elapsed,
                )
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
                running_losses.clear()
            if args.save_steps and step % args.save_steps == 0:
                save_checkpoint(args.output, model, config, step)

        if validation_loader is not None:
            validation_metrics = evaluate_detector(
                model,
                validation_loader,
                device,
                confidence_threshold=args.eval_confidence,
                use_amp=use_amp,
            )
            LOGGER.info(
                "validation epoch=%d mAP50=%.4f precision=%.4f recall=%.4f "
                "f1=%.4f best_f1=%.4f best_conf=%.3f",
                epoch,
                validation_metrics["map50"],
                validation_metrics["precision"],
                validation_metrics["recall"],
                validation_metrics["f1"],
                validation_metrics["best_f1"],
                validation_metrics["best_confidence"],
            )
            with metrics_path.open("a") as handle:
                handle.write(
                    json.dumps({"step": step, "epoch": epoch, "validation": validation_metrics})
                    + "\n"
                )
            if validation_metrics["map50"] > best_map50:
                best_map50 = validation_metrics["map50"]
                save_checkpoint(
                    args.output,
                    model,
                    config,
                    step,
                    name="best",
                    validation_metrics=validation_metrics,
                )

    save_checkpoint(args.output, model, config, step)
    LOGGER.info("Training complete: %d steps over %d epochs", step, args.epochs)


if __name__ == "__main__":
    main()
