"""Evaluate a native TR-Hash detector with the official COCO API.

The script evaluates O2M + class-aware NMS and the one-to-one NMS-free branch
under the same preprocessing, image list, precision and detection budget. It
writes predictions plus a machine-readable protocol and timing report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

from complexity.generative.detection.coco_evaluation import evaluate_coco_predictions
from complexity.generative.detection.hub import (
    COCO_CLASS_NAMES,
    load_detector_checkpoint,
    preprocess_detector_image,
    restore_detector_boxes,
)
from complexity.generative.detection.provenance import (
    NATIVE_COCO_DATASET,
    read_detector_provenance,
    validate_native_random_init_provenance,
)

BRANCHES = ("o2m-nms", "nms-free")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branch", choices=("both", *BRANCHES), default="both")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument(
        "--eval-backend",
        choices=("auto", "pycocotools", "faster"),
        default="auto",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=100,
        help="COCO maxDets budget; 100 is the official comparable setting",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="non-release smoke-test limit; zero evaluates all val2017 images",
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_sha256(path: Path) -> str | None:
    return _sha256_file(path) if path.is_file() else None


def _image_list_sha256(coco: Any, image_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for image_id in image_ids:
        record = coco.imgs[image_id]
        line = (
            f"{int(image_id)}\t{record['file_name']}\t"
            f"{int(record['width'])}\t{int(record['height'])}\n"
        )
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def _framework_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _package_version(module_name: str) -> str | None:
    try:
        from importlib import metadata

        return metadata.version(module_name)
    except metadata.PackageNotFoundError:
        return None


def _environment(device: torch.device) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    cuda_device = None
    cuda_capability = None
    if cuda_available:
        current = device.index if device.index is not None else torch.cuda.current_device()
        cuda_device = torch.cuda.get_device_name(current)
        cuda_capability = torch.cuda.get_device_capability(current)
    return {
        "python": sys.version.split()[0],
        "os": platform.platform(),
        "torch": torch.__version__,
        "onnxruntime": _package_version("onnxruntime"),
        "cuda_available": cuda_available,
        "torch_cuda": torch.version.cuda,
        "cuda_device": cuda_device,
        "cuda_capability": list(cuda_capability) if cuda_capability is not None else None,
        "tensorrt": _package_version("tensorrt"),
    }


def _configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _batches(values: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    converted = boxes.clone()
    converted[:, 2] = boxes[:, 2] - boxes[:, 0]
    converted[:, 3] = boxes[:, 3] - boxes[:, 1]
    return converted


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * fraction), len(ordered) - 1)
    return ordered[index]


def _timing_summary(batch_seconds: list[float], image_count: int) -> dict[str, float]:
    per_image_ms = [seconds * 1000.0 for seconds in batch_seconds]
    total_seconds = sum(batch_seconds)
    return {
        "mean_batch_ms": statistics.fmean(per_image_ms) if per_image_ms else 0.0,
        "p50_batch_ms": _percentile(per_image_ms, 0.50),
        "p95_batch_ms": _percentile(per_image_ms, 0.95),
        "p99_batch_ms": _percentile(per_image_ms, 0.99),
        "images_per_second": image_count / total_seconds if total_seconds else 0.0,
        "measured_seconds": total_seconds,
    }


def _branches_to_run(requested: str, *, has_nms_free: bool) -> tuple[str, ...]:
    branches = BRANCHES if requested == "both" else (requested,)
    if "nms-free" in branches and not has_nms_free:
        raise ValueError("checkpoint does not contain the NMS-free branch")
    return branches


def _branch_contract(branch: str) -> dict[str, Any]:
    return {
        "raw_output": "[batch, 34000, 148]",
        "regression": "68 LTRB/DFL logits",
        "classification": "80 quality-class logits",
        "postprocess": (
            "decode + confidence filtering + class-aware NMS"
            if branch == "o2m-nms"
            else "decode + confidence filtering, no NMS"
        ),
    }


def _run_branch(
    model: torch.nn.Module,
    coco: Any,
    image_root: Path,
    image_ids: list[int],
    category_ids: list[int],
    *,
    branch: str,
    device: torch.device,
    precision: str,
    batch_size: int,
    warmup: int,
    confidence: float,
    nms_iou: float,
    max_detections: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    results: list[dict[str, Any]] = []
    batch_times: list[float] = []
    batches = list(_batches(image_ids, batch_size))
    if not batches:
        return results, _timing_summary([], 0)

    def prepare(batch_ids: list[int]) -> tuple[torch.Tensor, list[Any]]:
        tensors = []
        metadata = []
        for image_id in batch_ids:
            record = coco.imgs[image_id]
            with Image.open(image_root / record["file_name"]) as image:
                values, geometry = preprocess_detector_image(image, model.config.image_size)
            tensors.append(values)
            metadata.append(geometry)
        return torch.stack(tensors).to(device), metadata

    first_values, _ = prepare(batches[0])

    def precision_context():
        if device.type == "cuda" and precision == "bf16":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    with torch.inference_mode():
        for _ in range(warmup):
            with precision_context():
                if branch == "nms-free":
                    model.predict_end_to_end(
                        first_values,
                        confidence_threshold=confidence,
                        max_detections=max_detections,
                    )
                else:
                    model.predict(
                        first_values,
                        confidence_threshold=confidence,
                        iou_threshold=nms_iou,
                        max_detections=max_detections,
                        nms_free=False,
                    )
        _synchronize(device)

        for batch_ids in batches:
            values, metadata = prepare(batch_ids)
            _synchronize(device)
            started = time.perf_counter()
            with precision_context():
                detections = (
                    model.predict_end_to_end(
                        values,
                        confidence_threshold=confidence,
                        max_detections=max_detections,
                    )
                    if branch == "nms-free"
                    else model.predict(
                        values,
                        confidence_threshold=confidence,
                        iou_threshold=nms_iou,
                        max_detections=max_detections,
                        nms_free=False,
                    )
                )
            _synchronize(device)
            batch_times.append(time.perf_counter() - started)
            for image_id, detection, geometry in zip(batch_ids, detections, metadata):
                boxes = restore_detector_boxes(detection["boxes"].float().cpu(), geometry)
                boxes = _xyxy_to_xywh(boxes)
                scores = detection["scores"].float().cpu()
                labels = detection["labels"].long().cpu()
                for box, score, label in zip(boxes, scores, labels):
                    results.append(
                        {
                            "image_id": image_id,
                            "category_id": category_ids[int(label)],
                            "bbox": [float(value) for value in box],
                            "score": float(score),
                        }
                    )
    return results, _timing_summary(batch_times, len(image_ids))


def _write_markdown_report(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Vision v8 COCO Accuracy Report",
        "",
        f"- Backend: `{report['backend']}`",
        f"- Framework commit: `{report['framework_commit']}`",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Checkpoint SHA-256: `{report['checkpoint_sha256'] or 'not recorded'}`",
        (
            f"- Dataset: `{report['dataset']['name']}` `{report['dataset']['split']}` "
            f"({report['dataset']['evaluated_images']} images)"
        ),
        f"- Annotation SHA-256: `{report['dataset']['annotations_sha256']}`",
        f"- Image-list SHA-256: `{report['dataset']['image_list_sha256']}`",
        "",
        "## Branch Metrics",
        "",
        "| Branch | AP | AP50 | AP75 | APs | APm | APl | AR100 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for branch, branch_report in report["branches"].items():
        metrics = branch_report["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    branch,
                    f"{float(metrics['map50_95']):.6f}",
                    f"{float(metrics['map50']):.6f}",
                    f"{float(metrics['map75']):.6f}",
                    f"{float(metrics['ap_small']):.6f}",
                    f"{float(metrics['ap_medium']):.6f}",
                    f"{float(metrics['ap_large']):.6f}",
                    f"{float(metrics['ar_100']):.6f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Runtime",
            "",
            f"- Python: `{report['environment']['python']}`",
            f"- OS: `{report['environment']['os']}`",
            f"- PyTorch: `{report['environment']['torch']}`",
            f"- ONNX Runtime: `{report['environment']['onnxruntime'] or 'not installed'}`",
            f"- CUDA available: `{report['environment']['cuda_available']}`",
            f"- CUDA runtime: `{report['environment']['torch_cuda'] or 'not available'}`",
            f"- TensorRT: `{report['environment']['tensorrt'] or 'not installed'}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.warmup < 0 or args.max_detections <= 0:
        raise ValueError("batch size/max detections must be positive and warmup non-negative")
    _configure_determinism(args.seed)
    provenance = read_detector_provenance(args.checkpoint)
    validate_native_random_init_provenance(provenance, dataset=NATIVE_COCO_DATASET)
    from pycocotools.coco import COCO

    coco = COCO(str(args.annotations))
    categories = sorted(coco.loadCats(coco.getCatIds()), key=lambda item: item["id"])
    category_names = tuple(str(category["name"]) for category in categories)
    if category_names != COCO_CLASS_NAMES:
        raise ValueError("COCO category order does not match the detector class contract")
    category_ids = [int(category["id"]) for category in categories]
    image_ids = sorted(coco.getImgIds())
    if args.limit:
        image_ids = image_ids[: args.limit]
    if not args.limit and len(image_ids) != 5_000:
        raise ValueError(f"release evaluation requires 5000 val2017 images, got {len(image_ids)}")

    device = torch.device(args.device)
    model = load_detector_checkpoint(args.checkpoint, device=device)
    if model.config.image_size != 640:
        raise ValueError("release evaluation requires a 640 px detector")
    if model.config.vision_num_experts != 4 or model.config.vision_top_k != 2:
        raise ValueError("release evaluation requires 4 experts with top-2 routing")
    branches = _branches_to_run(args.branch, has_nms_free=model.one_to_one_head is not None)
    args.output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": 1,
        "format_version": 1,
        "backend": "pytorch",
        "framework_commit": _framework_commit(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _checkpoint_sha256(args.checkpoint),
        "provenance": provenance,
        "dataset": {
            "name": NATIVE_COCO_DATASET,
            "split": "val2017",
            "images": len(image_ids),
            "evaluated_images": len(image_ids),
            "annotations": str(args.annotations),
            "annotations_sha256": _sha256_file(args.annotations),
            "image_list_sha256": _image_list_sha256(coco, image_ids),
        },
        "environment": _environment(device),
        "protocol": {
            "image_size": model.config.image_size,
            "precision": args.precision,
            "batch_size": args.batch_size,
            "warmup_batches": args.warmup,
            "seed": args.seed,
            "confidence_prefilter": args.confidence,
            "nms_iou": args.nms_iou,
            "max_detections": args.max_detections,
            "iou_thresholds": [round(0.5 + index * 0.05, 2) for index in range(10)],
            "area_ranges_px2": {
                "small": [0, 32**2],
                "medium": [32**2, 96**2],
                "large": [96**2, None],
            },
            "official_cocoeval": True,
            "evaluator_backend": args.eval_backend,
            "release_eligible": not bool(args.limit),
        },
        "branches": {},
    }
    for branch in branches:
        predictions, timing = _run_branch(
            model,
            coco,
            args.images,
            image_ids,
            category_ids,
            branch=branch,
            device=device,
            precision=args.precision,
            batch_size=args.batch_size,
            warmup=args.warmup,
            confidence=args.confidence,
            nms_iou=args.nms_iou,
            max_detections=args.max_detections,
        )
        prediction_path = args.output / f"predictions_{branch}.json"
        prediction_path.write_text(json.dumps(predictions))
        report["branches"][branch] = {
            "branch": branch,
            "contract": _branch_contract(branch),
            "predictions": str(prediction_path),
            "detections": len(predictions),
            "metrics": evaluate_coco_predictions(
                args.annotations,
                predictions,
                image_ids,
                backend=args.eval_backend,
                max_detections=args.max_detections,
                confidence_threshold=args.confidence,
            ),
            "timing": timing,
        }
        (args.output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
        _write_markdown_report(report, args.output / "evaluation.md")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
