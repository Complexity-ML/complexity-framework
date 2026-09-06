"""Evaluate a TR-Hash Vision v8 ONNX detector export on COCO.

The evaluator uses the deployment pipeline metadata sidecar to choose the
correct branch contract:

- O2M exports run decode, confidence filtering, and class-aware NMS.
- NMS-free exports run decode and confidence filtering without NMS.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from complexity.deploy.onnx_detector import OnnxDetectorPipeline
from complexity.deploy.onnx_detector.metadata import load_metadata
from complexity.generative.detection.coco_evaluation import evaluate_coco_predictions
from complexity.generative.detection.hub import COCO_CLASS_NAMES
from scripts.evaluate_tr_hash_coco import (
    _branch_contract,
    _checkpoint_sha256,
    _configure_determinism,
    _environment,
    _framework_commit,
    _image_list_sha256,
    _percentile,
    _sha256_file,
    _write_markdown_report,
)
from scripts.onnx_detect import provider_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--branch",
        choices=("auto", "o2m-nms", "nms-free"),
        default="auto",
        help="expected branch; auto trusts the metadata sidecar",
    )
    parser.add_argument(
        "--provider",
        action="append",
        default=None,
        help="Provider alias or ORT provider name. May be repeated or comma-separated.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "int8"),
        default="fp32",
        help="artifact precision label recorded for quantized release gates",
    )
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--nms-iou", type=float, default=None)
    parser.add_argument("--max-detections", type=int, default=None)
    parser.add_argument("--ort-intra-op-threads", type=int, default=1)
    parser.add_argument("--ort-inter-op-threads", type=int, default=1)
    parser.add_argument(
        "--eval-backend",
        choices=("auto", "pycocotools", "faster"),
        default="auto",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="non-release smoke-test limit; zero evaluates all val2017 images",
    )
    return parser.parse_args()


def _branch_name(metadata_branch: str) -> str:
    return "o2m-nms" if metadata_branch == "o2m" else metadata_branch


def _xyxy_to_xywh(values: tuple[float, float, float, float]) -> list[float]:
    x1, y1, x2, y2 = values
    return [x1, y1, x2 - x1, y2 - y1]


def _latency_summary_ms(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(values) if values else 0.0,
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "p99_ms": _percentile(values, 0.99),
        "measured_ms": sum(values),
    }


def _run_onnx_branch(
    pipeline: OnnxDetectorPipeline,
    coco: Any,
    image_root: Path,
    image_ids: list[int],
    category_ids: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    preprocess_ms: list[float] = []
    inference_ms: list[float] = []
    postprocess_ms: list[float] = []
    for image_id in image_ids:
        record = coco.imgs[image_id]
        result = pipeline.predict(image_root / record["file_name"])
        preprocess_ms.append(result.timing.preprocess_ms)
        inference_ms.append(result.timing.inference_ms)
        postprocess_ms.append(result.timing.postprocess_ms)
        for detection in result.detections:
            class_id = int(detection.class_id)
            if not 0 <= class_id < len(category_ids):
                raise ValueError(f"prediction class ID outside COCO mapping: {class_id}")
            predictions.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(category_ids[class_id]),
                    "bbox": _xyxy_to_xywh(detection.box_pixel),
                    "score": float(detection.score),
                }
            )
    return predictions, {
        "preprocess": _latency_summary_ms(preprocess_ms),
        "inference": _latency_summary_ms(inference_ms),
        "postprocess": _latency_summary_ms(postprocess_ms),
        "images_per_second": (
            len(image_ids) / (sum(inference_ms) / 1000.0) if sum(inference_ms) else 0.0
        ),
    }


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("limit must be non-negative")
    _configure_determinism(args.seed)

    from pycocotools.coco import COCO

    metadata = load_metadata(args.metadata)
    if args.confidence is not None:
        metadata = replace(metadata, confidence_threshold=args.confidence)
    if args.nms_iou is not None:
        metadata = replace(metadata, iou_threshold=args.nms_iou)
    if args.max_detections is not None:
        metadata = replace(metadata, max_detections=args.max_detections)

    branch = _branch_name(metadata.branch)
    if args.branch != "auto" and args.branch != branch:
        raise ValueError(f"metadata branch is {branch!r}, not requested {args.branch!r}")

    coco = COCO(str(args.annotations))
    categories = sorted(coco.loadCats(coco.getCatIds()), key=lambda item: item["id"])
    category_names = tuple(str(category["name"]) for category in categories)
    if category_names != COCO_CLASS_NAMES:
        raise ValueError("COCO category order does not match the detector class contract")
    if metadata.class_names is not None and tuple(metadata.class_names) != COCO_CLASS_NAMES:
        raise ValueError("metadata class names do not match the COCO detector contract")
    category_ids = [int(category["id"]) for category in categories]
    image_ids = sorted(coco.getImgIds())
    if args.limit:
        image_ids = image_ids[: args.limit]
    if not args.limit and len(image_ids) != 5_000:
        raise ValueError(f"release evaluation requires 5000 val2017 images, got {len(image_ids)}")

    args.output.mkdir(parents=True, exist_ok=True)
    session = OnnxDetectorPipeline.create_session(
        args.model,
        providers=provider_names(args.provider),
        intra_op_num_threads=args.ort_intra_op_threads,
        inter_op_num_threads=args.ort_inter_op_threads,
    ).open()
    pipeline = OnnxDetectorPipeline(metadata=metadata, session=session)
    predictions, timing = _run_onnx_branch(
        pipeline,
        coco,
        args.images,
        image_ids,
        category_ids,
    )
    prediction_path = args.output / f"predictions_{branch}.json"
    prediction_path.write_text(json.dumps(predictions), encoding="utf-8")

    environment = _environment(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    environment.update(
        {
            "requested_provider": provider_names(args.provider),
            "actual_provider": session.provider_used,
            "ort_intra_op_threads": args.ort_intra_op_threads,
            "ort_inter_op_threads": args.ort_inter_op_threads,
        }
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "format_version": 1,
        "backend": "onnx",
        "precision": args.precision,
        "framework_commit": _framework_commit(),
        "checkpoint": str(args.model),
        "checkpoint_sha256": _checkpoint_sha256(args.model),
        "model": str(args.model),
        "metadata": str(args.metadata),
        "metadata_sha256": _sha256_file(args.metadata),
        "dataset": {
            "name": "coco-2017",
            "split": "val2017",
            "images": len(image_ids),
            "evaluated_images": len(image_ids),
            "image_ids": image_ids,
            "annotations": str(args.annotations),
            "annotations_sha256": _sha256_file(args.annotations),
            "image_list_sha256": _image_list_sha256(coco, image_ids),
        },
        "environment": environment,
        "protocol": {
            "image_size": metadata.image_size,
            "seed": args.seed,
            "confidence_prefilter": metadata.confidence_threshold,
            "nms_iou": metadata.iou_threshold,
            "max_detections": metadata.max_detections,
            "release_eligible": not bool(args.limit),
        },
        "branches": {
            branch: {
                "branch": branch,
                "precision": args.precision,
                "contract": _branch_contract(branch),
                "predictions": str(prediction_path),
                "detections": len(predictions),
                "metrics": evaluate_coco_predictions(
                    args.annotations,
                    predictions,
                    image_ids,
                    backend=args.eval_backend,
                    max_detections=metadata.max_detections,
                    confidence_threshold=metadata.confidence_threshold,
                ),
                "timing": timing,
            }
        },
    }
    (args.output / "evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_markdown_report(report, args.output / "evaluation.md")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
