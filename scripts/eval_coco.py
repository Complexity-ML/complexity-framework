"""Evaluate a TR-Hash detector on COCO val2017 with official pycocotools metrics.

Runs inference on every image in the COCO validation set, then computes
standard COCO metrics (mAP@50, mAP@50:95, AP by size) using pycocotools.

Supports both inference branches:
  --nms-free    Use the NMS-free (one-to-one) branch
  (default)     Use O2M + NMS branch

Usage:
    python eval_coco.py /path/to/checkpoint \
        --images /path/to/val2017 \
        --annotations /path/to/annotations/instances_val2017.json

    python eval_coco.py /path/to/checkpoint \
        --images /path/to/val2017 \
        --annotations /path/to/annotations/instances_val2017.json \
        --nms-free

    python eval_coco.py /path/to/checkpoint \
        --images /path/to/val2017 \
        --annotations /path/to/annotations/instances_val2017.json \
        --device cuda --iou-threshold 0.5 --max-detections 100
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

os.environ["COMPLEXITY_DISABLE_KERNELS"] = "1"

from complexity.generative.detection.hub import (
    COCO_CLASS_NAMES,
    load_detector_checkpoint,
    preprocess_detector_image,
    restore_detector_boxes,
)


# COCO category IDs are not contiguous (1-90 with gaps).
# This maps our 0-79 class indices to official COCO category IDs.
COCO_CATEGORY_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Checkpoint directory")
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Path to COCO val2017 image directory",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Path to instances_val2017.json",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.001,
        help="Low threshold to maximize recall for mAP (default: %(default)s)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="NMS IoU threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=300,
        help="Max detections per image (default: %(default)s)",
    )
    parser.add_argument(
        "--nms-free",
        action="store_true",
        help="Use NMS-free (one-to-one) branch instead of O2M + NMS",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save detailed results as JSON",
    )
    return parser.parse_args()


def run_inference(
    model: torch.nn.Module,
    image_dir: Path,
    image_ids: List[int],
    image_filenames: Dict[int, str],
    *,
    device: str = "cpu",
    confidence_threshold: float = 0.001,
    iou_threshold: float = 0.5,
    max_detections: int = 300,
    nms_free: bool = False,
) -> List[dict]:
    """Run detection on all images and collect results in COCO format."""

    coco_results = []
    image_size = model.config.image_size

    for image_id in tqdm(image_ids, desc="Running inference"):
        filename = image_filenames[image_id]
        image_path = image_dir / filename
        image = Image.open(image_path).convert("RGB")

        pixels, metadata = preprocess_detector_image(image, image_size)

        with torch.inference_mode():
            predictions = model.predict(
                pixels.unsqueeze(0).to(device),
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                max_detections=max_detections,
                nms_free=nms_free,
            )[0]

        boxes = restore_detector_boxes(predictions["boxes"].cpu(), metadata)
        scores = predictions["scores"].cpu()
        labels = predictions["labels"].cpu()

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.tolist()
            coco_results.append(
                {
                    "image_id": image_id,
                    "category_id": COCO_CATEGORY_IDS[int(label)],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO uses xywh
                    "score": float(score),
                }
            )

    return coco_results


def evaluate_coco(
    annotations_path: Path,
    coco_results: List[dict],
) -> Dict[str, float]:
    """Compute official COCO metrics using pycocotools."""

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        raise ImportError("pycocotools not installed. Run: pip install pycocotools")

    coco_gt = COCO(str(annotations_path))

    if not coco_results:
        print("Warning: no detections produced")
        return {
            "mAP50": 0.0,
            "mAP50_95": 0.0,
            "AP_small": 0.0,
            "AP_medium": 0.0,
            "AP_large": 0.0,
            "AR_1": 0.0,
            "AR_10": 0.0,
            "AR_100": 0.0,
        }

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    return {
        "mAP50_95": float(stats[0]),
        "mAP50": float(stats[1]),
        "mAP75": float(stats[2]),
        "AP_small": float(stats[3]),
        "AP_medium": float(stats[4]),
        "AP_large": float(stats[5]),
        "AR_1": float(stats[6]),
        "AR_10": float(stats[7]),
        "AR_100": float(stats[8]),
        "AR_small": float(stats[9]),
        "AR_medium": float(stats[10]),
        "AR_large": float(stats[11]),
    }


def main() -> None:
    args = parse_args()

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_detector_checkpoint(args.checkpoint, device=args.device)
    branch = "NMS-free" if args.nms_free else "O2M + NMS"
    print(
        f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, "
        f"{model.config.image_size}px, {model.config.num_classes} classes"
    )
    print(f"Inference branch: {branch}")

    if args.nms_free and model.one_to_one_head is None:
        print("Warning: model does not have a one-to-one head, falling back to O2M + NMS")
        args.nms_free = False

    if model.config.num_classes != 80:
        print(
            f"Warning: model has {model.config.num_classes} classes, "
            f"COCO expects 80. Results will not be meaningful."
        )

    # Load annotations
    print(f"Loading annotations: {args.annotations}")
    with open(args.annotations) as f:
        coco_data = json.load(f)

    image_ids = [img["id"] for img in coco_data["images"]]
    image_filenames = {img["id"]: img["file_name"] for img in coco_data["images"]}
    print(f"Images: {len(image_ids)}")

    # Run inference
    start_time = time.time()
    coco_results = run_inference(
        model,
        args.images,
        image_ids,
        image_filenames,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        max_detections=args.max_detections,
        nms_free=args.nms_free,
    )
    elapsed = time.time() - start_time
    print(
        f"\nInference complete: {len(coco_results)} detections "
        f"in {elapsed:.1f}s ({len(image_ids) / elapsed:.1f} img/s)"
    )

    # Evaluate
    print(f"\n--- Official COCO Metrics (pycocotools) [{branch}] ---")
    metrics = evaluate_coco(args.annotations, coco_results)

    print(f"\n  mAP@50:      {metrics['mAP50']:.4f}")
    print(f"  mAP@50:95:   {metrics['mAP50_95']:.4f}")
    print(f"  AP small:    {metrics['AP_small']:.4f}")
    print(f"  AP medium:   {metrics['AP_medium']:.4f}")
    print(f"  AP large:    {metrics['AP_large']:.4f}")
    print(f"  AR@100:      {metrics['AR_100']:.4f}")

    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "branch": branch,
            "metrics": metrics,
            "num_images": len(image_ids),
            "num_detections": len(coco_results),
            "inference_time_seconds": round(elapsed, 1),
            "images_per_second": round(len(image_ids) / elapsed, 1),
            "config": {
                "confidence_threshold": args.confidence_threshold,
                "iou_threshold": args.iou_threshold,
                "max_detections": args.max_detections,
                "nms_free": args.nms_free,
                "device": args.device,
            },
        }
        args.output.write_text(json.dumps(output_data, indent=2) + "\n")
        print(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()
