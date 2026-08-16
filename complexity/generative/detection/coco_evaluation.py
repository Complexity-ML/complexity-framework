"""Official COCO prediction formatting and evaluator backends."""

from __future__ import annotations

import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

COCO_STAT_NAMES = (
    "map50_95",
    "map50",
    "map75",
    "ap_small",
    "ap_medium",
    "ap_large",
    "ar_1",
    "ar_10",
    "ar_100",
    "ar_small",
    "ar_medium",
    "ar_large",
)
COCO_EVALUATION_BACKENDS = frozenset({"auto", "pycocotools", "faster"})


def detections_to_coco(
    detections: Sequence[Mapping[str, torch.Tensor]],
    metadata: Sequence[Mapping[str, Any]],
    category_ids: Sequence[int],
) -> list[dict[str, Any]]:
    """Restore letterboxed normalized boxes and format COCO result records."""

    if len(detections) != len(metadata):
        raise ValueError("detections and metadata batch lengths differ")
    records: list[dict[str, Any]] = []
    for detection, sample in zip(detections, metadata):
        boxes = detection["boxes"].detach().float().cpu().clone()
        scores = detection["scores"].detach().float().cpu()
        labels = detection["labels"].detach().long().cpu()
        image_size = float(sample["image_size"])
        scale = float(sample["scale"])
        boxes[:, (0, 2)] = (boxes[:, (0, 2)] * image_size - float(sample["left"])) / scale
        boxes[:, (1, 3)] = (boxes[:, (1, 3)] * image_size - float(sample["top"])) / scale
        boxes[:, (0, 2)].clamp_(0.0, float(sample["original_width"]))
        boxes[:, (1, 3)].clamp_(0.0, float(sample["original_height"]))
        boxes[:, 2] -= boxes[:, 0]
        boxes[:, 3] -= boxes[:, 1]
        for box, score, label in zip(boxes, scores, labels):
            class_id = int(label)
            if not 0 <= class_id < len(category_ids):
                raise ValueError(f"prediction class ID outside COCO mapping: {class_id}")
            records.append(
                {
                    "image_id": int(sample["image_id"]),
                    "category_id": int(category_ids[class_id]),
                    "bbox": [float(value) for value in box],
                    "score": float(score),
                }
            )
    return records


def _resolve_backend(requested: str) -> str:
    if requested not in COCO_EVALUATION_BACKENDS:
        raise ValueError(f"unsupported COCO evaluator backend: {requested}")
    if requested == "auto":
        return "faster" if importlib.util.find_spec("faster_coco_eval") is not None else "pycocotools"
    return requested


def _api(backend: str) -> tuple[type[Any], type[Any]]:
    if backend == "faster":
        try:
            from faster_coco_eval import COCO, COCOeval_faster
        except ImportError as error:  # pragma: no cover - optional dependency guard
            raise RuntimeError("faster COCO evaluation requires faster-coco-eval") from error
        return COCO, COCOeval_faster
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as error:  # pragma: no cover - optional dependency guard
        raise RuntimeError("official COCO evaluation requires pycocotools") from error
    return COCO, COCOeval


def _empty_results(coco_type: type[Any], ground_truth: Any, image_ids: Sequence[int]) -> Any:
    detections = coco_type()
    selected = set(map(int, image_ids))
    detections.dataset = {
        "images": [item for item in ground_truth.dataset["images"] if int(item["id"]) in selected],
        "categories": list(ground_truth.dataset["categories"]),
        "annotations": [],
    }
    detections.createIndex()
    return detections


def _box_overlap(box: np.ndarray, targets: np.ndarray, *, crowd: bool) -> np.ndarray:
    if not len(targets):
        return np.empty(0, dtype=np.float64)
    top_left = np.maximum(box[:2], targets[:, :2])
    bottom_right = np.minimum(box[:2] + box[2:], targets[:, :2] + targets[:, 2:])
    intersection = np.maximum(bottom_right - top_left, 0.0).prod(axis=1)
    box_area = max(float(box[2] * box[3]), 0.0)
    target_area = np.maximum(targets[:, 2] * targets[:, 3], 0.0)
    denominator = box_area if crowd else box_area + target_area - intersection
    return intersection / np.maximum(denominator, 1e-12)


def _diagnostics(
    ground_truth: Any,
    predictions: Sequence[Mapping[str, Any]],
    image_ids: Sequence[int],
    confidence_threshold: float,
) -> dict[str, float]:
    """Compute lightweight IoU50 diagnostics once, independently of AP backend."""

    selected_images = set(map(int, image_ids))
    target_groups: dict[tuple[int, int], dict[str, list[list[float]]]] = {}
    total_targets = 0
    for annotation in ground_truth.dataset["annotations"]:
        image_id = int(annotation["image_id"])
        if image_id not in selected_images or annotation.get("ignore", 0):
            continue
        key = (image_id, int(annotation["category_id"]))
        group = target_groups.setdefault(key, {"regular": [], "crowd": []})
        target_kind = "crowd" if annotation.get("iscrowd", 0) else "regular"
        group[target_kind].append([float(value) for value in annotation["bbox"]])
        if target_kind == "regular":
            total_targets += 1

    prediction_groups: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for prediction in predictions:
        image_id = int(prediction["image_id"])
        if image_id in selected_images:
            key = (image_id, int(prediction["category_id"]))
            prediction_groups.setdefault(key, []).append(prediction)

    score_values = []
    match_values = []
    for key, records in prediction_groups.items():
        targets = target_groups.get(key, {"regular": [], "crowd": []})
        regular = np.asarray(targets["regular"], dtype=np.float64).reshape(-1, 4)
        crowds = np.asarray(targets["crowd"], dtype=np.float64).reshape(-1, 4)
        matched_targets = np.zeros(len(regular), dtype=bool)
        for prediction in sorted(records, key=lambda item: float(item["score"]), reverse=True):
            box = np.asarray(prediction["bbox"], dtype=np.float64)
            overlaps = _box_overlap(box, regular, crowd=False)
            overlaps[matched_targets] = -1.0
            target_index = int(np.argmax(overlaps)) if overlaps.size else -1
            if target_index >= 0 and overlaps[target_index] >= 0.5:
                matched_targets[target_index] = True
                matched = True
            elif (_box_overlap(box, crowds, crowd=True) >= 0.5).any():
                continue
            else:
                matched = False
            score_values.append(float(prediction["score"]))
            match_values.append(matched)

    score_values = np.asarray(score_values, dtype=np.float64)
    match_values = np.asarray(match_values, dtype=bool)
    fixed = score_values >= confidence_threshold
    true_positives = int(match_values[fixed].sum())
    prediction_count = int(fixed.sum())
    precision = true_positives / max(prediction_count, 1)
    recall = true_positives / max(total_targets, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

    if score_values.size:
        order = np.argsort(-score_values, kind="stable")
        ordered_matches = match_values[order]
        cumulative_true_positives = np.cumsum(ordered_matches)
        prediction_counts = np.arange(1, score_values.size + 1)
        f1_curve = 2.0 * cumulative_true_positives / np.maximum(
            prediction_counts + total_targets,
            1,
        )
        best_index = int(np.argmax(f1_curve))
        best_f1 = float(f1_curve[best_index])
        best_confidence = float(score_values[order[best_index]])
    else:
        best_f1 = 0.0
        best_confidence = 1.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "best_f1": best_f1,
        "best_confidence": best_confidence,
    }


def evaluate_coco_predictions(
    annotations_path: Path,
    predictions: Sequence[Mapping[str, Any]],
    image_ids: Sequence[int],
    *,
    backend: str = "auto",
    max_detections: int = 100,
    confidence_threshold: float = 0.001,
) -> dict[str, Any]:
    """Run an official COCO bbox evaluation and return machine-readable stats."""

    if max_detections != 100:
        raise ValueError("official comparable COCO evaluation requires max_detections=100")
    resolved = _resolve_backend(backend)
    coco_type, evaluator_type = _api(resolved)
    output = io.StringIO()
    with redirect_stdout(output):
        ground_truth = coco_type(str(annotations_path))
        detections = (
            ground_truth.loadRes(list(predictions))
            if predictions
            else _empty_results(coco_type, ground_truth, image_ids)
        )
        evaluator = evaluator_type(ground_truth, detections, "bbox")
        evaluator.params.imgIds = sorted(map(int, image_ids))
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    values = {
        name: float(evaluator.stats[index])
        for index, name in enumerate(COCO_STAT_NAMES)
    }
    values.update(
        _diagnostics(
            ground_truth,
            predictions,
            image_ids,
            confidence_threshold,
        )
    )
    values.update(
        {
            "coco_map50": values["map50"],
            "coco_map50_95": values["map50_95"],
            "coco_ap_small": values["ap_small"],
            "coco_ap_medium": values["ap_medium"],
            "coco_ap_large": values["ap_large"],
            "coco_ar100": values["ar_100"],
            "official_coco": True,
            "evaluator_backend": resolved,
            "evaluator_summary": output.getvalue(),
        }
    )
    return values
