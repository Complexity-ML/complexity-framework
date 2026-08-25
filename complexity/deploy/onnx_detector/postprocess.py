"""Confidence filtering and class-aware NMS for ONNX detector outputs."""

from __future__ import annotations

import numpy as np


def filter_by_confidence(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    conf_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return candidate detections whose score meets the inclusive threshold."""

    resolved_boxes = np.asarray(boxes, dtype=np.float32)
    resolved_scores = np.asarray(scores, dtype=np.float32)
    resolved_classes = np.asarray(classes)

    if resolved_boxes.ndim != 2 or resolved_boxes.shape[1] != 4:
        raise ValueError("boxes must have shape [N, 4]")
    if resolved_scores.shape != (resolved_boxes.shape[0],):
        raise ValueError("scores must have shape [N]")
    if resolved_classes.shape != (resolved_boxes.shape[0],):
        raise ValueError("classes must have shape [N]")

    indices = np.nonzero(resolved_scores >= conf_threshold)[0]
    return (
        resolved_boxes[indices],
        resolved_scores[indices],
        resolved_classes[indices],
        indices,
    )


def class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_threshold: float,
    max_detections: int,
) -> np.ndarray:
    """Run class-aware greedy NMS for the O2M branch."""

    if max_detections <= 0:
        raise ValueError("max_detections must be positive")

    resolved_boxes = np.asarray(boxes, dtype=np.float32)
    resolved_scores = np.asarray(scores, dtype=np.float32)
    resolved_classes = np.asarray(classes)
    if resolved_boxes.ndim != 2 or resolved_boxes.shape[1] != 4:
        raise ValueError("boxes must have shape [N, 4]")
    if resolved_scores.shape != (resolved_boxes.shape[0],):
        raise ValueError("scores must have shape [N]")
    if resolved_classes.shape != (resolved_boxes.shape[0],):
        raise ValueError("classes must have shape [N]")
    if resolved_boxes.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)

    kept: list[np.ndarray] = []
    for class_id in np.unique(resolved_classes):
        class_indices = np.nonzero(resolved_classes == class_id)[0]
        local_keep = _greedy_nms(
            resolved_boxes[class_indices],
            resolved_scores[class_indices],
            iou_threshold,
        )
        kept.append(class_indices[local_keep])

    keep = np.concatenate(kept).astype(np.int64, copy=False)
    order = np.argsort(resolved_scores[keep])[::-1]
    return keep[order[:max_detections]]


def _greedy_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    order = np.argsort(scores)[::-1]
    kept: list[int] = []

    while order.size:
        current = int(order[0])
        kept.append(current)
        if order.size == 1:
            break

        rest = order[1:]
        ious = _box_iou(boxes[current], boxes[rest])
        order = rest[ious <= iou_threshold]

    return np.asarray(kept, dtype=np.int64)


def _box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0)
    box_area = np.maximum(box[2] - box[0], 0.0) * np.maximum(box[3] - box[1], 0.0)
    boxes_area = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(
        boxes[:, 3] - boxes[:, 1], 0.0
    )
    union = box_area + boxes_area - intersection
    return intersection / np.maximum(union, 1e-7)
