"""Portable box operations used by TR-Hash detection."""

from __future__ import annotations

from typing import List, Optional

import torch

try:
    from torchvision.ops import batched_nms as _torchvision_batched_nms
except (ImportError, OSError, RuntimeError):  # Optional; keep the framework backend-neutral.
    _torchvision_batched_nms = None


def box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between ``xyxy`` box sets: ``[Na, 4]``, ``[Nb, 4]`` -> ``[Na, Nb]``."""

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp_min(0) * (
        boxes_a[:, 3] - boxes_a[:, 1]
    ).clamp_min(0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp_min(0) * (
        boxes_b[:, 3] - boxes_b[:, 1]
    ).clamp_min(0)

    top_left = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return intersection / union.clamp_min(1e-9)


def greedy_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Return indices (score-descending) of boxes kept after greedy NMS."""

    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    order = torch.argsort(scores, descending=True)
    keep: List[int] = []
    while order.numel() > 0:
        current = int(order[0])
        keep.append(current)
        if order.numel() == 1:
            break
        rest = order[1:]
        ious = box_iou(boxes[current : current + 1], boxes[rest])[0]
        order = rest[ious <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def class_aware_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
    max_detections: Optional[int] = None,
) -> torch.Tensor:
    """Apply class-aware NMS and return score-sorted indices.

    Torchvision's compiled operator avoids a device synchronization for every
    candidate box. The pure PyTorch implementation remains as a portable
    fallback for installations without torchvision and for unsupported
    accelerator backends.
    """

    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    if max_detections is not None and max_detections <= 0:
        raise ValueError("max_detections must be positive")

    if _torchvision_batched_nms is not None and boxes.device.type in {"cpu", "cuda"}:
        kept_indices = _torchvision_batched_nms(
            boxes.float(), scores.float(), labels, iou_threshold
        )
        return kept_indices[:max_detections] if max_detections is not None else kept_indices

    kept = []
    for label in labels.unique():
        class_indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        class_keep = greedy_nms(boxes[class_indices], scores[class_indices], iou_threshold)
        kept.append(class_indices[class_keep])
    if not kept:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    kept_indices = torch.cat(kept)
    kept_indices = kept_indices[torch.argsort(scores[kept_indices], descending=True)]
    return kept_indices[:max_detections] if max_detections is not None else kept_indices
