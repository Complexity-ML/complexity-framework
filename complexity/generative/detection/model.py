"""TR-Hash single-stage object detector — YOLO-style, anchor-free.

Predicts objectness + box + class directly per backbone patch/grid cell
(no anchors, no region proposals). The backbone is ``TRHashVisionTower``, so
detection gets the same real multi-expert TR-Hash MoE routing as
classification — this is a detection *head* bolted onto a compliant
backbone, not a separate architecture with its own routing scheme.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..vision_language.vision_tower import TRHashVisionTower
from .config import TRHashDetectorConfig


def box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between two sets of ``xyxy`` boxes: ``[Na, 4]``, ``[Nb, 4]`` -> ``[Na, Nb]``."""

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


class TRHashObjectDetector(nn.Module):
    """Image -> per-cell (objectness, box, class) predictions."""

    def __init__(self, config: Optional[TRHashDetectorConfig] = None):
        super().__init__()
        self.config = config or TRHashDetectorConfig()
        self.tower = TRHashVisionTower(self.config.vision_tower_config())
        self.head_norm = nn.LayerNorm(
            self.config.vision_hidden_size, eps=self.config.layer_norm_eps
        )
        self.head = nn.Linear(self.config.vision_hidden_size, 5 + self.config.num_classes)
        nn.init.zeros_(self.head.bias)

        grid = self.config.grid_size
        rows = torch.arange(grid).repeat_interleave(grid)
        cols = torch.arange(grid).repeat(grid)
        self.register_buffer("cell_rows", rows, persistent=True)
        self.register_buffer("cell_cols", cols, persistent=True)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Raw per-cell predictions ``[batch, num_cells, 5 + num_classes]``."""

        features = self.tower(pixel_values)
        return self.head(self.head_norm(features))

    def decode(self, raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Turn raw head output into boxes/objectness/class probabilities.

        Returns ``boxes`` (xyxy, normalized [0, 1]), ``objectness`` (prob),
        and ``class_probs``, each shaped ``[batch, num_cells, ...]``.
        """

        grid = self.config.grid_size
        tx, ty, tw, th, obj_logit = raw[..., 0], raw[..., 1], raw[..., 2], raw[..., 3], raw[..., 4]
        class_logits = raw[..., 5:]

        cx = (self.cell_cols.to(raw.device) + torch.sigmoid(tx)) / grid
        cy = (self.cell_rows.to(raw.device) + torch.sigmoid(ty)) / grid
        w = torch.sigmoid(tw)
        h = torch.sigmoid(th)

        boxes = torch.stack(
            (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=-1
        ).clamp(0.0, 1.0)
        return {
            "boxes": boxes,
            "objectness": torch.sigmoid(obj_logit),
            "class_probs": F.softmax(class_logits, dim=-1),
        }

    def _assign_targets(
        self, targets: List[torch.Tensor], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Build per-cell training targets from ``[N_i, 5]`` (cx, cy, w, h, class_id) boxes.

        If more than one ground-truth box's center falls in the same cell,
        the last one in the list wins — a known simplification (no
        multi-object-per-cell support, same limitation classic single-scale
        YOLO has).
        """

        grid = self.config.grid_size
        batch = len(targets)
        objectness_target = torch.zeros(batch, self.config.num_cells, device=device)
        box_target = torch.zeros(batch, self.config.num_cells, 4, device=device)
        class_target = torch.zeros(batch, self.config.num_cells, dtype=torch.long, device=device)
        positive_mask = torch.zeros(batch, self.config.num_cells, dtype=torch.bool, device=device)

        for image_index, image_boxes in enumerate(targets):
            if image_boxes.numel() == 0:
                continue
            cx, cy, w, h, class_id = image_boxes.unbind(dim=-1)
            col = cx.mul(grid).floor().long().clamp(0, grid - 1)
            row = cy.mul(grid).floor().long().clamp(0, grid - 1)
            cell_index = row * grid + col

            objectness_target[image_index, cell_index] = 1.0
            box_target[image_index, cell_index, 0] = cx
            box_target[image_index, cell_index, 1] = cy
            box_target[image_index, cell_index, 2] = w
            box_target[image_index, cell_index, 3] = h
            class_target[image_index, cell_index] = class_id.long()
            positive_mask[image_index, cell_index] = True

        return {
            "objectness": objectness_target,
            "boxes": box_target,
            "classes": class_target,
            "positive_mask": positive_mask,
        }

    def compute_loss(
        self, raw: torch.Tensor, targets: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """``targets``: length-``batch`` list of ``[N_i, 5]`` (cx, cy, w, h, class_id) tensors,
        normalized [0, 1] box coordinates."""

        if len(targets) != raw.size(0):
            raise ValueError("one target tensor is required per batch image")
        assigned = self._assign_targets(targets, raw.device)
        positive = assigned["positive_mask"]
        num_positive = positive.sum().clamp_min(1)

        obj_logit = raw[..., 4]
        objectness_loss = F.binary_cross_entropy_with_logits(
            obj_logit, assigned["objectness"]
        )

        tx, ty, tw, th = raw[..., 0], raw[..., 1], raw[..., 2], raw[..., 3]
        pred_cx = (self.cell_cols.to(raw.device) + torch.sigmoid(tx)) / self.config.grid_size
        pred_cy = (self.cell_rows.to(raw.device) + torch.sigmoid(ty)) / self.config.grid_size
        pred_w = torch.sigmoid(tw)
        pred_h = torch.sigmoid(th)
        pred_boxes = torch.stack((pred_cx, pred_cy, pred_w, pred_h), dim=-1)

        box_error = F.smooth_l1_loss(pred_boxes, assigned["boxes"], reduction="none").sum(-1)
        box_loss = (box_error * positive).sum() / num_positive

        class_logits = raw[..., 5:]
        class_error = F.cross_entropy(
            class_logits.reshape(-1, self.config.num_classes),
            assigned["classes"].reshape(-1),
            reduction="none",
        ).reshape(positive.shape)
        class_loss = (class_error * positive).sum() / num_positive

        total = (
            self.config.objectness_loss_weight * objectness_loss
            + self.config.box_loss_weight * box_loss
            + self.config.class_loss_weight * class_loss
        )
        return {
            "loss": total,
            "objectness_loss": objectness_loss,
            "box_loss": box_loss,
            "class_loss": class_loss,
        }

    @torch.no_grad()
    def predict(
        self,
        pixel_values: torch.Tensor,
        *,
        objectness_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> List[Dict[str, torch.Tensor]]:
        """Decoded, NMS-filtered detections per image.

        Returns a list (length = batch) of dicts with ``boxes`` (xyxy,
        normalized), ``scores``, and ``labels``.
        """

        raw = self(pixel_values)
        decoded = self.decode(raw)
        batch = pixel_values.size(0)
        results = []
        for image_index in range(batch):
            objectness = decoded["objectness"][image_index]
            class_probs = decoded["class_probs"][image_index]
            boxes = decoded["boxes"][image_index]

            class_confidence, labels = class_probs.max(dim=-1)
            scores = objectness * class_confidence
            keep = scores >= objectness_threshold
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            kept_indices = greedy_nms(boxes, scores, iou_threshold)
            results.append(
                {
                    "boxes": boxes[kept_indices],
                    "scores": scores[kept_indices],
                    "labels": labels[kept_indices],
                }
            )
        return results

    def num_parameters(self, trainable_only: bool = False) -> int:
        parameters = self.parameters()
        if trainable_only:
            parameters = (parameter for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)
