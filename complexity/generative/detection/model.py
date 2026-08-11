"""TR-Hash single-stage object detector — YOLO-style, anchor-free.

Predicts local LTRB box distributions and joint quality-class scores directly
per backbone patch/grid cell (no anchors or region proposals). The backbone is
``TRHashVisionTower``, so
detection gets the same real multi-expert TR-Hash MoE routing as
classification — this is a detection *head* bolted onto a compliant
backbone, not a separate architecture with its own routing scheme.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..vision_language.vision_tower import TRHashVisionTower
from .config import TRHashDetectorConfig
from .head import DecoupledDetectionHead
from .losses import distribution_focal_loss, quality_focal_loss
from .ops import box_iou, class_aware_nms
from .ops import greedy_nms as greedy_nms


def complete_iou_loss(predicted_cxcywh: torch.Tensor, target_cxcywh: torch.Tensor) -> torch.Tensor:
    """Elementwise Complete-IoU loss for normalized ``cxcywh`` boxes."""

    pred_center = predicted_cxcywh[:, :2]
    target_center = target_cxcywh[:, :2]
    pred_size = predicted_cxcywh[:, 2:].clamp_min(1e-7)
    target_size = target_cxcywh[:, 2:].clamp_min(1e-7)
    pred_xyxy = torch.cat((pred_center - pred_size / 2, pred_center + pred_size / 2), dim=-1)
    target_xyxy = torch.cat(
        (target_center - target_size / 2, target_center + target_size / 2), dim=-1
    )

    intersection_top_left = torch.maximum(pred_xyxy[:, :2], target_xyxy[:, :2])
    intersection_bottom_right = torch.minimum(pred_xyxy[:, 2:], target_xyxy[:, 2:])
    intersection = (intersection_bottom_right - intersection_top_left).clamp_min(0).prod(-1)
    pred_area = pred_size.prod(-1)
    target_area = target_size.prod(-1)
    iou = intersection / (pred_area + target_area - intersection).clamp_min(1e-7)

    enclosing_top_left = torch.minimum(pred_xyxy[:, :2], target_xyxy[:, :2])
    enclosing_bottom_right = torch.maximum(pred_xyxy[:, 2:], target_xyxy[:, 2:])
    enclosing_diagonal = (
        (enclosing_bottom_right - enclosing_top_left).square().sum(-1).clamp_min(1e-7)
    )
    center_distance = (pred_center - target_center).square().sum(-1)

    aspect_penalty = (4.0 / math.pi**2) * (
        torch.atan(target_size[:, 0] / target_size[:, 1])
        - torch.atan(pred_size[:, 0] / pred_size[:, 1])
    ).square()
    with torch.no_grad():
        aspect_weight = aspect_penalty / (1.0 - iou + aspect_penalty).clamp_min(1e-7)
    return 1.0 - iou + center_distance / enclosing_diagonal + aspect_weight * aspect_penalty


class TRHashObjectDetector(nn.Module):
    """TR-Hash backbone with a lightweight multi-scale anchor-free head."""

    def __init__(self, config: Optional[TRHashDetectorConfig] = None):
        super().__init__()
        self.config = config or TRHashDetectorConfig()
        self.tower = TRHashVisionTower(self.config.vision_tower_config())
        hidden = self.config.vision_hidden_size
        self.fpn_upsample = (
            nn.Sequential(
                nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
                nn.GELU(),
                nn.Conv2d(hidden, hidden, 1),
            )
            if self.config.p2_head
            else None
        )
        coarse_grids = self.config.grid_sizes[1:] if self.config.p2_head else self.config.grid_sizes
        self.fpn_downsamples = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(hidden, hidden, 3, stride=2, padding=1, groups=hidden),
                nn.GELU(),
                nn.Conv2d(hidden, hidden, 1),
            )
            for _ in coarse_grids[1:]
        )
        self.head = DecoupledDetectionHead(self.config)

    def _feature_pyramid(self, features: torch.Tensor) -> List[torch.Tensor]:
        """Build the prediction pyramid from already-encoded patch features."""

        batch = features.size(0)
        grid = self.config.grid_size
        feature_map = features.transpose(1, 2).reshape(
            batch, self.config.vision_hidden_size, grid, grid
        )
        feature_maps = []
        if self.fpn_upsample is not None:
            fine_map = F.interpolate(
                feature_map, scale_factor=2.0, mode="bilinear", align_corners=False
            )
            feature_maps.append(self.fpn_upsample(fine_map))
        feature_maps.append(feature_map)
        for downsample in self.fpn_downsamples:
            feature_map = downsample(feature_map)
            feature_maps.append(feature_map)
        return feature_maps

    def _predictions_from_features(
        self,
        features: torch.Tensor,
        *,
        return_hidden: bool = False,
    ) -> tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Run detector heads without recomputing the shared vision tower."""

        feature_maps = self._feature_pyramid(features)

        return self.head(feature_maps, return_hidden=return_hidden)

    def forward_predictions(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return stable one-to-many predictions."""

        features = self.tower(pixel_values)
        one_to_many, _ = self._predictions_from_features(features)
        return one_to_many

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Raw one-to-many predictions, preserving the original public API."""

        return self.forward_predictions(pixel_values)

    def _cell_geometry(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = []
        cols = []
        denominators = []
        levels = []
        for level, grid in enumerate(self.config.grid_sizes):
            rows.append(torch.arange(grid, device=device).repeat_interleave(grid))
            cols.append(torch.arange(grid, device=device).repeat(grid))
            denominators.append(torch.full((grid * grid,), grid, device=device))
            levels.append(torch.full((grid * grid,), level, device=device, dtype=torch.long))
        return tuple(torch.cat(values) for values in (rows, cols, denominators, levels))

    def _level_offsets(self) -> tuple[int, ...]:
        offsets = [0]
        for grid in self.config.grid_sizes[:-1]:
            offsets.append(offsets[-1] + grid * grid)
        return tuple(offsets)

    def decode(self, raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Decode stride-local LTRB distributions and quality-class scores.

        Returns normalized ``boxes``/``boxes_cxcywh`` and independent sigmoid
        ``class_scores`` shaped ``[batch, num_cells, num_classes]``.
        """

        regression = raw[..., : self.config.regression_width]
        class_logits = raw[..., self.config.regression_width :]
        rows, cols, denominators, levels = self._cell_geometry(raw.device)
        centers_x = (cols + 0.5) / denominators
        centers_y = (rows + 0.5) / denominators
        if self.config.reg_max:
            distributions = regression.reshape(
                *regression.shape[:-1], 4, self.config.dfl_bins
            ).softmax(-1)
            bins = torch.arange(
                self.config.dfl_bins,
                device=raw.device,
                dtype=raw.dtype,
            )
            distances = (distributions * bins).sum(-1) / denominators[None, :, None]
        else:
            distances = F.softplus(regression) / denominators[None, :, None]
        left, top, right, bottom = distances.unbind(-1)
        boxes = torch.stack(
            (centers_x - left, centers_y - top, centers_x + right, centers_y + bottom),
            dim=-1,
        ).clamp(0.0, 1.0)
        sizes = (boxes[..., 2:] - boxes[..., :2]).clamp_min(0.0)
        centers = (boxes[..., 2:] + boxes[..., :2]) / 2
        boxes_cxcywh = torch.cat((centers, sizes), dim=-1)
        return {
            "boxes": boxes,
            "boxes_cxcywh": boxes_cxcywh,
            "class_scores": class_logits.sigmoid(),
            "class_logits": class_logits,
            "regression_logits": regression,
            "levels": levels,
        }

    def _level_for_box(self, width: float, height: float) -> int:
        desired_grid = self.config.assignment_object_cells / max(width, height, 1e-6)
        distances = [abs(math.log(grid / desired_grid)) for grid in self.config.grid_sizes]
        return min(range(len(distances)), key=distances.__getitem__)

    def _assign_targets(
        self,
        targets: List[torch.Tensor],
        device: torch.device,
        decoded: Optional[Dict[str, torch.Tensor]] = None,
        *,
        assignment_top_k: Optional[int] = None,
        allow_stal: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Build per-cell training targets from ``[N_i, 5]`` (cx, cy, w, h, class_id) boxes.

        With dynamic assignment enabled, candidate cells near each object are
        ranked using class confidence and predicted box IoU. Positive class
        targets receive detached IoU quality, calibrating classification and
        localization in one score.
        """

        batch = len(targets)
        quality_target = torch.zeros(batch, self.config.num_cells, device=device)
        box_target = torch.zeros(batch, self.config.num_cells, 4, device=device)
        class_target = torch.zeros(batch, self.config.num_cells, dtype=torch.long, device=device)
        target_indices = torch.full(
            (batch, self.config.num_cells), -1, dtype=torch.long, device=device
        )
        positive_mask = torch.zeros(batch, self.config.num_cells, dtype=torch.bool, device=device)
        assignment_score = torch.full((batch, self.config.num_cells), -1.0, device=device)
        rows, cols, _, _ = self._cell_geometry(device)
        offsets = self._level_offsets()

        for image_index, image_boxes in enumerate(targets):
            if image_boxes.numel() == 0:
                continue
            for target_index, image_box in enumerate(image_boxes):
                cx, cy, width, height, class_id = image_box
                small_object = (
                    allow_stal
                    and self.config.stal_enabled
                    and max(float(width), float(height)) <= self.config.stal_small_object_threshold
                )
                level = 0 if small_object else self._level_for_box(float(width), float(height))
                grid = self.config.grid_sizes[level]
                offset = offsets[level]
                level_slice = slice(offset, offset + grid * grid)
                level_rows = rows[level_slice]
                level_cols = cols[level_slice]

                if not self.config.dynamic_assignment or decoded is None:
                    col = int(torch.floor(cx * grid).clamp(0, grid - 1))
                    row = int(torch.floor(cy * grid).clamp(0, grid - 1))
                    candidate_indices = torch.tensor([offset + row * grid + col], device=device)
                    candidate_scores = torch.ones(1, device=device)
                    candidate_quality = torch.ones(1, device=device)
                else:
                    center_col = cx * grid - 0.5
                    center_row = cy * grid - 0.5
                    center_radius = (
                        self.config.stal_center_radius
                        if small_object
                        else self.config.assignment_center_radius
                    )
                    candidate_mask = ((level_cols - center_col).abs() <= center_radius) & (
                        (level_rows - center_row).abs() <= center_radius
                    )
                    anchor_x = (level_cols + 0.5) / grid
                    anchor_y = (level_rows + 0.5) / grid
                    inside_box = (
                        (anchor_x >= cx - width / 2)
                        & (anchor_x <= cx + width / 2)
                        & (anchor_y >= cy - height / 2)
                        & (anchor_y <= cy + height / 2)
                    )
                    candidate_mask &= inside_box
                    local_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
                    if not len(local_indices):
                        distances = (level_cols - center_col).square() + (
                            level_rows - center_row
                        ).square()
                        if inside_box.any():
                            distances = distances.masked_fill(~inside_box, float("inf"))
                        local_indices = distances.argmin().reshape(1)
                    candidate_indices = local_indices + offset
                    target_xyxy = torch.stack(
                        (
                            cx - width / 2,
                            cy - height / 2,
                            cx + width / 2,
                            cy + height / 2,
                        )
                    ).reshape(1, 4)
                    predicted_boxes = decoded["boxes"][image_index, candidate_indices].detach()
                    ious = box_iou(predicted_boxes, target_xyxy).flatten()
                    class_scores = decoded["class_scores"][
                        image_index, candidate_indices, int(class_id)
                    ].detach()
                    alignment = class_scores.clamp_min(1e-6).pow(
                        self.config.assignment_class_power
                    ) * ious.clamp_min(1e-6).pow(self.config.assignment_iou_power)
                    requested_top_k = assignment_top_k or self.config.assignment_top_k
                    if small_object and assignment_top_k is None:
                        requested_top_k = max(requested_top_k, self.config.stal_top_k)
                    top_k = min(requested_top_k, len(candidate_indices))
                    candidate_scores, selected = alignment.topk(top_k)
                    candidate_indices = candidate_indices[selected]
                    candidate_quality = ious[selected].clamp_min(0.05)

                replace = candidate_scores > assignment_score[image_index, candidate_indices]
                selected_cells = candidate_indices[replace]
                assignment_score[image_index, selected_cells] = candidate_scores[replace]
                quality_target[image_index, selected_cells] = candidate_quality[replace]
                box_target[image_index, selected_cells] = image_box[:4]
                class_target[image_index, selected_cells] = class_id.long()
                target_indices[image_index, selected_cells] = target_index
                positive_mask[image_index, selected_cells] = True

        return {
            "quality": quality_target,
            "boxes": box_target,
            "classes": class_target,
            "target_indices": target_indices,
            "positive_mask": positive_mask,
        }

    def _compute_branch_loss(
        self,
        raw: torch.Tensor,
        targets: List[torch.Tensor],
        *,
        assignment_top_k: Optional[int] = None,
        allow_stal: bool = True,
        training_progress: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        decoded = self.decode(raw)
        assigned = self._assign_targets(
            targets,
            raw.device,
            decoded=decoded,
            assignment_top_k=assignment_top_k,
            allow_stal=allow_stal,
        )
        positive = assigned["positive_mask"]
        positive_weights = assigned["quality"] * positive
        num_positive = positive_weights.sum().clamp_min(1.0)

        quality_targets = raw.new_zeros(
            raw.size(0), self.config.num_cells, self.config.num_classes
        )
        if positive.any():
            batch_indices, cell_indices = torch.nonzero(positive, as_tuple=True)
            quality_targets[
                batch_indices,
                cell_indices,
                assigned["classes"][positive],
            ] = assigned["quality"][positive]
        quality_loss = quality_focal_loss(
            decoded["class_logits"],
            quality_targets,
            beta=self.config.quality_focal_beta,
        )

        pred_boxes = decoded["boxes_cxcywh"]

        box_l1_error = F.smooth_l1_loss(pred_boxes, assigned["boxes"], reduction="none").sum(-1)
        box_l1_loss = (box_l1_error * positive_weights).sum() / num_positive
        if positive.any():
            box_iou_error = complete_iou_loss(pred_boxes[positive], assigned["boxes"][positive])
            box_iou_loss = (box_iou_error * positive_weights[positive]).sum() / num_positive
        else:
            box_iou_loss = raw.sum() * 0.0
        dfl_loss = raw.sum() * 0.0
        if positive.any() and self.config.reg_max:
            rows, cols, denominators, _ = self._cell_geometry(raw.device)
            centers_x = (cols + 0.5) / denominators
            centers_y = (rows + 0.5) / denominators
            target_boxes = assigned["boxes"]
            target_x1 = target_boxes[..., 0] - target_boxes[..., 2] / 2
            target_y1 = target_boxes[..., 1] - target_boxes[..., 3] / 2
            target_x2 = target_boxes[..., 0] + target_boxes[..., 2] / 2
            target_y2 = target_boxes[..., 1] + target_boxes[..., 3] / 2
            target_distances = torch.stack(
                (
                    centers_x[None] - target_x1,
                    centers_y[None] - target_y1,
                    target_x2 - centers_x[None],
                    target_y2 - centers_y[None],
                ),
                dim=-1,
            ) * denominators[None, :, None]
            regression_logits = decoded["regression_logits"].reshape(
                raw.size(0), self.config.num_cells, 4, self.config.dfl_bins
            )
            dfl_error = distribution_focal_loss(
                regression_logits[positive],
                target_distances[positive],
                reg_max=self.config.reg_max,
            )
            dfl_loss = (dfl_error * positive_weights[positive]).sum() / num_positive
        box_loss = (
            self.config.box_l1_weight * box_l1_loss
            + self.config.box_iou_weight * box_iou_loss
            + self.config.dfl_loss_weight * dfl_loss
        )

        progress = min(max(float(training_progress), 0.0), 1.0)
        if self.config.progressive_loss_enabled:
            box_scale = (
                self.config.progressive_box_start
                + (1.0 - self.config.progressive_box_start) * progress
            )
            quality_scale = (
                self.config.progressive_quality_start
                + (1.0 - self.config.progressive_quality_start) * progress
            )
        else:
            box_scale = 1.0
            quality_scale = 1.0
        total = (
            self.config.quality_loss_weight * quality_scale * quality_loss
            + self.config.box_loss_weight * box_scale * box_loss
        )
        return {
            "loss": total,
            "quality_loss": quality_loss,
            "box_loss": box_loss,
            "box_l1_loss": box_l1_loss,
            "box_iou_loss": box_iou_loss,
            "dfl_loss": dfl_loss,
        }

    def compute_loss(
        self,
        raw: torch.Tensor,
        targets: List[torch.Tensor],
        *,
        training_progress: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute the stable one-to-many detector loss.

        ``targets`` is a length-``batch`` list of ``[N_i, 5]`` tensors in
        normalized ``(cx, cy, w, h, class_id)`` format.
        """

        if len(targets) != raw.size(0):
            raise ValueError("one target tensor is required per batch image")
        losses = self._compute_branch_loss(
            raw,
            targets,
            training_progress=training_progress,
        )
        losses["one_to_many_loss"] = losses["loss"]
        return losses

    @torch.no_grad()
    def predict(
        self,
        pixel_values: torch.Tensor,
        *,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        postprocess_on_cpu: bool = False,
        max_detections: int = 300,
    ) -> List[Dict[str, torch.Tensor]]:
        """Decode detections with class-aware NMS.

        Returns a list (length = batch) of dicts with ``boxes`` (xyxy,
        normalized), ``scores``, and ``labels``.
        """

        raw = self.forward_predictions(pixel_values)
        if postprocess_on_cpu:
            raw = raw.cpu()
        decoded = self.decode(raw)
        batch = pixel_values.size(0)
        results = []
        for image_index in range(batch):
            class_scores = decoded["class_scores"][image_index]
            boxes = decoded["boxes"][image_index]

            scores, labels = class_scores.max(dim=-1)
            keep = scores >= confidence_threshold
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            kept_indices = class_aware_nms(
                boxes,
                scores,
                labels,
                iou_threshold,
                max_detections=max_detections,
            )
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
