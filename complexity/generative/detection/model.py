"""TR-Hash single-stage object detector — YOLO-style, anchor-free.

Predicts objectness + box + class directly per backbone patch/grid cell
(no anchors, no region proposals). The backbone is ``TRHashVisionTower``, so
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
from .ops import box_iou, class_aware_nms
from .ops import greedy_nms as greedy_nms


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    """Binary focal loss, averaged over all grid cells."""

    cross_entropy = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probabilities = torch.sigmoid(logits)
    probability_target = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    loss = cross_entropy * (1.0 - probability_target).pow(gamma)
    alpha_target = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_target * loss).mean()


def varifocal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    """Quality-aware objectness loss with IoU-like target values in ``[0, 1]``."""

    probabilities = torch.sigmoid(logits)
    negative_weight = alpha * probabilities.pow(gamma) * (targets <= 0).to(logits.dtype)
    positive_weight = targets * (targets > 0).to(logits.dtype)
    weight = negative_weight + positive_weight
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * weight
    return loss.sum() / (targets.sum().clamp_min(1.0))


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
        output_width = 5 + self.config.num_classes
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
        self.scale_heads = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(hidden, eps=self.config.layer_norm_eps),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, output_width),
            )
            for _ in self.config.grid_sizes
        )
        for head in self.scale_heads:
            self._initialize_prediction_layer(head[-1])

    @staticmethod
    def _initialize_prediction_layer(layer: nn.Linear) -> None:
        nn.init.zeros_(layer.bias)
        with torch.no_grad():
            layer.bias[4] = math.log(0.01 / 0.99)

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

        outputs = []
        hidden_outputs = []
        for level, (head, feature_map) in enumerate(zip(self.scale_heads, feature_maps)):
            tokens = feature_map.flatten(2).transpose(1, 2)
            hidden_tokens = head[2](head[1](head[0](tokens)))
            if return_hidden:
                hidden_outputs.append(hidden_tokens)
            outputs.append(head[3](hidden_tokens))
        one_to_many = torch.cat(outputs, dim=1)
        return one_to_many, hidden_outputs if return_hidden else None

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
        """Turn raw head output into boxes/objectness/class probabilities.

        Returns ``boxes`` (xyxy, normalized [0, 1]), ``objectness`` (prob),
        and ``class_probs``, each shaped ``[batch, num_cells, ...]``.
        """

        tx, ty, tw, th, obj_logit = raw[..., 0], raw[..., 1], raw[..., 2], raw[..., 3], raw[..., 4]
        class_logits = raw[..., 5:]
        rows, cols, denominators, levels = self._cell_geometry(raw.device)
        if self.config.center_offset_mode == "linear":
            # YOLOX-style unconstrained center offsets. Dynamic assignment may
            # mark neighbouring cells as positives, so a positive cell must be
            # able to regress an object center outside its own boundaries.
            cx = (cols + 0.5 + tx) / denominators
            cy = (rows + 0.5 + ty) / denominators
        else:
            # Compatibility path for checkpoints trained before center-offset
            # versioning, whose logits were learned with cell-bounded centers.
            cx = (cols + torch.sigmoid(tx)) / denominators
            cy = (rows + torch.sigmoid(ty)) / denominators
        w = torch.sigmoid(tw)
        h = torch.sigmoid(th)
        boxes_cxcywh = torch.stack((cx, cy, w, h), dim=-1)
        boxes = torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=-1).clamp(
            0.0, 1.0
        )
        return {
            "boxes": boxes,
            "boxes_cxcywh": boxes_cxcywh,
            "objectness": torch.sigmoid(obj_logit),
            "class_probs": F.softmax(class_logits, dim=-1),
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
        ranked using class confidence and predicted box IoU. Their objectness
        targets are the detached prediction quality, which calibrates scores.
        """

        batch = len(targets)
        objectness_target = torch.zeros(batch, self.config.num_cells, device=device)
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
                    local_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
                    if not len(local_indices):
                        distances = (level_cols - center_col).square() + (
                            level_rows - center_row
                        ).square()
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
                    class_scores = decoded["class_probs"][
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
                    candidate_quality = ious[selected].clamp_min(0.2)

                replace = candidate_scores > assignment_score[image_index, candidate_indices]
                selected_cells = candidate_indices[replace]
                assignment_score[image_index, selected_cells] = candidate_scores[replace]
                objectness_target[image_index, selected_cells] = candidate_quality[replace]
                box_target[image_index, selected_cells] = image_box[:4]
                class_target[image_index, selected_cells] = class_id.long()
                target_indices[image_index, selected_cells] = target_index
                positive_mask[image_index, selected_cells] = True

        return {
            "objectness": objectness_target,
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
        positive_weights = assigned["objectness"] * positive
        num_positive = positive_weights.sum().clamp_min(1.0)

        obj_logit = raw[..., 4]
        if self.config.objectness_loss_type == "varifocal":
            objectness_loss = varifocal_loss(
                obj_logit,
                assigned["objectness"],
                alpha=self.config.varifocal_alpha,
                gamma=self.config.varifocal_gamma,
            )
        else:
            objectness_loss = sigmoid_focal_loss(
                obj_logit,
                assigned["objectness"],
                alpha=self.config.focal_alpha,
                gamma=self.config.focal_gamma,
            )

        pred_boxes = decoded["boxes_cxcywh"]

        box_l1_error = F.smooth_l1_loss(pred_boxes, assigned["boxes"], reduction="none").sum(-1)
        box_l1_loss = (box_l1_error * positive_weights).sum() / num_positive
        if positive.any():
            box_iou_error = complete_iou_loss(pred_boxes[positive], assigned["boxes"][positive])
            box_iou_loss = (box_iou_error * positive_weights[positive]).sum() / num_positive
        else:
            box_iou_loss = raw.sum() * 0.0
        box_loss = (
            self.config.box_l1_weight * box_l1_loss + self.config.box_iou_weight * box_iou_loss
        )

        class_logits = raw[..., 5:]
        class_error = F.cross_entropy(
            class_logits.reshape(-1, self.config.num_classes),
            assigned["classes"].reshape(-1),
            reduction="none",
            label_smoothing=self.config.class_label_smoothing,
        ).reshape(positive.shape)
        class_loss = (class_error * positive_weights).sum() / num_positive

        progress = min(max(float(training_progress), 0.0), 1.0)
        if self.config.progressive_loss_enabled:
            box_scale = (
                self.config.progressive_box_start
                + (1.0 - self.config.progressive_box_start) * progress
            )
            objectness_scale = (
                self.config.progressive_objectness_start
                + (1.0 - self.config.progressive_objectness_start) * progress
            )
        else:
            box_scale = 1.0
            objectness_scale = 1.0
        total = (
            self.config.objectness_loss_weight * objectness_scale * objectness_loss
            + self.config.box_loss_weight * box_scale * box_loss
            + self.config.class_loss_weight * class_loss
        )
        return {
            "loss": total,
            "objectness_loss": objectness_loss,
            "box_loss": box_loss,
            "box_l1_loss": box_l1_loss,
            "box_iou_loss": box_iou_loss,
            "class_loss": class_loss,
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
        objectness_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        postprocess_on_cpu: bool = False,
        max_detections: int = 300,
        prediction_branch: str = "auto",
    ) -> List[Dict[str, torch.Tensor]]:
        """Decode detections with class-aware NMS.

        Returns a list (length = batch) of dicts with ``boxes`` (xyxy,
        normalized), ``scores``, and ``labels``.
        """

        if prediction_branch not in {"auto", "one_to_many"}:
            raise ValueError("only the one_to_many prediction branch is supported")
        raw = self.forward_predictions(pixel_values)
        if postprocess_on_cpu:
            raw = raw.cpu()
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
