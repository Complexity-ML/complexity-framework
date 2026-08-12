"""TR-Hash single-stage object detector — YOLO-style, anchor-free.

Predicts local LTRB box distributions and joint quality-class scores directly
per backbone patch/grid cell (no anchors or region proposals). The backbone is
the hierarchical v6 TR-Hash vision tower, so
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

from .config import TRHashDetectorConfig
from .head import DecoupledDetectionHead, OneToOnePredictionHead
from .hierarchical_tower import HierarchicalTRHashVisionTower
from .losses import distribution_focal_loss, quality_focal_loss
from .neck import CrossScaleFusionNeck
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
        self.tower = HierarchicalTRHashVisionTower(self.config)
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
        self.neck = (
            None
            if self.config.neck_mode == "baseline"
            else CrossScaleFusionNeck(
                hidden,
                len(self.config.grid_sizes),
                self.config.neck_mode,
            )
        )
        self.head = DecoupledDetectionHead(self.config)
        self.one_to_one_head = (
            OneToOnePredictionHead(self.config) if self.config.end_to_end else None
        )
        if self.one_to_one_head is not None:
            self.one_to_one_head.initialize_from(self.head)

    def _feature_pyramid(
        self,
        features: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Build the prediction pyramid from already-encoded patch features."""

        source_maps = list(features)
        feature_map = source_maps[0]
        feature_maps = []
        if self.fpn_upsample is not None:
            fine_map = F.interpolate(
                feature_map, scale_factor=2.0, mode="bilinear", align_corners=False
            )
            feature_maps.append(self.fpn_upsample(fine_map))
        feature_maps.extend(source_maps)
        return feature_maps if self.neck is None else self.neck(feature_maps)

    def _predictions_from_features(
        self,
        features: List[torch.Tensor],
        *,
        return_hidden: bool = False,
    ) -> tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Run detector heads without recomputing the shared vision tower."""

        feature_maps = self._feature_pyramid(features)

        return self.head(feature_maps, return_hidden=return_hidden)

    def forward_branches(
        self,
        pixel_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        features = self.tower(pixel_values)
        feature_maps = self._feature_pyramid(features)
        one_to_many, hidden_outputs = self.head(
            feature_maps,
            return_branch_hidden=self.one_to_one_head is not None,
        )
        one_to_one = None
        if self.one_to_one_head is not None:
            assert hidden_outputs is not None
            gradient_scale = self.config.one_to_one_shared_gradient_scale
            if gradient_scale:
                branch_hidden = [
                    tuple(
                        hidden.detach() + gradient_scale * (hidden - hidden.detach())
                        for hidden in pair
                    )
                    for pair in hidden_outputs
                ]
            else:
                branch_hidden = [
                    tuple(hidden.detach() for hidden in pair) for pair in hidden_outputs
                ]
            one_to_one = self.one_to_one_head(branch_hidden)
        return one_to_many, one_to_one

    def forward_predictions(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return stable one-to-many predictions."""

        features = self.tower(pixel_values)
        one_to_many, _ = self._predictions_from_features(features)
        return one_to_many

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        return_branches: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """Raw one-to-many predictions, preserving the original public API."""

        branches = self.forward_branches(pixel_values)
        return branches if return_branches else branches[0]

    def _grid_sizes_for_raw(self, raw: torch.Tensor) -> tuple[int, ...]:
        """Infer the runtime pyramid geometry for variable-resolution inputs."""

        cells = raw.size(1)
        if cells == self.config.num_cells:
            return self.config.grid_sizes
        for base_grid in range(1, self.config.grid_size + 1):
            grids = tuple(
                (base_grid + factor - 1) // factor for factor in self.config.scale_factors
            )
            if self.config.p2_head:
                grids = (base_grid * 2, *grids)
            if sum(grid * grid for grid in grids) == cells:
                return grids
        raise ValueError(f"cannot infer prediction pyramid geometry from {cells} cells")

    def _cell_geometry(
        self,
        device: torch.device,
        grid_sizes: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        grid_sizes = grid_sizes or self.config.grid_sizes
        rows = []
        cols = []
        denominators = []
        levels = []
        for level, grid in enumerate(grid_sizes):
            rows.append(torch.arange(grid, device=device).repeat_interleave(grid))
            cols.append(torch.arange(grid, device=device).repeat(grid))
            denominators.append(torch.full((grid * grid,), grid, device=device))
            levels.append(torch.full((grid * grid,), level, device=device, dtype=torch.long))
        return tuple(torch.cat(values) for values in (rows, cols, denominators, levels))

    def _level_offsets(
        self,
        grid_sizes: tuple[int, ...] | None = None,
    ) -> tuple[int, ...]:
        grid_sizes = grid_sizes or self.config.grid_sizes
        offsets = [0]
        for grid in grid_sizes[:-1]:
            offsets.append(offsets[-1] + grid * grid)
        return tuple(offsets)

    def decode(self, raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Decode stride-local LTRB distributions and quality-class scores.

        Returns normalized ``boxes``/``boxes_cxcywh`` and independent sigmoid
        ``class_scores`` shaped ``[batch, num_cells, num_classes]``.
        """

        regression = raw[..., : self.config.regression_width]
        class_logits = raw[..., self.config.regression_width :]
        grid_sizes = self._grid_sizes_for_raw(raw)
        rows, cols, denominators, levels = self._cell_geometry(raw.device, grid_sizes)
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
            "grid_sizes": grid_sizes,
        }

    def _level_for_box(
        self,
        width: float,
        height: float,
        grid_sizes: tuple[int, ...],
    ) -> int:
        desired_grid = self.config.assignment_object_cells / max(width, height, 1e-6)
        distances = [abs(math.log(grid / desired_grid)) for grid in grid_sizes]
        return min(range(len(distances)), key=distances.__getitem__)

    def _assign_targets(
        self,
        targets: List[torch.Tensor],
        device: torch.device,
        decoded: Optional[Dict[str, torch.Tensor]] = None,
        *,
        target_values: Optional[List[list[list[float]]]] = None,
        assignment_top_k: Optional[int] = None,
        allow_stal: bool = True,
        unique_per_target: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Build per-cell training targets from ``[N_i, 5]`` (cx, cy, w, h, class_id) boxes.

        With dynamic assignment enabled, candidate cells near each object are
        ranked using class confidence and predicted box IoU. Positive class
        targets receive detached IoU quality, calibrating classification and
        localization in one score.
        """

        batch = len(targets)
        grid_sizes = tuple(decoded["grid_sizes"]) if decoded is not None else self.config.grid_sizes
        num_cells = sum(grid * grid for grid in grid_sizes)
        quality_target = torch.zeros(batch, num_cells, device=device)
        box_target = torch.zeros(batch, num_cells, 4, device=device)
        class_target = torch.zeros(batch, num_cells, dtype=torch.long, device=device)
        target_indices = torch.full((batch, num_cells), -1, dtype=torch.long, device=device)
        positive_mask = torch.zeros(batch, num_cells, dtype=torch.bool, device=device)
        assignment_score = torch.full((batch, num_cells), -1.0, device=device)
        rows, cols, _, _ = self._cell_geometry(device, grid_sizes)
        offsets = self._level_offsets(grid_sizes)

        if target_values is None:
            target_counts = [len(image_boxes) for image_boxes in targets]
            non_empty_targets = [image_boxes for image_boxes in targets if len(image_boxes)]
            flat_target_values = (
                torch.cat(non_empty_targets, dim=0).detach().float().cpu().tolist()
                if non_empty_targets
                else []
            )
            target_values = []
            cursor = 0
            for target_count in target_counts:
                target_values.append(flat_target_values[cursor : cursor + target_count])
                cursor += target_count

        unique_batches: list[tuple[int, torch.Tensor, list[torch.Tensor]]] = []
        for image_index, (image_boxes, image_box_values) in enumerate(
            zip(targets, target_values)
        ):
            if image_boxes.numel() == 0:
                continue
            unique_candidate_tensors: list[torch.Tensor] = []
            # Materialize scalar box metadata once per image. Converting CUDA
            # scalars with float()/int() inside the object loop otherwise
            # introduces several host synchronizations for every target.
            for target_index, (image_box, image_box_value) in enumerate(
                zip(image_boxes, image_box_values)
            ):
                cx, cy, width, height, class_id = image_box
                _, _, width_value, height_value, class_id_value = image_box_value
                small_object = (
                    allow_stal
                    and self.config.stal_enabled
                    and max(width_value, height_value) <= self.config.stal_small_object_threshold
                )
                level = (
                    0
                    if small_object
                    else self._level_for_box(width_value, height_value, grid_sizes)
                )
                grid = grid_sizes[level]
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
                        image_index, candidate_indices, int(class_id_value)
                    ].detach()
                    alignment = class_scores.clamp_min(1e-6).pow(
                        self.config.assignment_class_power
                    ) * ious.clamp_min(1e-6).pow(self.config.assignment_iou_power)
                    if unique_per_target:
                        selected = torch.argsort(alignment, descending=True)
                        candidate_scores = alignment[selected]
                    else:
                        requested_top_k = assignment_top_k or self.config.assignment_top_k
                        if small_object and assignment_top_k is None:
                            requested_top_k = max(requested_top_k, self.config.stal_top_k)
                        top_k = min(requested_top_k, len(candidate_indices))
                        candidate_scores, selected = alignment.topk(top_k)
                    candidate_indices = candidate_indices[selected]
                    candidate_quality = ious[selected].clamp_min(0.05)

                if unique_per_target:
                    unique_candidate_tensors.append(
                        torch.stack(
                            (
                                candidate_indices.float(),
                                candidate_scores.float(),
                                candidate_quality.float(),
                            ),
                            dim=-1,
                        )
                    )
                    continue

                replace = candidate_scores > assignment_score[image_index, candidate_indices]
                selected_cells = candidate_indices[replace]
                assignment_score[image_index, selected_cells] = candidate_scores[replace]
                quality_target[image_index, selected_cells] = candidate_quality[replace]
                box_target[image_index, selected_cells] = image_box[:4]
                class_target[image_index, selected_cells] = class_id.long()
                target_indices[image_index, selected_cells] = target_index
                positive_mask[image_index, selected_cells] = True

            if unique_per_target:
                unique_batches.append(
                    (image_index, image_boxes, unique_candidate_tensors)
                )

        if unique_batches:
            # Materialize candidates from every image with one device-to-host
            # synchronization, then perform the deterministic matching on CPU.
            all_candidate_tensors = [
                candidate_tensor
                for _, _, candidate_tensors in unique_batches
                for candidate_tensor in candidate_tensors
            ]
            flat_candidates = torch.cat(all_candidate_tensors, dim=0).cpu().tolist()
            cursor = 0
            for image_index, image_boxes, candidate_tensors in unique_batches:
                unique_candidates: Dict[int, list[tuple[int, float, float]]] = {}
                for target_index, candidate_tensor in enumerate(candidate_tensors):
                    candidate_count = len(candidate_tensor)
                    unique_candidates[target_index] = [
                        (int(cell_index), score, quality)
                        for cell_index, score, quality in flat_candidates[
                            cursor : cursor + candidate_count
                        ]
                    ]
                    cursor += candidate_count
                cell_owner: Dict[int, int] = {}
                target_match: Dict[int, tuple[int, float, float]] = {}

                def match_target(target_index: int, visited_cells: set[int]) -> bool:
                    for cell_index, score, quality in unique_candidates[target_index]:
                        if cell_index in visited_cells:
                            continue
                        visited_cells.add(cell_index)
                        previous_target = cell_owner.get(cell_index)
                        if previous_target is None or match_target(
                            previous_target,
                            visited_cells,
                        ):
                            cell_owner[cell_index] = target_index
                            target_match[target_index] = (cell_index, score, quality)
                            return True
                    return False

                target_order = sorted(
                    unique_candidates,
                    key=lambda index: (
                        -unique_candidates[index][0][1],
                        index,
                    ),
                )
                for target_index in target_order:
                    match_target(target_index, set())

                for target_index, (cell_index, score, quality) in target_match.items():
                    image_box = image_boxes[target_index]
                    quality_target[image_index, cell_index] = quality
                    box_target[image_index, cell_index] = image_box[:4]
                    class_target[image_index, cell_index] = image_box[4].long()
                    target_indices[image_index, cell_index] = target_index
                    positive_mask[image_index, cell_index] = True
                    assignment_score[image_index, cell_index] = score

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
        decoded: Optional[Dict[str, torch.Tensor]] = None,
        assignment_decoded: Optional[Dict[str, torch.Tensor]] = None,
        target_values: Optional[List[list[list[float]]]] = None,
        unique_per_target: bool = False,
    ) -> Dict[str, torch.Tensor]:
        decoded = decoded if decoded is not None else self.decode(raw)
        assigned = self._assign_targets(
            targets,
            raw.device,
            decoded=assignment_decoded if assignment_decoded is not None else decoded,
            target_values=target_values,
            assignment_top_k=assignment_top_k,
            allow_stal=allow_stal,
            unique_per_target=unique_per_target,
        )
        positive = assigned["positive_mask"]
        positive_weights = assigned["quality"] * positive
        num_positive = positive_weights.sum().clamp_min(1.0)
        batch_indices, cell_indices = torch.nonzero(positive, as_tuple=True)
        has_positive = batch_indices.numel() > 0

        quality_targets = raw.new_zeros(raw.size(0), raw.size(1), self.config.num_classes)
        if has_positive:
            quality_targets[
                batch_indices,
                cell_indices,
                assigned["classes"][positive],
            ] = assigned["quality"][positive].to(raw.dtype)
        quality_loss = quality_focal_loss(
            decoded["class_logits"],
            quality_targets,
            beta=self.config.quality_focal_beta,
        )

        pred_boxes = decoded["boxes_cxcywh"]

        box_l1_error = F.smooth_l1_loss(pred_boxes, assigned["boxes"], reduction="none").sum(-1)
        box_l1_loss = (box_l1_error * positive_weights).sum() / num_positive
        if has_positive:
            box_iou_error = complete_iou_loss(pred_boxes[positive], assigned["boxes"][positive])
            box_iou_loss = (box_iou_error * positive_weights[positive]).sum() / num_positive
        else:
            box_iou_loss = raw.sum() * 0.0
        dfl_loss = raw.sum() * 0.0
        if has_positive and self.config.reg_max:
            grid_sizes = tuple(decoded["grid_sizes"])
            rows, cols, denominators, _ = self._cell_geometry(raw.device, grid_sizes)
            centers_x = (cols + 0.5) / denominators
            centers_y = (rows + 0.5) / denominators
            target_boxes = assigned["boxes"]
            target_x1 = target_boxes[..., 0] - target_boxes[..., 2] / 2
            target_y1 = target_boxes[..., 1] - target_boxes[..., 3] / 2
            target_x2 = target_boxes[..., 0] + target_boxes[..., 2] / 2
            target_y2 = target_boxes[..., 1] + target_boxes[..., 3] / 2
            target_distances = (
                torch.stack(
                    (
                        centers_x[None] - target_x1,
                        centers_y[None] - target_y1,
                        target_x2 - centers_x[None],
                        target_y2 - centers_y[None],
                    ),
                    dim=-1,
                )
                * denominators[None, :, None]
            )
            regression_logits = decoded["regression_logits"].reshape(
                raw.size(0), raw.size(1), 4, self.config.dfl_bins
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
        monitor_loss = (
            self.config.quality_loss_weight * quality_loss + self.config.box_loss_weight * box_loss
        )
        return {
            "loss": total,
            "monitor_loss": monitor_loss,
            "quality_loss": quality_loss,
            "box_loss": box_loss,
            "box_l1_loss": box_l1_loss,
            "box_iou_loss": box_iou_loss,
            "dfl_loss": dfl_loss,
            "quality_weight_scale": quality_loss.new_tensor(quality_scale),
            "box_weight_scale": quality_loss.new_tensor(box_scale),
        }

    def compute_loss(
        self,
        raw: torch.Tensor | tuple[torch.Tensor, torch.Tensor | None],
        targets: List[torch.Tensor],
        *,
        training_progress: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute the stable one-to-many detector loss.

        ``targets`` is a length-``batch`` list of ``[N_i, 5]`` tensors in
        normalized ``(cx, cy, w, h, class_id)`` format.
        """

        one_to_many, one_to_one = raw if isinstance(raw, tuple) else (raw, None)
        if len(targets) != one_to_many.size(0):
            raise ValueError("one target tensor is required per batch image")
        target_counts = [len(image_boxes) for image_boxes in targets]
        non_empty_targets = [image_boxes for image_boxes in targets if len(image_boxes)]
        flat_target_values = (
            torch.cat(non_empty_targets, dim=0).detach().float().cpu().tolist()
            if non_empty_targets
            else []
        )
        target_values: List[list[list[float]]] = []
        cursor = 0
        for target_count in target_counts:
            target_values.append(flat_target_values[cursor : cursor + target_count])
            cursor += target_count
        one_to_many_decoded = self.decode(one_to_many)
        losses = self._compute_branch_loss(
            one_to_many,
            targets,
            training_progress=training_progress,
            decoded=one_to_many_decoded,
            target_values=target_values,
        )
        losses["one_to_many_loss"] = losses["loss"]
        losses["one_to_many_monitor_loss"] = losses["monitor_loss"]
        if one_to_one is not None:
            one_to_one_losses = self._compute_branch_loss(
                one_to_one,
                targets,
                assignment_top_k=1,
                allow_stal=True,
                training_progress=training_progress,
                assignment_decoded=one_to_many_decoded,
                target_values=target_values,
                unique_per_target=True,
            )
            losses["one_to_one_loss"] = one_to_one_losses["loss"]
            losses["one_to_one_monitor_loss"] = one_to_one_losses["monitor_loss"]
            progress = min(max(float(training_progress), 0.0), 1.0)
            one_to_one_weight = (
                self.config.one_to_one_loss_start
                + (self.config.one_to_one_loss_weight - self.config.one_to_one_loss_start)
                * progress
            )
            losses["one_to_one_weight"] = one_to_one_losses["loss"].new_tensor(one_to_one_weight)
            losses["loss"] = (
                losses["one_to_many_loss"] + one_to_one_weight * one_to_one_losses["loss"]
            )
            losses["monitor_loss"] = 0.5 * (
                losses["one_to_many_monitor_loss"] + losses["one_to_one_monitor_loss"]
            )
        return losses

    @torch.no_grad()
    def predict_end_to_end(
        self,
        pixel_values: torch.Tensor,
        *,
        confidence_threshold: float = 0.25,
        max_detections: int = 300,
    ) -> List[Dict[str, torch.Tensor]]:
        """Decode the one-to-one branch directly without NMS."""

        if self.one_to_one_head is None:
            raise RuntimeError("NMS-free inference requires end_to_end=True")
        _, raw = self.forward_branches(pixel_values)
        assert raw is not None
        decoded = self.decode(raw)
        results = []
        for image_index in range(pixel_values.size(0)):
            scores, labels = decoded["class_scores"][image_index].max(dim=-1)
            keep = torch.nonzero(scores >= confidence_threshold, as_tuple=False).flatten()
            if len(keep) > max_detections:
                keep = keep[scores[keep].topk(max_detections).indices]
            results.append(
                {
                    "boxes": decoded["boxes"][image_index, keep],
                    "scores": scores[keep],
                    "labels": labels[keep],
                }
            )
        return results

    @torch.no_grad()
    def predict(
        self,
        pixel_values: torch.Tensor,
        *,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        postprocess_on_cpu: bool = False,
        max_detections: int = 300,
        nms_free: bool | None = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """Decode detections, using the NMS-free branch by default when available.

        Returns a list (length = batch) of dicts with ``boxes`` (xyxy,
        normalized), ``scores``, and ``labels``.
        """

        if nms_free is None:
            nms_free = self.one_to_one_head is not None
        if nms_free:
            return self.predict_end_to_end(
                pixel_values,
                confidence_threshold=confidence_threshold,
                max_detections=max_detections,
            )

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
