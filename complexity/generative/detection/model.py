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
from .specialization import (
    ClassLevelHashGate,
    MultiScaleLevelAdapters,
    TemporalMotionPyramid,
)


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
                normalized_fusion=self.config.neck_normalized_fusion,
            )
        )
        self.extra_necks = nn.ModuleList(
            CrossScaleFusionNeck(
                hidden,
                len(self.config.grid_sizes),
                self.config.neck_mode,
                normalized_fusion=self.config.neck_normalized_fusion,
            )
            for _ in range(self.config.neck_repeats - 1)
        )
        self.level_adapters = (
            MultiScaleLevelAdapters(
                hidden,
                len(self.config.grid_sizes),
                self.config.level_adapter_ratio,
            )
            if self.config.level_adapters_enabled
            else None
        )
        self.video_motion = (
            TemporalMotionPyramid(
                hidden,
                len(self.config.grid_sizes),
                self.config.video_motion_hidden_size,
                self.config.video_motion_scale_init,
            )
            if self.config.video_motion_enabled
            else None
        )
        self.class_level_hash_gate = (
            ClassLevelHashGate(self.config) if self.config.class_level_hash_gate_enabled else None
        )
        self.head = DecoupledDetectionHead(self.config)
        self.one_to_one_head = (
            OneToOnePredictionHead(self.config) if self.config.end_to_end else None
        )
        if self.one_to_one_head is not None:
            self.one_to_one_head.initialize_from(self.head)
        self.register_buffer(
            "object_weight_table",
            torch.ones(self.config.num_classes, 3, 3),
            persistent=False,
        )
        self.object_contrastive_projection = (
            nn.Linear(
                self.config.resolved_head_hidden_size,
                self.config.object_contrastive_dim,
                bias=False,
            )
            if self.config.object_contrastive_loss_weight
            else None
        )

    def set_object_weight_table(self, weights: torch.Tensor) -> None:
        """Install class x size x density weights derived from the train set."""

        expected = (self.config.num_classes, 3, 3)
        if tuple(weights.shape) != expected:
            raise ValueError(f"object weight table must have shape {expected}")
        if not torch.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("object weights must be finite and positive")
        self.object_weight_table.copy_(
            weights.to(
                device=self.object_weight_table.device,
                dtype=self.object_weight_table.dtype,
            )
        )

    def _feature_pyramid(
        self,
        features: List[torch.Tensor],
        *,
        pixel_values: torch.Tensor | None = None,
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
        feature_maps = feature_maps if self.neck is None else self.neck(feature_maps)
        for extra_neck in self.extra_necks:
            feature_maps = extra_neck(feature_maps)
        if self.video_motion is not None:
            if pixel_values is None:
                raise ValueError("video motion fusion requires source pixel values")
            motion_maps = self.video_motion(
                pixel_values,
                [tuple(values.shape[-2:]) for values in feature_maps],
            )
            feature_maps = [values + motion for values, motion in zip(feature_maps, motion_maps)]
        if self.level_adapters is not None:
            feature_maps = self.level_adapters(feature_maps)
        return feature_maps

    def _tower_features(
        self,
        pixel_values: torch.Tensor,
    ) -> List[torch.Tensor]:
        if pixel_values.ndim == 5 and self.video_motion is None:
            raise ValueError("video input requires video_motion_enabled=True")
        spatial = TemporalMotionPyramid.center_frame(pixel_values)
        return self.tower(spatial)

    def _predictions_from_features(
        self,
        features: List[torch.Tensor],
        *,
        pixel_values: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Run detector heads without recomputing the shared vision tower."""

        feature_maps = self._feature_pyramid(features, pixel_values=pixel_values)

        return self.head(
            feature_maps,
            class_level_bias=self._class_level_outputs(feature_maps)[0],
            return_hidden=return_hidden,
        )

    def forward_branches(
        self,
        pixel_values: torch.Tensor,
        *,
        return_auxiliary: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor | None]
        | tuple[torch.Tensor, torch.Tensor | None, Dict[str, torch.Tensor]]
    ):
        features = self._tower_features(pixel_values)
        feature_maps = self._feature_pyramid(features, pixel_values=pixel_values)
        class_level_bias, class_level_logits = self._class_level_outputs(feature_maps)
        need_branch_hidden = self.one_to_one_head is not None or return_auxiliary
        one_to_many, hidden_outputs = self.head(
            feature_maps,
            class_level_bias=class_level_bias,
            return_branch_hidden=need_branch_hidden,
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
            one_to_one = self.one_to_one_head(
                branch_hidden,
                class_level_bias=class_level_bias,
            )
        if not return_auxiliary:
            return one_to_many, one_to_one
        assert hidden_outputs is not None
        auxiliary = {
            "classification_hidden": torch.cat(
                [classification_hidden for _, classification_hidden in hidden_outputs],
                dim=1,
            ),
        }
        if class_level_logits is not None:
            auxiliary["class_level_logits"] = class_level_logits
        return one_to_many, one_to_one, auxiliary

    def forward_predictions(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return stable one-to-many predictions."""

        features = self._tower_features(pixel_values)
        feature_maps = self._feature_pyramid(features, pixel_values=pixel_values)
        one_to_many, _ = self.head(
            feature_maps,
            class_level_bias=self._class_level_outputs(feature_maps)[0],
        )
        return one_to_many

    def _class_level_outputs(
        self,
        feature_maps: List[torch.Tensor],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.class_level_hash_gate is None:
            return None, None
        logits = self.class_level_hash_gate(feature_maps)
        log_weights = F.log_softmax(
            logits / self.config.class_level_gate_temperature,
            dim=1,
        )
        return log_weights + math.log(len(feature_maps)), logits

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        return_branches: bool = False,
        return_auxiliary: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor | None]
        | tuple[torch.Tensor, torch.Tensor | None, Dict[str, torch.Tensor]]
    ):
        """Raw one-to-many predictions, preserving the original public API."""

        branches = self.forward_branches(
            pixel_values,
            return_auxiliary=return_auxiliary,
        )
        return branches if return_branches or return_auxiliary else branches[0]

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
        """Vectorized TAL/STAL assignment with deterministic collision handling.

        Candidate construction, IoU scoring and class scoring are performed for
        every target in an image at once.  The previous implementation launched
        many tiny CUDA kernels per object, which left fast GPUs idle on dense
        COCO batches.  O2M collisions still select the highest-scoring target
        per cell (the earliest target wins exact ties), while O2O retains the
        deterministic augmenting-path matcher.
        """

        if decoded is None or not self.config.dynamic_assignment:
            return self._assign_targets_reference(
                targets,
                device,
                decoded=decoded,
                target_values=target_values,
                assignment_top_k=assignment_top_k,
                allow_stal=allow_stal,
                unique_per_target=unique_per_target,
            )

        batch = len(targets)
        grid_sizes = tuple(decoded["grid_sizes"])
        num_cells = sum(grid * grid for grid in grid_sizes)
        quality_target = torch.zeros(batch, num_cells, device=device)
        box_target = torch.zeros(batch, num_cells, 4, device=device)
        class_target = torch.zeros(batch, num_cells, dtype=torch.long, device=device)
        target_indices = torch.full((batch, num_cells), -1, dtype=torch.long, device=device)
        positive_mask = torch.zeros(batch, num_cells, dtype=torch.bool, device=device)

        rows, cols, denominators, cell_levels = self._cell_geometry(device, grid_sizes)
        rows = rows.float()
        cols = cols.float()
        denominators = denominators.float()
        anchor_x = (cols + 0.5) / denominators
        anchor_y = (rows + 0.5) / denominators

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

        unique_batches: list[tuple[int, torch.Tensor, torch.Tensor, int]] = []
        with torch.no_grad():
            for image_index, (image_boxes, image_box_values) in enumerate(
                zip(targets, target_values)
            ):
                target_count = len(image_boxes)
                if not target_count:
                    continue

                metadata = []
                for _, _, width, height, _ in image_box_values:
                    small = (
                        allow_stal
                        and self.config.stal_enabled
                        and max(width, height) <= self.config.stal_small_object_threshold
                    )
                    level = 0 if small else self._level_for_box(width, height, grid_sizes)
                    radius = (
                        self.config.stal_center_radius
                        if small
                        else self.config.assignment_center_radius
                    )
                    requested = assignment_top_k or self.config.assignment_top_k
                    if small and assignment_top_k is None:
                        requested = max(requested, self.config.stal_top_k)
                    metadata.append((level, radius, requested))

                target_boxes = image_boxes[:, :4].detach().float()
                classes = image_boxes[:, 4].long()
                levels = torch.tensor(
                    [value[0] for value in metadata], device=device, dtype=torch.long
                )
                radii = torch.tensor(
                    [value[1] for value in metadata], device=device, dtype=torch.float32
                )
                requested_top_k = torch.tensor(
                    [value[2] for value in metadata], device=device, dtype=torch.long
                )
                target_grids = torch.tensor(
                    [grid_sizes[value[0]] for value in metadata],
                    device=device,
                    dtype=torch.float32,
                )

                cx, cy, width, height = target_boxes.unbind(-1)
                center_col = cx[:, None] * target_grids[:, None] - 0.5
                center_row = cy[:, None] * target_grids[:, None] - 0.5
                same_level = levels[:, None] == cell_levels[None, :]
                center_mask = ((cols[None, :] - center_col).abs() <= radii[:, None]) & (
                    (rows[None, :] - center_row).abs() <= radii[:, None]
                )
                inside_box = (
                    (anchor_x[None, :] >= cx[:, None] - width[:, None] / 2)
                    & (anchor_x[None, :] <= cx[:, None] + width[:, None] / 2)
                    & (anchor_y[None, :] >= cy[:, None] - height[:, None] / 2)
                    & (anchor_y[None, :] <= cy[:, None] + height[:, None] / 2)
                )
                candidate_mask = same_level & center_mask & inside_box

                missing = ~candidate_mask.any(dim=1)
                if missing.any():
                    distances = (cols[None, :] - center_col).square() + (
                        rows[None, :] - center_row
                    ).square()
                    distances = distances.masked_fill(~same_level, float("inf"))
                    has_inside = (same_level & inside_box).any(dim=1)
                    prefer_inside = has_inside[:, None] & ~inside_box
                    distances = distances.masked_fill(prefer_inside, float("inf"))
                    fallback_cells = distances.argmin(dim=1)
                    missing_rows = torch.nonzero(missing, as_tuple=False).flatten()
                    candidate_mask[missing_rows, fallback_cells[missing_rows]] = True

                target_xyxy = torch.stack(
                    (
                        cx - width / 2,
                        cy - height / 2,
                        cx + width / 2,
                        cy + height / 2,
                    ),
                    dim=-1,
                )
                predicted_boxes = decoded["boxes"][image_index].detach().float()
                left_top = torch.maximum(target_xyxy[:, None, :2], predicted_boxes[None, :, :2])
                right_bottom = torch.minimum(target_xyxy[:, None, 2:], predicted_boxes[None, :, 2:])
                intersection = (right_bottom - left_top).clamp_min(0).prod(dim=-1)
                target_area = ((target_xyxy[:, 2:] - target_xyxy[:, :2]).clamp_min(0).prod(dim=-1))[
                    :, None
                ]
                predicted_area = (
                    (predicted_boxes[:, 2:] - predicted_boxes[:, :2]).clamp_min(0).prod(dim=-1)
                )[None, :]
                ious = intersection / (target_area + predicted_area - intersection).clamp_min(1e-7)
                class_scores = decoded["class_scores"][image_index, :, classes].T.detach().float()
                alignment = class_scores.clamp_min(1e-6).pow(
                    self.config.assignment_class_power
                ) * ious.clamp_min(1e-6).pow(self.config.assignment_iou_power)
                masked_alignment = alignment.masked_fill(~candidate_mask, float("-inf"))
                candidate_counts = candidate_mask.sum(dim=1)

                if unique_per_target:
                    max_radius = max(
                        self.config.assignment_center_radius,
                        self.config.stal_center_radius if allow_stal else 0.0,
                    )
                    max_slots = min(
                        num_cells,
                        max(1, (math.ceil(2.0 * max_radius) + 2) ** 2),
                    )
                    candidate_scores, candidate_cells = masked_alignment.topk(
                        max_slots, dim=1, largest=True, sorted=True
                    )
                    candidate_quality = ious.gather(1, candidate_cells).clamp_min(0.05)
                    counts = candidate_counts.clamp_max(max_slots)
                    count_column = counts[:, None].expand(-1, max_slots).float()
                    packed = torch.stack(
                        (
                            candidate_cells.float(),
                            candidate_scores.float(),
                            candidate_quality.float(),
                            count_column,
                        ),
                        dim=-1,
                    )
                    unique_batches.append((image_index, image_boxes, packed, target_count))
                    continue

                max_top_k = min(num_cells, int(requested_top_k.max().item()))
                selected_scores, selected_cells = masked_alignment.topk(
                    max_top_k, dim=1, largest=True, sorted=True
                )
                selected_quality = ious.gather(1, selected_cells).clamp_min(0.05)
                kept_counts = torch.minimum(candidate_counts, requested_top_k)
                kept = torch.arange(max_top_k, device=device)[None, :] < kept_counts[:, None]
                score_matrix = alignment.new_full((target_count, num_cells), -1.0)
                score_matrix.scatter_(
                    1,
                    selected_cells,
                    torch.where(kept, selected_scores, selected_scores.new_full((), -1.0)),
                )
                quality_matrix = ious.new_zeros((target_count, num_cells))
                quality_matrix.scatter_(
                    1,
                    selected_cells,
                    torch.where(kept, selected_quality, selected_quality.new_zeros(())),
                )
                winning_scores, winning_targets = score_matrix.max(dim=0)
                selected_mask = winning_scores >= 0.0
                selected_cells_final = torch.nonzero(selected_mask, as_tuple=False).flatten()
                selected_targets = winning_targets[selected_cells_final]
                quality_target[image_index, selected_cells_final] = quality_matrix[
                    selected_targets, selected_cells_final
                ]
                box_target[image_index, selected_cells_final] = image_boxes[selected_targets, :4]
                class_target[image_index, selected_cells_final] = classes[selected_targets]
                target_indices[image_index, selected_cells_final] = selected_targets
                positive_mask[image_index, selected_cells_final] = True

        if unique_batches:
            packed_values = torch.cat([value[2] for value in unique_batches], dim=0).cpu().tolist()
            packed_cursor = 0
            for image_index, image_boxes, _, target_count in unique_batches:
                unique_candidates: Dict[int, list[tuple[int, float, float]]] = {}
                for target_index in range(target_count):
                    row = packed_values[packed_cursor]
                    packed_cursor += 1
                    candidate_count = int(row[0][3])
                    unique_candidates[target_index] = [
                        (int(cell), score, quality)
                        for cell, score, quality, _ in row[:candidate_count]
                    ]
                cell_owner: Dict[int, int] = {}
                target_match: Dict[int, tuple[int, float, float]] = {}

                def match_target(target_index: int, visited_cells: set[int]) -> bool:
                    for cell_index, score, quality in unique_candidates[target_index]:
                        if cell_index in visited_cells:
                            continue
                        visited_cells.add(cell_index)
                        previous_target = cell_owner.get(cell_index)
                        if previous_target is None or match_target(previous_target, visited_cells):
                            cell_owner[cell_index] = target_index
                            target_match[target_index] = (cell_index, score, quality)
                            return True
                    return False

                target_order = sorted(
                    unique_candidates,
                    key=lambda index: (-unique_candidates[index][0][1], index),
                )
                for target_index in target_order:
                    match_target(target_index, set())
                for target_index, (cell_index, _, quality) in target_match.items():
                    image_box = image_boxes[target_index]
                    quality_target[image_index, cell_index] = quality
                    box_target[image_index, cell_index] = image_box[:4]
                    class_target[image_index, cell_index] = image_box[4].long()
                    target_indices[image_index, cell_index] = target_index
                    positive_mask[image_index, cell_index] = True

        return {
            "quality": quality_target,
            "boxes": box_target,
            "classes": class_target,
            "target_indices": target_indices,
            "positive_mask": positive_mask,
        }

    def _assign_targets_reference(
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
        for image_index, (image_boxes, image_box_values) in enumerate(zip(targets, target_values)):
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
                unique_batches.append((image_index, image_boxes, unique_candidate_tensors))

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
    ) -> tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
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
        object_weights = self._object_cell_weights(assigned, targets, positive, raw)
        positive_weights = assigned["quality"] * object_weights * positive
        num_positive = positive_weights.sum().clamp_min(1.0)
        batch_indices, cell_indices = torch.nonzero(positive, as_tuple=True)
        has_positive = batch_indices.numel() > 0

        quality_targets = raw.new_zeros(raw.size(0), raw.size(1), self.config.num_classes)
        if has_positive:
            quality_targets[
                batch_indices,
                cell_indices,
                assigned["classes"][positive],
            ] = assigned[
                "quality"
            ][positive].to(raw.dtype)
        quality_loss = quality_focal_loss(
            decoded["class_logits"],
            quality_targets,
            beta=self.config.quality_focal_beta,
            weights=object_weights[..., None],
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
        losses = {
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
        return losses, assigned, quality_targets, object_weights

    def _object_cell_weights(
        self,
        assigned: Dict[str, torch.Tensor],
        targets: List[torch.Tensor],
        positive: torch.Tensor,
        raw: torch.Tensor,
    ) -> torch.Tensor:
        weights = raw.new_ones(raw.shape[:2])
        if not self.config.object_weighting_enabled or not positive.any():
            return weights
        batch_indices, cell_indices = torch.nonzero(positive, as_tuple=True)
        classes = assigned["classes"][positive]
        areas = assigned["boxes"][positive][:, 2:].prod(dim=-1)
        size_bins = torch.bucketize(
            areas,
            raw.new_tensor((0.02, 0.15)),
        )
        density_per_image = torch.tensor(
            [len(image_targets) for image_targets in targets],
            device=raw.device,
        )
        density_bins = torch.bucketize(
            density_per_image[batch_indices],
            torch.tensor((3, 10), device=raw.device),
        )
        weights[batch_indices, cell_indices] = self.object_weight_table[
            classes,
            size_bins,
            density_bins,
        ].to(raw.dtype)
        return weights

    def compute_loss(
        self,
        raw: (
            torch.Tensor
            | tuple[torch.Tensor, torch.Tensor | None]
            | tuple[torch.Tensor, torch.Tensor | None, Dict[str, torch.Tensor]]
        ),
        targets: List[torch.Tensor],
        *,
        training_progress: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute the stable one-to-many detector loss.

        ``targets`` is a length-``batch`` list of ``[N_i, 5]`` tensors in
        normalized ``(cx, cy, w, h, class_id)`` format.
        """

        auxiliary = None
        if isinstance(raw, tuple):
            if len(raw) == 3:
                one_to_many, one_to_one, auxiliary = raw
            else:
                one_to_many, one_to_one = raw
        else:
            one_to_many, one_to_one = raw, None
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
        losses, assigned, quality_targets, object_weights = self._compute_branch_loss(
            one_to_many,
            targets,
            training_progress=training_progress,
            decoded=one_to_many_decoded,
            target_values=target_values,
        )
        losses["one_to_many_loss"] = losses["loss"]
        losses["one_to_many_monitor_loss"] = losses["monitor_loss"]
        if one_to_one is not None:
            one_to_one_losses, _, _, _ = self._compute_branch_loss(
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
        auxiliary_total = one_to_many.sum() * 0.0
        if self.config.level_aux_loss_weight:
            level_aux = self._level_auxiliary_loss(
                one_to_many_decoded,
                quality_targets,
                object_weights,
            )
            losses["level_aux_loss"] = level_aux
            auxiliary_total = auxiliary_total + self.config.level_aux_loss_weight * level_aux
        if self.config.gate_calibration_loss_weight:
            if auxiliary is None or "class_level_logits" not in auxiliary:
                raise ValueError("gate calibration requires forward(return_auxiliary=True)")
            gate_loss = self._gate_calibration_loss(
                auxiliary["class_level_logits"],
                assigned,
                object_weights,
                one_to_many_decoded,
            )
            losses["gate_calibration_loss"] = gate_loss
            auxiliary_total = auxiliary_total + self.config.gate_calibration_loss_weight * gate_loss
        if self.config.object_contrastive_loss_weight:
            if auxiliary is None or "classification_hidden" not in auxiliary:
                raise ValueError("object contrast requires forward(return_auxiliary=True)")
            contrastive = self._object_contrastive_loss(
                auxiliary["classification_hidden"],
                assigned,
            )
            losses["object_contrastive_loss"] = contrastive
            auxiliary_total = (
                auxiliary_total + self.config.object_contrastive_loss_weight * contrastive
            )
        losses["auxiliary_loss"] = auxiliary_total
        losses["loss"] = losses["loss"] + auxiliary_total
        losses["monitor_loss"] = losses["monitor_loss"] + auxiliary_total
        return losses

    def _level_auxiliary_loss(
        self,
        decoded: Dict[str, torch.Tensor],
        quality_targets: torch.Tensor,
        object_weights: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        offset = 0
        for grid_size in decoded["grid_sizes"]:
            cells = grid_size * grid_size
            level_slice = slice(offset, offset + cells)
            losses.append(
                quality_focal_loss(
                    decoded["class_logits"][:, level_slice],
                    quality_targets[:, level_slice],
                    beta=self.config.quality_focal_beta,
                    weights=object_weights[:, level_slice, None],
                )
            )
            offset += cells
        return torch.stack(losses).mean()

    def _gate_calibration_loss(
        self,
        logits: torch.Tensor,
        assigned: Dict[str, torch.Tensor],
        object_weights: torch.Tensor,
        decoded: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        positive = assigned["positive_mask"]
        if not positive.any():
            return logits.sum() * 0.0
        batch_indices, cell_indices = torch.nonzero(positive, as_tuple=True)
        levels = decoded["levels"][cell_indices]
        classes = assigned["classes"][positive]
        values = assigned["quality"][positive] * object_weights[positive]
        targets = logits.new_zeros(logits.shape)
        targets.index_put_(
            (batch_indices, levels, classes),
            values.to(logits.dtype),
            accumulate=True,
        )
        totals = targets.sum(dim=1, keepdim=True)
        present = totals.squeeze(1) > 0
        distributions = targets / totals.clamp_min(1e-6)
        per_class = -(distributions * logits.log_softmax(dim=1)).sum(dim=1)
        return per_class[present].mean()

    def _object_contrastive_loss(
        self,
        hidden: torch.Tensor,
        assigned: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        positive = assigned["positive_mask"]
        if positive.sum() < 2 or self.object_contrastive_projection is None:
            return hidden.sum() * 0.0
        embeddings = F.normalize(
            self.object_contrastive_projection(hidden[positive]),
            dim=-1,
        )
        labels = assigned["classes"][positive]
        similarity = embeddings @ embeddings.transpose(0, 1)
        similarity = similarity / self.config.object_contrastive_temperature
        identity = torch.eye(len(embeddings), device=hidden.device, dtype=torch.bool)
        positive_pairs = labels[:, None].eq(labels[None]) & ~identity
        valid = positive_pairs.any(dim=1)
        if not valid.any():
            return hidden.sum() * 0.0
        denominator = torch.logsumexp(similarity.masked_fill(identity, -torch.inf), dim=1)
        log_probabilities = similarity - denominator[:, None]
        per_embedding = -(
            log_probabilities.masked_fill(~positive_pairs, 0.0).sum(dim=1)
            / positive_pairs.sum(dim=1).clamp_min(1)
        )
        return per_embedding[valid].mean()

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
