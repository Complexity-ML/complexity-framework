"""Target matching for the NMS-free TR-Hash detector branch."""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import torch

from .config import TRHashDetectorConfig
from .ops import box_iou


def assign_one_to_one_targets(
    targets: List[torch.Tensor],
    *,
    student_decoded: Dict[str, torch.Tensor],
    teacher_decoded: Dict[str, torch.Tensor],
    config: TRHashDetectorConfig,
    rows: torch.Tensor,
    cols: torch.Tensor,
    denominators: torch.Tensor,
    level_offsets: Sequence[int],
    level_for_box: Callable[[float, float], int],
) -> Dict[str, torch.Tensor]:
    """Globally match each object to a unique cell using detached O2M predictions."""

    device = student_decoded["boxes"].device
    batch = len(targets)
    objectness_target = torch.zeros(batch, config.num_cells, device=device)
    box_target = torch.zeros(batch, config.num_cells, 4, device=device)
    class_target = torch.zeros(batch, config.num_cells, dtype=torch.long, device=device)
    target_indices = torch.full((batch, config.num_cells), -1, dtype=torch.long, device=device)
    positive_mask = torch.zeros(batch, config.num_cells, dtype=torch.bool, device=device)
    cell_x = (cols + 0.5) / denominators
    cell_y = (rows + 0.5) / denominators

    for image_index, image_boxes in enumerate(targets):
        if image_boxes.numel() == 0:
            continue
        edge_scores = []
        edge_cells = []
        edge_targets = []
        for target_index, image_box in enumerate(image_boxes):
            cx, cy, width, height, class_id = image_box
            preferred_level = level_for_box(float(width), float(height))
            levels = (
                range(len(config.grid_sizes))
                if config.one_to_one_multiscale_candidates
                else (preferred_level,)
            )
            candidates = []
            for level in levels:
                grid = config.grid_sizes[level]
                offset = level_offsets[level]
                level_slice = slice(offset, offset + grid * grid)
                distance_x = cols[level_slice] - (cx * grid - 0.5)
                distance_y = rows[level_slice] - (cy * grid - 0.5)
                candidate_mask = (distance_x.abs() <= config.assignment_center_radius) & (
                    distance_y.abs() <= config.assignment_center_radius
                )
                local = torch.nonzero(candidate_mask, as_tuple=False).flatten()
                if not len(local):
                    local = (distance_x.square() + distance_y.square()).argmin().reshape(1)
                candidates.append(local + offset)
            candidate_indices = torch.cat(candidates)
            target_xyxy = torch.stack(
                (
                    cx - width / 2,
                    cy - height / 2,
                    cx + width / 2,
                    cy + height / 2,
                )
            ).reshape(1, 4)
            teacher_boxes = teacher_decoded["boxes"][image_index, candidate_indices]
            teacher_iou = box_iou(teacher_boxes, target_xyxy).flatten()
            teacher_class = teacher_decoded["class_probs"][
                image_index, candidate_indices, int(class_id)
            ]
            cell_distance = (
                (cell_x[candidate_indices] - cx) * denominators[candidate_indices]
            ).square() + (
                (cell_y[candidate_indices] - cy) * denominators[candidate_indices]
            ).square()
            center_prior = torch.exp(-0.5 * cell_distance)
            alignment = (
                teacher_class.clamp_min(1e-6).pow(config.assignment_class_power)
                * (teacher_iou + 0.05).pow(config.one_to_one_iou_power)
                * center_prior
            )
            edge_scores.append(alignment)
            edge_cells.append(candidate_indices)
            edge_targets.append(torch.full_like(candidate_indices, target_index, dtype=torch.long))

        scores = torch.cat(edge_scores)
        cells = torch.cat(edge_cells)
        target_ids = torch.cat(edge_targets)
        order = torch.argsort(scores, descending=True).tolist()
        cell_values = cells.tolist()
        target_values = target_ids.tolist()
        used_cells = set()
        matched_targets = set()
        matches = {}
        for edge_index in order:
            cell = cell_values[edge_index]
            target_index = target_values[edge_index]
            if cell in used_cells or target_index in matched_targets:
                continue
            matches[target_index] = cell
            used_cells.add(cell)
            matched_targets.add(target_index)

        for target_index, image_box in enumerate(image_boxes):
            if target_index in matched_targets or len(used_cells) == config.num_cells:
                continue
            cx, cy = image_box[:2]
            distances = ((cell_x - cx) * denominators).square() + (
                (cell_y - cy) * denominators
            ).square()
            if used_cells:
                used = torch.tensor(sorted(used_cells), device=device)
                distances[used] = torch.inf
            cell = int(distances.argmin())
            matches[target_index] = cell
            used_cells.add(cell)
            matched_targets.add(target_index)

        for target_index, cell in matches.items():
            image_box = image_boxes[target_index]
            cx, cy, width, height, class_id = image_box
            target_xyxy = torch.stack(
                (
                    cx - width / 2,
                    cy - height / 2,
                    cx + width / 2,
                    cy + height / 2,
                )
            ).reshape(1, 4)
            quality = (
                box_iou(
                    student_decoded["boxes"][image_index, cell].detach().reshape(1, 4),
                    target_xyxy,
                )
                .flatten()[0]
                .clamp_min(0.2)
            )
            objectness_target[image_index, cell] = quality
            box_target[image_index, cell] = image_box[:4]
            class_target[image_index, cell] = class_id.long()
            target_indices[image_index, cell] = target_index
            positive_mask[image_index, cell] = True

    return {
        "objectness": objectness_target,
        "boxes": box_target,
        "classes": class_target,
        "target_indices": target_indices,
        "positive_mask": positive_mask,
    }
