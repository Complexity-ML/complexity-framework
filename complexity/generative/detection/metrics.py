"""COCO-style IoU and object-size metrics for detector validation."""

from __future__ import annotations

import math
from typing import Dict

import torch

IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))
SIZE_RANGES = {
    "small": (0.0, 32.0**2),
    "medium": (32.0**2, 96.0**2),
    "large": (96.0**2, float("inf")),
}


def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    centers, sizes = boxes[:, :2], boxes[:, 2:]
    return torch.cat((centers - sizes / 2, centers + sizes / 2), dim=-1)


def _box_areas(boxes: torch.Tensor, image_size: int) -> torch.Tensor:
    sizes = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0)
    return sizes.prod(-1) * float(image_size**2)


def _average_precision(
    scores: torch.Tensor,
    true_positives: torch.Tensor,
    total_ground_truth: int,
) -> float:
    if total_ground_truth == 0:
        return float("nan")
    if not len(scores):
        return 0.0
    order = torch.argsort(scores, descending=True)
    true_positive_cumulative = true_positives[order].cumsum(0)
    false_positive_cumulative = (1.0 - true_positives[order]).cumsum(0)
    recall = true_positive_cumulative / total_ground_truth
    precision = true_positive_cumulative / (
        true_positive_cumulative + false_positive_cumulative
    ).clamp_min(1e-9)
    recall = torch.cat((torch.tensor([0.0]), recall, torch.tensor([1.0])))
    precision = torch.cat((torch.tensor([1.0]), precision, torch.tensor([0.0])))
    precision = torch.cummax(precision.flip(0), dim=0).values.flip(0)
    changing = torch.nonzero(recall[1:] != recall[:-1], as_tuple=False).flatten()
    return float(((recall[changing + 1] - recall[changing]) * precision[changing + 1]).sum())


def _match_thresholds(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, Dict[float, torch.Tensor]]:
    order = torch.argsort(scores, descending=True)
    scores = scores[order]
    boxes = boxes[order]
    if len(boxes) and len(targets):
        top_left = torch.maximum(boxes[:, None, :2], targets[None, :, :2])
        bottom_right = torch.minimum(boxes[:, None, 2:], targets[None, :, 2:])
        intersections = (bottom_right - top_left).clamp_min(0).prod(-1)
        box_areas = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(-1)
        target_areas = (targets[:, 2:] - targets[:, :2]).clamp_min(0).prod(-1)
        ious = intersections / (
            box_areas[:, None] + target_areas[None, :] - intersections
        ).clamp_min(1e-9)
    else:
        ious = torch.empty(len(boxes), len(targets))

    matches = {}
    for threshold in IOU_THRESHOLDS:
        matched = torch.zeros(len(scores), dtype=torch.float32)
        used_targets = torch.zeros(len(targets), dtype=torch.bool)
        for prediction_index in range(len(scores)):
            if not len(targets):
                break
            available = ious[prediction_index].masked_fill(used_targets, -1.0)
            best_iou, target_index = available.max(dim=0)
            if float(best_iou) >= threshold:
                matched[prediction_index] = 1.0
                used_targets[int(target_index)] = True
        matches[threshold] = matched
    return scores, matches


class DetectionMetricsAccumulator:
    """Accumulate AP50/AP50-95 globally and for COCO pixel-area ranges."""

    def __init__(self, num_classes: int, image_size: int):
        self.num_classes = num_classes
        self.image_size = image_size
        scopes = ("all", *SIZE_RANGES)
        self.scores = {
            scope: {class_id: [] for class_id in range(num_classes)} for scope in scopes
        }
        self.matches = {
            scope: {
                threshold: {class_id: [] for class_id in range(num_classes)}
                for threshold in IOU_THRESHOLDS
            }
            for scope in scopes
        }
        self.target_counts = {
            scope: [0 for _ in range(num_classes)] for scope in scopes
        }

    def update(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        boxes = boxes.float().cpu()
        scores = scores.float().cpu()
        labels = labels.long().cpu()
        targets = targets.float().cpu()
        target_boxes = _xywh_to_xyxy(targets[:, :4])
        target_labels = targets[:, 4].long()
        box_areas = _box_areas(boxes, self.image_size)
        target_areas = _box_areas(target_boxes, self.image_size)

        scopes = {"all": (torch.ones(len(boxes), dtype=torch.bool), torch.ones(len(targets), dtype=torch.bool))}
        for name, (lower, upper) in SIZE_RANGES.items():
            scopes[name] = (
                (box_areas >= lower) & (box_areas < upper),
                (target_areas >= lower) & (target_areas < upper),
            )

        for scope, (prediction_scope, target_scope) in scopes.items():
            for class_id in range(self.num_classes):
                prediction_mask = prediction_scope & (labels == class_id)
                target_mask = target_scope & (target_labels == class_id)
                class_scores, class_matches = _match_thresholds(
                    boxes[prediction_mask],
                    scores[prediction_mask],
                    target_boxes[target_mask],
                )
                self.scores[scope][class_id].append(class_scores)
                for threshold, values in class_matches.items():
                    self.matches[scope][threshold][class_id].append(values)
                self.target_counts[scope][class_id] += int(target_mask.sum())

    @staticmethod
    def _concatenate(values: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(values) if values else torch.empty(0)

    def _map(self, scope: str, threshold: float) -> float:
        values = []
        for class_id in range(self.num_classes):
            average_precision = _average_precision(
                self._concatenate(self.scores[scope][class_id]),
                self._concatenate(self.matches[scope][threshold][class_id]),
                self.target_counts[scope][class_id],
            )
            if not math.isnan(average_precision):
                values.append(average_precision)
        return sum(values) / max(len(values), 1)

    def compute(self, confidence_threshold: float) -> Dict[str, float]:
        map_by_threshold = {
            threshold: self._map("all", threshold) for threshold in IOU_THRESHOLDS
        }
        scores = torch.cat(
            [
                self._concatenate(self.scores["all"][class_id])
                for class_id in range(self.num_classes)
            ]
        )
        matches = torch.cat(
            [
                self._concatenate(self.matches["all"][0.5][class_id])
                for class_id in range(self.num_classes)
            ]
        )
        order = torch.argsort(scores, descending=True)
        scores = scores[order]
        matches = matches[order]
        total_targets = sum(self.target_counts["all"])
        fixed = scores >= confidence_threshold
        fixed_true_positives = float(matches[fixed].sum())
        fixed_predictions = int(fixed.sum())
        precision = fixed_true_positives / max(fixed_predictions, 1)
        recall = fixed_true_positives / max(total_targets, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
        if len(scores):
            cumulative_true_positives = matches.cumsum(0)
            prediction_counts = torch.arange(1, len(scores) + 1)
            f1_curve = 2.0 * cumulative_true_positives / (
                prediction_counts + total_targets
            ).clamp_min(1)
            best_f1, best_index = f1_curve.max(dim=0)
            best_confidence = float(scores[int(best_index)])
        else:
            best_f1 = torch.tensor(0.0)
            best_confidence = 1.0

        size_ap = {
            name: sum(self._map(name, threshold) for threshold in IOU_THRESHOLDS)
            / len(IOU_THRESHOLDS)
            for name in SIZE_RANGES
        }
        return {
            "map50": map_by_threshold[0.5],
            "map50_95": sum(map_by_threshold.values()) / len(map_by_threshold),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "best_f1": float(best_f1),
            "best_confidence": best_confidence,
            "ap_small": size_ap["small"],
            "ap_medium": size_ap["medium"],
            "ap_large": size_ap["large"],
        }
