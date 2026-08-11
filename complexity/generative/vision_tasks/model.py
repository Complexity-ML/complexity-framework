"""Shared TR-Hash vision models for the main perception task families.

The task variants deliberately reuse the same deterministic routed vision
tower as the detector. Dense tasks attach small convolutional decoders;
instance segmentation and oriented detection extend the detector so box,
class, and objectness knowledge transfers directly between checkpoints.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..detection import TRHashDetectorConfig, TRHashObjectDetector
from ..detection.model import class_aware_nms
from ..vision_language.vision_tower import TRHashVisionClassifier, TRHashVisionTower

VisionTask = Literal[
    "detection",
    "instance_segmentation",
    "semantic_segmentation",
    "depth",
    "classification",
    "pose",
    "obb",
]

SUPPORTED_VISION_TASKS: tuple[VisionTask, ...] = (
    "detection",
    "instance_segmentation",
    "semantic_segmentation",
    "depth",
    "classification",
    "pose",
    "obb",
)


class _DenseDecoder(nn.Module):
    """Cheap spatial decoder shared by segmentation, depth, and pose."""

    def __init__(self, hidden_size: int, output_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1, groups=hidden_size),
            nn.GELU(),
            nn.Conv2d(hidden_size, hidden_size, 1),
            nn.GELU(),
            nn.Conv2d(hidden_size, output_channels, 1),
        )

    def forward(self, features: torch.Tensor, output_size: int) -> torch.Tensor:
        output = self.layers(features)
        return F.interpolate(
            output,
            size=(output_size, output_size),
            mode="bilinear",
            align_corners=False,
        )


class _TRHashDenseVisionModel(nn.Module):
    def __init__(self, config: TRHashDetectorConfig):
        super().__init__()
        self.config = config
        self.tower = TRHashVisionTower(config.vision_tower_config())

    def spatial_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        tokens = self.tower(pixel_values)
        grid = self.config.grid_size
        return tokens.transpose(1, 2).reshape(
            tokens.size(0), self.config.vision_hidden_size, grid, grid
        )

    def num_parameters(self, trainable_only: bool = False) -> int:
        parameters = self.parameters()
        if trainable_only:
            parameters = (value for value in parameters if value.requires_grad)
        return sum(value.numel() for value in parameters)


class TRHashImageClassifier(TRHashVisionClassifier):
    """Classification variant constructed from the common detector config."""

    def __init__(self, config: TRHashDetectorConfig, num_classes: int):
        super().__init__(config.vision_tower_config(), num_classes)
        self.detector_config = config

    def num_parameters(self, trainable_only: bool = False) -> int:
        parameters = self.parameters()
        if trainable_only:
            parameters = (value for value in parameters if value.requires_grad)
        return sum(value.numel() for value in parameters)


class TRHashSemanticSegmenter(_TRHashDenseVisionModel):
    """Per-pixel semantic class prediction."""

    def __init__(self, config: TRHashDetectorConfig, num_classes: int):
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        super().__init__(config)
        self.num_classes = num_classes
        self.decoder = _DenseDecoder(config.vision_hidden_size, num_classes)

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        *,
        ignore_index: int = 255,
    ) -> Dict[str, torch.Tensor]:
        logits = self.decoder(self.spatial_features(pixel_values), self.config.image_size)
        output = {"logits": logits}
        if labels is not None:
            if labels.shape != logits.shape[:1] + logits.shape[2:]:
                raise ValueError("semantic labels must have shape [batch, height, width]")
            output["loss"] = F.cross_entropy(logits, labels.long(), ignore_index=ignore_index)
        return output


class TRHashDepthEstimator(_TRHashDenseVisionModel):
    """Positive monocular depth with a scale-invariant logarithmic loss."""

    def __init__(self, config: TRHashDetectorConfig, max_depth: Optional[float] = None):
        if max_depth is not None and max_depth <= 0:
            raise ValueError("max_depth must be positive")
        super().__init__(config)
        self.max_depth = max_depth
        self.decoder = _DenseDecoder(config.vision_hidden_size, 1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        depth_targets: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        raw = self.decoder(self.spatial_features(pixel_values), self.config.image_size)
        depth = F.softplus(raw) + 1e-4
        if self.max_depth is not None:
            depth = depth.clamp_max(self.max_depth)
        output = {"depth": depth}
        if depth_targets is not None:
            if depth_targets.ndim == 3:
                depth_targets = depth_targets.unsqueeze(1)
            if depth_targets.shape != depth.shape:
                raise ValueError("depth targets must match [batch, 1, height, width]")
            valid = depth_targets > 0
            if valid_mask is not None:
                if valid_mask.ndim == 3:
                    valid_mask = valid_mask.unsqueeze(1)
                valid = valid & valid_mask.bool()
            if not valid.any():
                output["loss"] = depth.sum() * 0.0
            else:
                delta = torch.log(depth[valid]) - torch.log(depth_targets[valid])
                output["loss"] = delta.square().mean() - 0.5 * delta.mean().square()
        return output


class TRHashPoseEstimator(_TRHashDenseVisionModel):
    """Keypoint heatmap estimator for human or generic articulated pose."""

    def __init__(self, config: TRHashDetectorConfig, num_keypoints: int):
        if num_keypoints <= 0:
            raise ValueError("num_keypoints must be positive")
        super().__init__(config)
        self.num_keypoints = num_keypoints
        self.decoder = _DenseDecoder(config.vision_hidden_size, num_keypoints)

    def forward(
        self,
        pixel_values: torch.Tensor,
        heatmap_targets: Optional[torch.Tensor] = None,
        visibility: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        logits = self.decoder(self.spatial_features(pixel_values), self.config.image_size)
        heatmaps = torch.sigmoid(logits)
        output = {"heatmaps": heatmaps, "heatmap_logits": logits}
        if heatmap_targets is not None:
            if heatmap_targets.shape != heatmaps.shape:
                raise ValueError("pose targets must match [batch, keypoints, height, width]")
            error = (heatmaps - heatmap_targets).square()
            if visibility is not None:
                if visibility.shape != heatmaps.shape[:2]:
                    raise ValueError("visibility must have shape [batch, keypoints]")
                weights = visibility.to(error.dtype)[:, :, None, None]
                output["loss"] = (error * weights).sum() / (
                    weights.sum() * heatmaps.shape[-2] * heatmaps.shape[-1]
                ).clamp_min(1.0)
            else:
                output["loss"] = error.mean()
        return output


class TRHashInstanceSegmenter(TRHashObjectDetector):
    """Detection plus prototype masks and per-detection mask coefficients."""

    def __init__(
        self,
        config: Optional[TRHashDetectorConfig] = None,
        *,
        num_prototypes: int = 16,
        mask_loss_weight: float = 1.0,
    ):
        if num_prototypes <= 0 or mask_loss_weight < 0:
            raise ValueError("invalid prototype count or mask loss weight")
        super().__init__(config)
        hidden = self.config.vision_hidden_size
        self.num_prototypes = num_prototypes
        self.mask_loss_weight = mask_loss_weight
        self.prototype_head = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, num_prototypes, 1),
        )
        self.mask_coefficient_heads = nn.ModuleList(
            nn.Linear(hidden, num_prototypes) for _ in self.config.grid_sizes
        )

    def forward_instance(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.tower(pixel_values)
        raw, one_to_one_raw, hidden = self._predictions_from_features(features, return_hidden=True)
        assert hidden is not None
        coefficients = torch.cat(
            [head(tokens) for head, tokens in zip(self.mask_coefficient_heads, hidden)],
            dim=1,
        )
        fine_grid = self.config.grid_sizes[0]
        fine = (
            hidden[0]
            .transpose(1, 2)
            .reshape(pixel_values.size(0), self.config.vision_hidden_size, fine_grid, fine_grid)
        )
        prototypes = self.prototype_head(fine)
        mask_size = max(self.config.image_size // 4, fine_grid)
        if prototypes.shape[-1] != mask_size:
            prototypes = F.interpolate(
                prototypes,
                size=(mask_size, mask_size),
                mode="bilinear",
                align_corners=False,
            )
        return {
            "raw": raw,
            "one_to_one_raw": one_to_one_raw,
            "mask_coefficients": coefficients,
            "prototypes": prototypes,
        }

    def compute_instance_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        detection_targets: List[torch.Tensor],
        mask_targets: List[torch.Tensor],
        *,
        training_progress: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        if len(mask_targets) != len(detection_targets):
            raise ValueError("mask and detection target batches must have equal length")
        raw = outputs["raw"]
        losses = self.compute_loss(
            raw,
            detection_targets,
            one_to_one_raw=outputs.get("one_to_one_raw"),
            training_progress=training_progress,
        )
        assigned = self._assign_targets(detection_targets, raw.device, decoded=self.decode(raw))
        coefficients = outputs["mask_coefficients"]
        prototypes = outputs["prototypes"]
        mask_losses = []
        for image_index, target_masks in enumerate(mask_targets):
            positive = assigned["positive_mask"][image_index]
            if not positive.any():
                continue
            target_indices = assigned["target_indices"][image_index, positive]
            if target_masks.size(0) != detection_targets[image_index].size(0):
                raise ValueError("one instance mask is required per detection target")
            resized_targets = F.interpolate(
                target_masks[:, None].float().to(raw.device),
                size=prototypes.shape[-2:],
                mode="nearest",
            )[:, 0]
            target = resized_targets[target_indices]
            mask_logits = torch.einsum(
                "np,phw->nhw",
                coefficients[image_index, positive],
                prototypes[image_index],
            )
            bce = F.binary_cross_entropy_with_logits(mask_logits, target)
            probabilities = torch.sigmoid(mask_logits)
            intersection = (probabilities * target).flatten(1).sum(1)
            denominator = probabilities.flatten(1).sum(1) + target.flatten(1).sum(1)
            dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
            mask_losses.append(bce + dice)
        mask_loss = torch.stack(mask_losses).mean() if mask_losses else raw.sum() * 0.0
        losses["mask_loss"] = mask_loss
        losses["loss"] = losses["loss"] + self.mask_loss_weight * mask_loss
        return losses

    @torch.no_grad()
    def predict_instance(
        self,
        pixel_values: torch.Tensor,
        *,
        objectness_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 100,
        mask_threshold: float = 0.5,
    ) -> List[Dict[str, torch.Tensor]]:
        """Return boxes and full-resolution binary masks for each instance."""

        outputs = self.forward_instance(pixel_values)
        raw = outputs["one_to_one_raw"]
        use_nms = raw is None
        raw = outputs["raw"] if raw is None else raw
        decoded = self.decode(raw)
        results = []
        for image_index in range(pixel_values.size(0)):
            class_confidence, labels = decoded["class_probs"][image_index].max(-1)
            scores = decoded["objectness"][image_index] * class_confidence
            candidate_indices = torch.nonzero(
                scores >= objectness_threshold, as_tuple=False
            ).flatten()
            boxes = decoded["boxes"][image_index, candidate_indices]
            selected_scores = scores[candidate_indices]
            selected_labels = labels[candidate_indices]
            if use_nms:
                local_keep = class_aware_nms(
                    boxes,
                    selected_scores,
                    selected_labels,
                    iou_threshold,
                    max_detections=max_detections,
                )
            else:
                local_keep = torch.argsort(selected_scores, descending=True)[:max_detections]
            kept = candidate_indices[local_keep]
            kept_boxes = decoded["boxes"][image_index, kept]
            mask_logits = torch.einsum(
                "np,phw->nhw",
                outputs["mask_coefficients"][image_index, kept],
                outputs["prototypes"][image_index],
            )
            masks = F.interpolate(
                mask_logits[:, None],
                size=(self.config.image_size, self.config.image_size),
                mode="bilinear",
                align_corners=False,
            )[:, 0].sigmoid()
            if len(masks):
                coordinates = torch.arange(self.config.image_size, device=masks.device) + 0.5
                x = coordinates[None, None, :]
                y = coordinates[None, :, None]
                pixel_boxes = kept_boxes * self.config.image_size
                crop = (
                    (x >= pixel_boxes[:, 0, None, None])
                    & (x < pixel_boxes[:, 2, None, None])
                    & (y >= pixel_boxes[:, 1, None, None])
                    & (y < pixel_boxes[:, 3, None, None])
                )
                masks = masks * crop
            results.append(
                {
                    "boxes": kept_boxes,
                    "masks": masks >= mask_threshold,
                    "mask_scores": masks,
                    "scores": scores[kept],
                    "labels": labels[kept],
                }
            )
        return results


class TRHashOBBDetector(TRHashObjectDetector):
    """Anchor-free oriented boxes with a periodic sine/cosine angle head."""

    def __init__(
        self,
        config: Optional[TRHashDetectorConfig] = None,
        *,
        angle_loss_weight: float = 1.0,
    ):
        if angle_loss_weight < 0:
            raise ValueError("angle_loss_weight must be non-negative")
        super().__init__(config)
        self.angle_loss_weight = angle_loss_weight
        self.angle_heads = nn.ModuleList(
            nn.Linear(self.config.vision_hidden_size, 2) for _ in self.config.grid_sizes
        )

    def forward_obb(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.tower(pixel_values)
        raw, one_to_one_raw, hidden = self._predictions_from_features(features, return_hidden=True)
        assert hidden is not None
        angle_vectors = torch.cat(
            [head(tokens) for head, tokens in zip(self.angle_heads, hidden)], dim=1
        )
        angle_vectors = F.normalize(angle_vectors, dim=-1, eps=1e-6)
        angles = torch.atan2(angle_vectors[..., 0], angle_vectors[..., 1])
        return {
            "raw": raw,
            "one_to_one_raw": one_to_one_raw,
            "angle_vectors": angle_vectors,
            "angles": angles,
        }

    def compute_obb_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[torch.Tensor],
        *,
        training_progress: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Train targets shaped ``[N, 6]`` as ``cx,cy,w,h,angle,class``."""

        if any(target.ndim != 2 or target.shape[-1] != 6 for target in targets):
            raise ValueError("OBB targets must have shape [objects, 6]")
        detection_targets = [
            torch.cat((target[:, :4], target[:, 5:6]), dim=-1) for target in targets
        ]
        raw = outputs["raw"]
        losses = self.compute_loss(
            raw,
            detection_targets,
            one_to_one_raw=outputs.get("one_to_one_raw"),
            training_progress=training_progress,
        )
        assigned = self._assign_targets(detection_targets, raw.device, decoded=self.decode(raw))
        angle_losses = []
        for image_index, target in enumerate(targets):
            positive = assigned["positive_mask"][image_index]
            if not positive.any():
                continue
            indices = assigned["target_indices"][image_index, positive]
            target_angles = target[indices, 4].to(raw.device)
            target_vectors = torch.stack((target_angles.sin(), target_angles.cos()), dim=-1)
            predicted = outputs["angle_vectors"][image_index, positive]
            angle_losses.append((1.0 - (predicted * target_vectors).sum(-1)).mean())
        angle_loss = torch.stack(angle_losses).mean() if angle_losses else raw.sum() * 0.0
        losses["angle_loss"] = angle_loss
        losses["loss"] = losses["loss"] + self.angle_loss_weight * angle_loss
        return losses

    @torch.no_grad()
    def predict_obb(
        self,
        pixel_values: torch.Tensor,
        *,
        objectness_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 300,
    ) -> List[Dict[str, torch.Tensor]]:
        outputs = self.forward_obb(pixel_values)
        raw = outputs["one_to_one_raw"]
        use_nms = raw is None
        raw = outputs["raw"] if raw is None else raw
        decoded = self.decode(raw)
        results = []
        for image_index in range(pixel_values.size(0)):
            class_confidence, labels = decoded["class_probs"][image_index].max(-1)
            scores = decoded["objectness"][image_index] * class_confidence
            candidate_indices = torch.nonzero(
                scores >= objectness_threshold, as_tuple=False
            ).flatten()
            boxes = decoded["boxes"][image_index, candidate_indices]
            selected_scores = scores[candidate_indices]
            selected_labels = labels[candidate_indices]
            if use_nms:
                local_keep = class_aware_nms(
                    boxes,
                    selected_scores,
                    selected_labels,
                    iou_threshold,
                    max_detections=max_detections,
                )
            else:
                local_keep = torch.argsort(selected_scores, descending=True)[:max_detections]
            kept = candidate_indices[local_keep]
            results.append(
                {
                    "boxes": decoded["boxes"][image_index, kept],
                    "angles": outputs["angles"][image_index, kept],
                    "scores": scores[kept],
                    "labels": labels[kept],
                }
            )
        return results


def create_vision_model(
    task: VisionTask,
    config: Optional[TRHashDetectorConfig] = None,
    *,
    num_classes: Optional[int] = None,
    num_keypoints: int = 17,
    num_prototypes: int = 16,
    max_depth: Optional[float] = None,
) -> nn.Module:
    """Create a TR-Hash model variant with a uniform task selector."""

    if task not in SUPPORTED_VISION_TASKS:
        raise ValueError(f"unsupported vision task: {task}")
    config = config or TRHashDetectorConfig()
    classes = config.num_classes if num_classes is None else num_classes
    if task == "detection":
        return TRHashObjectDetector(config)
    if task == "instance_segmentation":
        return TRHashInstanceSegmenter(config, num_prototypes=num_prototypes)
    if task == "semantic_segmentation":
        return TRHashSemanticSegmenter(config, classes)
    if task == "depth":
        return TRHashDepthEstimator(config, max_depth=max_depth)
    if task == "classification":
        return TRHashImageClassifier(config, classes)
    if task == "pose":
        return TRHashPoseEstimator(config, num_keypoints)
    if task == "obb":
        return TRHashOBBDetector(config)
    raise AssertionError("unreachable task selector")
