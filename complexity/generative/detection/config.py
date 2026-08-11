"""Configuration for the TR-Hash single-stage object detector."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Tuple

from ..vision_language.vision_tower import TRHashVisionTowerConfig


@dataclass(frozen=True)
class TRHashDetectorConfig:
    """Architecture contract for a YOLO-style anchor-free detector.

    Predictions use the backbone patch grid and optional lightweight
    depthwise downsampled grids. Multi-scale output, dynamic positive-cell
    assignment, and Varifocal objectness are enabled by default.
    """

    image_size: int = 224
    patch_size: int = 16
    vision_hidden_size: int = 384
    vision_layers: int = 6
    vision_heads: int = 6
    vision_num_experts: int = 4
    vision_top_k: int = 2
    vision_shared_width: int = 0
    vision_expert_width: int = 96
    vision_precision: str = "bf16"
    route_seed: int = 0x71D5A17
    num_classes: int = 80
    multi_scale: bool = True
    p2_head: bool = False
    scale_factors: Tuple[int, ...] = (1, 2, 4)
    dynamic_assignment: bool = True
    assignment_top_k: int = 5
    assignment_center_radius: float = 2.5
    assignment_object_cells: float = 4.0
    assignment_class_power: float = 1.0
    assignment_iou_power: float = 6.0
    stal_enabled: bool = True
    stal_small_object_threshold: float = 0.08
    stal_top_k: int = 9
    stal_center_radius: float = 3.5
    center_offset_mode: str = "linear"
    # Retained only so older checkpoint configs deserialize. The failed
    # one-to-one experiment is no longer constructed or trained.
    end_to_end: bool = False
    progressive_loss_enabled: bool = True
    progressive_box_start: float = 0.5
    progressive_objectness_start: float = 1.5
    box_loss_weight: float = 5.0
    objectness_loss_weight: float = 1.0
    class_loss_weight: float = 1.0
    box_l1_weight: float = 0.25
    box_iou_weight: float = 1.0
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    objectness_loss_type: str = "varifocal"
    varifocal_alpha: float = 0.75
    varifocal_gamma: float = 2.0
    class_label_smoothing: float = 0.0
    dropout: float = 0.0
    layer_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        object.__setattr__(self, "scale_factors", tuple(self.scale_factors))
        object.__setattr__(self, "end_to_end", False)
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if self.image_size <= 0 or self.patch_size <= 0:
            raise ValueError("image_size and patch_size must be positive")
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.vision_hidden_size % self.vision_heads:
            raise ValueError("vision_hidden_size must be divisible by vision_heads")
        if self.vision_top_k > self.vision_num_experts:
            raise ValueError("vision_top_k cannot exceed vision_num_experts")
        if self.vision_precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("vision_precision must be fp32, bf16, or fp16")
        if not self.scale_factors or self.scale_factors[0] != 1:
            raise ValueError("scale_factors must start with 1")
        if self.multi_scale and self.scale_factors != tuple(
            2**level for level in range(len(self.scale_factors))
        ):
            raise ValueError("multi-scale factors must form a 1, 2, 4, ... pyramid")
        if any(factor <= 0 for factor in self.scale_factors):
            raise ValueError("scale_factors must be positive")
        if self.assignment_top_k <= 0:
            raise ValueError("assignment_top_k must be positive")
        if self.assignment_center_radius <= 0.0 or self.assignment_object_cells <= 0.0:
            raise ValueError("assignment radii must be positive")
        if not 0.0 < self.stal_small_object_threshold <= 1.0:
            raise ValueError("stal_small_object_threshold must be in (0, 1]")
        if self.stal_top_k <= 0 or self.stal_center_radius <= 0.0:
            raise ValueError("STAL top-k and center radius must be positive")
        if self.center_offset_mode not in {"linear", "sigmoid"}:
            raise ValueError("center_offset_mode must be linear or sigmoid")
        if not 0.0 < self.progressive_box_start <= 1.0:
            raise ValueError("progressive_box_start must be in (0, 1]")
        if self.progressive_objectness_start < 1.0:
            raise ValueError("progressive_objectness_start must be at least 1")
        if not 0.0 <= self.focal_alpha <= 1.0:
            raise ValueError("focal_alpha must be in [0, 1]")
        if self.focal_gamma < 0.0:
            raise ValueError("focal_gamma must be non-negative")
        if self.objectness_loss_type not in {"focal", "varifocal"}:
            raise ValueError("objectness_loss_type must be focal or varifocal")
        if not 0.0 <= self.varifocal_alpha <= 1.0 or self.varifocal_gamma < 0.0:
            raise ValueError("invalid varifocal parameters")
        if not 0.0 <= self.class_label_smoothing < 1.0:
            raise ValueError("class_label_smoothing must be in [0, 1)")

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_cells(self) -> int:
        return sum(grid**2 for grid in self.grid_sizes)

    @property
    def grid_sizes(self) -> Tuple[int, ...]:
        grids = (
            (self.grid_size,)
            if not self.multi_scale
            else tuple((self.grid_size + factor - 1) // factor for factor in self.scale_factors)
        )
        return ((self.grid_size * 2,) + grids) if self.p2_head else grids

    def vision_tower_config(self) -> TRHashVisionTowerConfig:
        return TRHashVisionTowerConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            hidden_size=self.vision_hidden_size,
            num_hidden_layers=self.vision_layers,
            num_attention_heads=self.vision_heads,
            num_experts=self.vision_num_experts,
            top_k=self.vision_top_k,
            shared_width=self.vision_shared_width,
            expert_width=self.vision_expert_width,
            precision=self.vision_precision,
            route_seed=self.route_seed,
            attention_dropout=self.dropout,
            layer_norm_eps=self.layer_norm_eps,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "TRHashDetectorConfig":
        values = dict(values)
        for deprecated in (
            "one_to_one_loss_weight",
            "one_to_one_loss_warmup_fraction",
            "one_to_one_teacher_assignment",
            "one_to_one_multiscale_candidates",
            "one_to_one_iou_power",
        ):
            values.pop(deprecated, None)
        # Checkpoints written before center-offset versioning used cell-bounded
        # sigmoid offsets. Preserve their inference semantics on load.
        values.setdefault("center_offset_mode", "sigmoid")
        # New training-only branches are opt-in for unversioned checkpoints so
        # their state dictionaries and inference outputs remain unchanged.
        values.setdefault("stal_enabled", False)
        values["end_to_end"] = False
        values.setdefault("progressive_loss_enabled", False)
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown detector config fields: {unknown}")
        return cls(**values)
