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
    assignment, a decoupled head, stride-local LTRB/DFL regression, and
    unified sigmoid quality-class scores are enabled by default.
    """

    architecture_version: int = 5
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
    neck_mode: str = "pan"
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
    reg_max: int = 16
    head_hidden_size: int = 0
    dfl_loss_weight: float = 0.5
    quality_focal_beta: float = 2.0
    progressive_loss_enabled: bool = True
    progressive_box_start: float = 0.5
    progressive_quality_start: float = 1.5
    box_loss_weight: float = 5.0
    quality_loss_weight: float = 1.0
    box_l1_weight: float = 0.25
    box_iou_weight: float = 1.0
    dropout: float = 0.0
    layer_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        object.__setattr__(self, "scale_factors", tuple(self.scale_factors))
        if self.architecture_version != 5:
            raise ValueError("only detector architecture_version=5 is supported")
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
        if self.neck_mode not in {"baseline", "fpn", "pan"}:
            raise ValueError("neck_mode must be baseline, fpn, or pan")
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
        if self.reg_max < 0:
            raise ValueError("reg_max must be non-negative")
        if self.reg_max == 1:
            raise ValueError("reg_max must be 0 (disabled) or at least 2")
        if self.head_hidden_size < 0:
            raise ValueError("head_hidden_size must be non-negative")
        if self.dfl_loss_weight < 0.0 or self.quality_focal_beta < 0.0:
            raise ValueError("DFL and QFL parameters must be non-negative")
        if not 0.0 < self.progressive_box_start <= 1.0:
            raise ValueError("progressive_box_start must be in (0, 1]")
        if self.progressive_quality_start < 1.0:
            raise ValueError("progressive_quality_start must be at least 1")

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_cells(self) -> int:
        return sum(grid**2 for grid in self.grid_sizes)

    @property
    def dfl_bins(self) -> int:
        return self.reg_max + 1 if self.reg_max else 1

    @property
    def regression_width(self) -> int:
        return 4 * self.dfl_bins

    @property
    def prediction_width(self) -> int:
        return self.regression_width + self.num_classes

    @property
    def resolved_head_hidden_size(self) -> int:
        return self.head_hidden_size or max(32, self.vision_hidden_size // 2)

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
        if values.get("architecture_version") != 5:
            raise ValueError("only TR-Hash detector architecture v5 checkpoints are supported")
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown detector config fields: {unknown}")
        return cls(**values)
