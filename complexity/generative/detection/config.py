"""Configuration for the TR-Hash single-stage object detector."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class TRHashDetectorConfig:
    """Architecture contract for a YOLO-style anchor-free detector.

    Predictions use the backbone patch grid and optional lightweight
    depthwise downsampled grids. Multi-scale output, dynamic positive-cell
    assignment, a decoupled head, stride-local LTRB/DFL regression, and
    unified sigmoid quality-class scores are enabled by default.
    """

    architecture_version: int = 6
    image_size: int = 224
    patch_size: int = 16
    vision_hidden_size: int = 384
    vision_layers: int = 6
    vision_heads: int = 6
    vision_num_experts: int = 4
    vision_top_k: int = 2
    vision_shared_width: int = 224
    vision_expert_width: int = 64
    vision_stage_depths: Tuple[int, ...] = (1, 1, 4)
    vision_window_size: int = 8
    vision_precision: str = "bf16"
    route_seed: int = 0x71D5A17
    num_classes: int = 80
    multi_scale: bool = True
    p2_head: bool = False
    neck_mode: str = "pan"
    neck_normalized_fusion: bool = False
    neck_repeats: int = 1
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
    head_spatial_mixing: bool = False
    regression_logit_scale: bool = False
    dfl_loss_weight: float = 0.5
    quality_focal_beta: float = 2.0
    end_to_end: bool = True
    one_to_one_loss_weight: float = 1.0
    one_to_one_loss_start: float = 0.25
    one_to_one_shared_gradient_scale: float = 0.25
    progressive_loss_enabled: bool = True
    progressive_box_start: float = 0.5
    progressive_quality_start: float = 1.5
    box_loss_weight: float = 5.0
    quality_loss_weight: float = 1.0
    box_l1_weight: float = 0.25
    box_iou_weight: float = 1.0
    level_adapters_enabled: bool = False
    level_adapter_ratio: float = 0.25
    class_level_hash_gate_enabled: bool = False
    class_level_gate_temperature: float = 1.0
    object_weighting_enabled: bool = False
    object_weighting_beta: float = 0.999
    object_weighting_max: float = 4.0
    level_aux_loss_weight: float = 0.0
    gate_calibration_loss_weight: float = 0.0
    object_contrastive_loss_weight: float = 0.0
    object_contrastive_temperature: float = 0.1
    object_contrastive_dim: int = 64
    video_motion_enabled: bool = False
    video_motion_hidden_size: int = 64
    video_motion_scale_init: float = 0.1
    dropout: float = 0.0
    layer_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        object.__setattr__(self, "scale_factors", tuple(self.scale_factors))
        object.__setattr__(self, "vision_stage_depths", tuple(self.vision_stage_depths))
        if self.architecture_version not in (6, 8):
            raise ValueError("only detector architecture_version 6 and 8 are supported")
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
        if not self.vision_stage_depths or any(depth <= 0 for depth in self.vision_stage_depths):
            raise ValueError("vision_stage_depths must contain positive depths")
        if sum(self.vision_stage_depths) != self.vision_layers:
            raise ValueError("vision_layers must equal sum(vision_stage_depths)")
        if len(self.vision_stage_depths) != len(self.scale_factors):
            raise ValueError("vision stages must match the configured feature pyramid")
        if self.vision_window_size <= 0:
            raise ValueError("vision_window_size must be positive")
        if self.vision_precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("vision_precision must be fp32, bf16, or fp16")
        if self.neck_mode not in {"baseline", "fpn", "pan"}:
            raise ValueError("neck_mode must be baseline, fpn, or pan")
        if self.neck_repeats <= 0:
            raise ValueError("neck_repeats must be positive")
        if self.neck_mode == "baseline" and self.neck_repeats != 1:
            raise ValueError("baseline neck does not support repeated fusion")
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
        if self.one_to_one_loss_weight < 0.0:
            raise ValueError("one_to_one_loss_weight must be non-negative")
        if not 0.0 <= self.one_to_one_loss_start <= self.one_to_one_loss_weight:
            raise ValueError(
                "one_to_one_loss_start must be between zero and one_to_one_loss_weight"
            )
        if not 0.0 <= self.one_to_one_shared_gradient_scale <= 1.0:
            raise ValueError("one_to_one_shared_gradient_scale must be in [0, 1]")
        if not 0.0 < self.progressive_box_start <= 1.0:
            raise ValueError("progressive_box_start must be in (0, 1]")
        if self.progressive_quality_start < 1.0:
            raise ValueError("progressive_quality_start must be at least 1")
        if not 0.0 < self.level_adapter_ratio <= 1.0:
            raise ValueError("level_adapter_ratio must be in (0, 1]")
        if self.class_level_gate_temperature <= 0.0:
            raise ValueError("class_level_gate_temperature must be positive")
        if not 0.0 <= self.object_weighting_beta < 1.0:
            raise ValueError("object_weighting_beta must be in [0, 1)")
        if self.object_weighting_max < 1.0:
            raise ValueError("object_weighting_max must be at least 1")
        if (
            min(
                self.level_aux_loss_weight,
                self.gate_calibration_loss_weight,
                self.object_contrastive_loss_weight,
            )
            < 0.0
        ):
            raise ValueError("auxiliary loss weights must be non-negative")
        if self.gate_calibration_loss_weight and not self.class_level_hash_gate_enabled:
            raise ValueError("gate calibration requires class_level_hash_gate_enabled")
        if self.object_contrastive_temperature <= 0.0:
            raise ValueError("object_contrastive_temperature must be positive")
        if self.object_contrastive_dim <= 0:
            raise ValueError("object_contrastive_dim must be positive")
        if self.video_motion_hidden_size <= 0:
            raise ValueError("video_motion_hidden_size must be positive")
        if self.video_motion_scale_init < 0.0:
            raise ValueError("video_motion_scale_init must be non-negative")

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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "TRHashDetectorConfig":
        values = dict(values)
        if values.get("architecture_version") not in (6, 8):
            raise ValueError(
                "only TR-Hash detector architecture v6 and v8 checkpoints are supported"
            )
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown detector config fields: {unknown}")
        return cls(**values)
