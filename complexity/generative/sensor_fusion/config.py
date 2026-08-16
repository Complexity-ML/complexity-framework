"""Configuration for the TR-Hash multimodal robot perception model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Tuple

from complexity.tr_hash import TRHashPrecision

SENSOR_MODALITIES: Tuple[str, ...] = (
    "depth",
    "ir",
    "thermal",
    "imu",
    "radar",
    "skeleton",
)


@dataclass(frozen=True)
class TRHashSensorFusionConfig:
    """Architecture contract for the TR-Hash multimodal perception model.

    Six sensor streams (depth, infrared, and thermal clips; IMU, radar, and
    skeleton sequences) are tokenized, fused through TR-Hash blocks routed
    by ``(modality, local token position)``, and pooled into a
    classification head. Visual streams share one TR-Hash MoE vision tower.
    Every capability below is always active -- there is no reduced/legacy
    variant.
    """

    num_classes: int = 40
    hidden_size: int = 256
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    num_experts: int = 8
    top_k: int = 2
    shared_width: int = 256
    expert_width: int = 128
    route_seed: int = 0x71D5A17
    precision: str = "bf16"
    attention_dropout: float = 0.0
    classifier_dropout: float = 0.1
    layer_norm_eps: float = 1e-6

    # The visual encoders reduce arbitrary clips to a fixed token lattice.
    visual_channels: Tuple[int, int, int] = (3, 1, 3)
    visual_token_grid: Tuple[int, int, int] = (4, 4, 4)

    # The three visual streams (depth, IR, thermal) share one TR-Hash MoE
    # vision tower. These fields are part of the checkpoint contract so
    # transfer compatibility can be checked explicitly.
    vision_image_size: int = 224
    vision_patch_size: int = 8
    vision_hidden_size: int = 128
    vision_layers: int = 4
    vision_heads: int = 4
    vision_num_experts: int = 4
    vision_top_k: int = 2
    vision_shared_width: int = 112
    vision_expert_width: int = 32
    vision_stage_depths: Tuple[int, ...] = (1, 1, 2)
    vision_window_size: int = 8

    # Confidence-gated late-fusion path: a fixed TR-Hash route scores every
    # (modality, class) pair and blends per-modality logits with the shared
    # fused head.
    late_fusion_weight: float = 0.5

    # Feature dimensions are explicit so all parameters exist before the
    # optimizer is built.
    imu_features: int = 45
    radar_features: int = 16
    skeleton_joints: int = 17
    skeleton_coordinates: int = 3
    sequence_tokens: int = 32

    # Residual classifier capacity. Every class identity follows a fixed
    # balanced top-k route while the ordinary dense head remains the primary
    # prediction path.
    class_hash_shared_width: int = 96
    class_hash_expert_width: int = 24
    class_hash_initial_scale: float = 0.05

    def __post_init__(self) -> None:
        object.__setattr__(self, "visual_channels", tuple(self.visual_channels))
        object.__setattr__(self, "visual_token_grid", tuple(self.visual_token_grid))
        object.__setattr__(self, "vision_stage_depths", tuple(self.vision_stage_depths))
        TRHashPrecision(self.precision)
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if self.hidden_size <= 0 or self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be positive and divisible by num_attention_heads")
        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if self.num_experts != 8:
            raise ValueError("this model contract requires exactly 8 experts")
        if self.top_k <= 0 or self.top_k > self.num_experts:
            raise ValueError("top_k must be positive and cannot exceed num_experts")
        if self.shared_width < 0 or self.expert_width <= 0:
            raise ValueError("invalid shared or expert width")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")
        if not 0.0 <= self.classifier_dropout < 1.0:
            raise ValueError("classifier_dropout must be in [0, 1)")
        if len(self.visual_channels) != 3 or any(value <= 0 for value in self.visual_channels):
            raise ValueError("visual_channels must describe depth, IR, and thermal")
        if len(self.visual_token_grid) != 3 or any(value <= 0 for value in self.visual_token_grid):
            raise ValueError("visual_token_grid must contain three positive dimensions")
        if self.vision_image_size <= 0 or self.vision_patch_size <= 0:
            raise ValueError("vision image and patch sizes must be positive")
        if self.vision_image_size % self.vision_patch_size:
            raise ValueError("vision_image_size must be divisible by vision_patch_size")
        if self.vision_hidden_size <= 0 or self.vision_hidden_size % self.vision_heads:
            raise ValueError("vision hidden size must be positive and divisible by heads")
        if sum(self.vision_stage_depths) != self.vision_layers:
            raise ValueError("vision_layers must equal sum(vision_stage_depths)")
        if self.vision_num_experts <= 0 or not 0 < self.vision_top_k <= self.vision_num_experts:
            raise ValueError("invalid vision expert routing configuration")
        if self.vision_shared_width < 0 or self.vision_expert_width <= 0:
            raise ValueError("invalid vision shared or expert width")
        if self.vision_window_size <= 0:
            raise ValueError("vision_window_size must be positive")
        if self.late_fusion_weight < 0.0:
            raise ValueError("late_fusion_weight must be non-negative")
        if (
            min(
                self.imu_features,
                self.radar_features,
                self.skeleton_joints,
                self.skeleton_coordinates,
                self.sequence_tokens,
            )
            <= 0
        ):
            raise ValueError("sensor dimensions and sequence_tokens must be positive")
        if self.class_hash_shared_width < 0 or self.class_hash_expert_width <= 0:
            raise ValueError("invalid class-hash shared or expert width")
        if self.class_hash_initial_scale < 0.0:
            raise ValueError("class_hash_initial_scale must be non-negative")

    @property
    def visual_tokens(self) -> int:
        time, height, width = self.visual_token_grid
        return time * height * width

    @property
    def modality_token_counts(self) -> Tuple[int, ...]:
        return (
            self.visual_tokens,
            self.visual_tokens,
            self.visual_tokens,
            self.sequence_tokens,
            self.sequence_tokens,
            self.sequence_tokens,
        )

    @property
    def route_vocab_size(self) -> int:
        # +1 for the always-present fusion CLS token.
        return sum(self.modality_token_counts) + 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "TRHashSensorFusionConfig":
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown sensor-fusion config fields: {unknown}")
        return cls(**values)
