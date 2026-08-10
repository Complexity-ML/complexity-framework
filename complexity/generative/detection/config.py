"""Configuration for the TR-Hash single-stage object detector."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict

from ..vision_language.vision_tower import TRHashVisionTowerConfig


@dataclass(frozen=True)
class TRHashDetectorConfig:
    """Architecture contract for a YOLO-style anchor-free detector.

    Single-scale: predictions come directly off the backbone's patch grid
    (one grid cell per patch, ``image_size // patch_size`` cells per side) —
    no multi-scale feature pyramid. A real detector benefits from one, but
    this is out of scope for the first pass; the grid resolution can still
    be raised via a smaller ``patch_size``.
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
    route_seed: int = 0x71D5A17
    num_classes: int = 80
    box_loss_weight: float = 5.0
    objectness_loss_weight: float = 1.0
    class_loss_weight: float = 1.0
    dropout: float = 0.0
    layer_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if self.vision_hidden_size % self.vision_heads:
            raise ValueError("vision_hidden_size must be divisible by vision_heads")
        if self.vision_top_k > self.vision_num_experts:
            raise ValueError("vision_top_k cannot exceed vision_num_experts")

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_cells(self) -> int:
        return self.grid_size ** 2

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
            route_seed=self.route_seed,
            attention_dropout=self.dropout,
            layer_norm_eps=self.layer_norm_eps,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "TRHashDetectorConfig":
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown detector config fields: {unknown}")
        return cls(**values)
