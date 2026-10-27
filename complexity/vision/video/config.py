"""Configuration for the TR-Hash MoE video tower."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict

from complexity.tr_hash.attention import resolve_domain_heads, resolve_domain_widths


@dataclass(frozen=True)
class TRHashVideoTowerConfig:
    """Architecture contract for a tubelet-tokenized TR-Hash MoE video backbone."""

    image_size: int = 224
    patch_size: int = 16
    num_frames: int = 16
    temporal_patch_size: int = 2
    num_channels: int = 3
    hidden_size: int = 384
    num_hidden_layers: int = 6
    num_attention_heads: int = 4
    num_key_value_heads: int | None = field(default=None, kw_only=True)
    num_experts: int = 4
    top_k: int = 2
    shared_width: int | None = None
    expert_width: int | None = None
    attention_dropout: float = 0.0
    layer_norm_eps: float = 1e-06

    def __post_init__(self) -> None:
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        resolve_domain_heads(self, heads="num_attention_heads", kv="num_key_value_heads")
        resolve_domain_widths(
            self, hidden="hidden_size", shared="shared_width", expert="expert_width"
        )
        if self.num_experts != 4 or self.top_k != 2:
            raise ValueError("TR-HASH requires four experts and top_k=2")
        if self.image_size <= 0 or self.image_size % self.patch_size:
            raise ValueError("image_size must be positive and divisible by patch_size")
        if self.num_frames <= 0 or self.num_frames % self.temporal_patch_size:
            raise ValueError("num_frames must be positive and divisible by temporal_patch_size")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")

    @property
    def num_spatial_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def num_temporal_patches(self) -> int:
        return self.num_frames // self.temporal_patch_size

    @property
    def num_patches(self) -> int:
        return self.num_spatial_patches * self.num_temporal_patches

    @property
    def route_vocab_size(self) -> int:
        return self.num_patches

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "TRHashVideoTowerConfig":
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown video tower config fields: {unknown}")
        values = dict(values)
        return cls(**values)
