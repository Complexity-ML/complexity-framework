"""Configuration for the TR-Hash MoE video tower."""

from __future__ import annotations

from complexity.tr_hash.attention import (
    resolve_domain_heads,
    resolve_domain_widths,
    validate_domain_profile,
)

from dataclasses import asdict, dataclass, fields, field
from typing import Any, Dict


@dataclass(frozen=True)
class TRHashVideoTowerConfig:
    """Architecture contract for a tubelet-tokenized TR-Hash MoE video backbone."""

    architecture_profile: str = field(default="source67", kw_only=True)
    architecture_profile_version: int = field(default=1, kw_only=True)
    image_size: int = 224
    patch_size: int = 16
    num_frames: int = 16
    temporal_patch_size: int = 2
    num_channels: int = 3
    hidden_size: int = 384
    num_hidden_layers: int = 6
    num_attention_heads: int = 6
    num_key_value_heads: int | None = field(default=None, kw_only=True)
    num_experts: int = 4
    top_k: int = 2
    shared_width: int | None = None
    expert_width: int | None = None
    route_seed: int = 0x71D5A17
    attention_dropout: float = 0.0
    layer_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        validate_domain_profile(self.architecture_profile, self.architecture_profile_version)
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        resolve_domain_heads(self, heads="num_attention_heads", kv="num_key_value_heads")
        resolve_domain_widths(
            self,
            hidden="hidden_size",
            shared="shared_width",
            expert="expert_width",
            legacy_shared=128,
            legacy_expert=48,
        )
        if self.architecture_profile == "source67":
            if self.num_experts != 4 or self.top_k != 2:
                raise ValueError("source67 requires four experts and top_k=2")
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
        values.setdefault("architecture_profile", "legacy")
        return cls(**values)
