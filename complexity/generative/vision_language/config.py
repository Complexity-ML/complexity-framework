"""Configuration for direct image-and-text to text generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, field
from typing import Any, Dict

from complexity.tr_hash.attention import (
    resolve_domain_heads,
    resolve_domain_widths,
    validate_domain_profile,
)

from complexity.config import ModelConfig

from .vision_tower import TRHashVisionTowerConfig


@dataclass(frozen=True)
class TRHashVisionLanguageConfig:
    """A compact VLM with a ViT prefix and a canonical TR-Hash decoder."""

    architecture_profile: str = field(default="source67", kw_only=True)
    architecture_profile_version: int = field(default=1, kw_only=True)
    image_size: int = 224
    patch_size: int = 16
    vision_hidden_size: int = 384
    vision_layers: int = 6
    vision_heads: int = 6
    vision_key_value_heads: int | None = field(default=None, kw_only=True)
    vision_num_experts: int = 4
    vision_top_k: int = 2
    vision_shared_width: int | None = None
    vision_expert_width: int | None = None
    num_visual_tokens: int = 64
    vocab_size: int = 32_000
    hidden_size: int = 768
    decoder_layers: int = 16
    attention_heads: int = 12
    key_value_heads: int | None = None
    max_position_embeddings: int = 2_048
    num_experts: int = 4
    top_k: int = 2
    shared_width: int | None = None
    routed_width: int | None = None
    route_seed: int = 73_856_093
    dropout: float = 0.0

    def __post_init__(self) -> None:
        validate_domain_profile(self.architecture_profile, self.architecture_profile_version)
        resolve_domain_heads(self, heads="vision_heads", kv="vision_key_value_heads")
        resolve_domain_widths(
            self,
            hidden="vision_hidden_size",
            shared="vision_shared_width",
            expert="vision_expert_width",
            legacy_shared=224,
            legacy_expert=64,
        )
        if self.key_value_heads is None:
            object.__setattr__(
                self, "key_value_heads", 1 if self.architecture_profile == "source67" else 4
            )
        if self.shared_width is None:
            object.__setattr__(
                self,
                "shared_width",
                2 * self.hidden_size if self.architecture_profile == "source67" else 1536,
            )
        if self.routed_width is None:
            object.__setattr__(
                self,
                "routed_width",
                self.hidden_size if self.architecture_profile == "source67" else 2048,
            )
        if self.architecture_profile == "source67" and (self.num_experts != 4 or self.top_k != 2):
            raise ValueError("source67 requires four experts and top_k=2")
        if self.image_size <= 0 or self.image_size % self.patch_size:
            raise ValueError("image_size must be positive and divisible by patch_size")
        if self.vision_hidden_size % self.vision_heads:
            raise ValueError("vision_hidden_size must be divisible by vision_heads")
        if self.vision_top_k > self.vision_num_experts:
            raise ValueError("vision_top_k cannot exceed vision_num_experts")
        if self.hidden_size % self.attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        if self.attention_heads % self.key_value_heads:
            raise ValueError("attention_heads must be divisible by key_value_heads")
        if not 0 < self.num_visual_tokens < self.vocab_size:
            raise ValueError("num_visual_tokens must lie in (0, vocab_size)")
        if self.routed_width % self.num_experts:
            raise ValueError("routed_width must be divisible by num_experts")
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError("top_k must lie in [1, num_experts]")

    def vision_tower_config(self) -> TRHashVisionTowerConfig:
        """Build the TR-Hash MoE vision tower configuration."""

        return TRHashVisionTowerConfig(
            architecture_profile=self.architecture_profile,
            architecture_profile_version=self.architecture_profile_version,
            num_key_value_heads=self.vision_key_value_heads,
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
        )

    def decoder_config(self) -> ModelConfig:
        """Build the native causal decoder configuration."""

        return ModelConfig(
            architecture_profile=self.architecture_profile,
            architecture_profile_version=self.architecture_profile_version,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.decoder_layers,
            num_attention_heads=self.attention_heads,
            num_key_value_heads=self.key_value_heads,
            intermediate_size=self.routed_width,
            max_position_embeddings=self.max_position_embeddings,
            attention_type="gqa",
            mlp_type="tr_hash_engine",
            num_experts=self.num_experts,
            top_k=self.top_k,
            top_k_primary_weight=0.5,
            routing_strategy=(
                "token_id_composed_modulo"
                if self.architecture_profile == "source67"
                else "token_id_balanced_hash"
            ),
            shared_expert=True,
            shared_intermediate_size=self.shared_width,
            attention_dropout=self.dropout,
            norm_type="rmsnorm",
            tie_word_embeddings=True,
            use_sdpa=True,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "TRHashVisionLanguageConfig":
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown vision-language config fields: {unknown}")
        values = dict(values)
        values.setdefault("architecture_profile", "legacy")
        return cls(**values)
