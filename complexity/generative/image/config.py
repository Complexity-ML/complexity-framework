"""Configuration for the latent TR-Hash text-to-image model."""

from __future__ import annotations

from complexity.tr_hash.attention import (
    resolve_domain_heads,
    resolve_domain_widths,
    validate_domain_profile,
)

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class TRHashImageConfig:
    """Architecture contract for a roughly 200M-parameter latent flow model."""

    architecture_profile: str = field(default="source67", kw_only=True)
    architecture_profile_version: int = field(default=1, kw_only=True)
    image_size: int = 256
    vae_downsample_factor: int = 8
    latent_channels: int = 4
    latent_patch_size: int = 2
    vocab_size: int = 32_000
    max_text_length: int = 128
    text_hidden_size: int = 384
    text_layers: int = 4
    text_heads: int = 6
    text_key_value_heads: int | None = field(default=None, kw_only=True)
    hidden_size: int = 768
    num_layers: int = 14
    num_attention_heads: int = 12
    num_key_value_heads: int | None = field(default=None, kw_only=True)
    num_experts: int = 4
    top_k: int = 2
    shared_width: int | None = None
    expert_width: int | None = None
    time_buckets: int = 256
    route_seed: int = 0x71D5A17
    caption_dropout: float = 0.1
    attention_dropout: float = 0.0
    layer_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        validate_domain_profile(self.architecture_profile, self.architecture_profile_version)
        resolve_domain_heads(self, heads="num_attention_heads", kv="num_key_value_heads")
        resolve_domain_heads(self, heads="text_heads", kv="text_key_value_heads")
        resolve_domain_widths(
            self,
            hidden="hidden_size",
            shared="shared_width",
            expert="expert_width",
            legacy_shared=1536,
            legacy_expert=96,
        )
        if self.architecture_profile == "source67":
            if self.num_experts != 4 or self.top_k != 2:
                raise ValueError("source67 requires four experts and top_k=2")
        positive = (
            "image_size",
            "vae_downsample_factor",
            "latent_channels",
            "latent_patch_size",
            "vocab_size",
            "max_text_length",
            "text_hidden_size",
            "text_layers",
            "text_heads",
            "hidden_size",
            "num_layers",
            "num_attention_heads",
            "num_experts",
            "top_k",
            "shared_width",
            "expert_width",
            "time_buckets",
        )
        for name in positive:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.image_size % self.vae_downsample_factor:
            raise ValueError("image_size must be divisible by vae_downsample_factor")
        if self.latent_resolution % self.latent_patch_size:
            raise ValueError("latent resolution must be divisible by latent_patch_size")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.text_hidden_size % self.text_heads:
            raise ValueError("text_hidden_size must be divisible by text_heads")
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        if not 0.0 <= self.caption_dropout < 1.0:
            raise ValueError("caption_dropout must be in [0, 1)")

    @property
    def latent_resolution(self) -> int:
        return self.image_size // self.vae_downsample_factor

    @property
    def latent_grid_size(self) -> int:
        return self.latent_resolution // self.latent_patch_size

    @property
    def image_token_count(self) -> int:
        return self.latent_grid_size**2

    @property
    def route_vocab_size(self) -> int:
        return self.image_token_count * self.time_buckets

    @property
    def latent_patch_features(self) -> int:
        return self.latent_channels * self.latent_patch_size**2

    def estimated_parameter_count(self) -> int:
        """Exact architecture estimate without allocating the full model."""

        h = self.hidden_size
        t = self.text_hidden_size
        text = self.vocab_size * t + self.max_text_length * t
        if self.architecture_profile == "source67":
            text_head_dim = t // self.text_heads
            text += self.text_layers * (
                11 * t * t
                + 2 * t * self.text_key_value_heads * text_head_dim
                + 4 * t
                + 2 * text_head_dim
            )
        else:
            text += self.text_layers * (12 * t * t + 13 * t)
        text += 2 * t + t * h + h
        time = 8 * h * h + 5 * h
        patch = self.latent_patch_features * h + h + self.image_token_count * h
        if self.architecture_profile == "source67":
            head_dim = h // self.num_attention_heads
            per_block = (
                12 * h * h + 2 * h * self.num_key_value_heads * head_dim + 10 * h + 2 * head_dim
            )
        else:
            per_block = 14 * h * h + 14 * h
        per_block += 3 * h * self.shared_width
        per_block += 3 * self.num_experts * h * self.expert_width
        final = 2 * h * h + 2 * h + h * self.latent_patch_features
        final += self.latent_patch_features
        return text + time + patch + self.num_layers * per_block + final

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "TRHashImageConfig":
        values = dict(values)
        values.setdefault("architecture_profile", "legacy")
        return cls(**values)
