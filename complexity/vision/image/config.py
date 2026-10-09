"""Configuration for the latent TR-Hash text-to-image model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class TRHashImageConfig:
    """Architecture contract for a roughly 200M-parameter latent flow model."""

    image_size: int = 256
    vae_downsample_factor: int = 8
    latent_channels: int = 4
    latent_patch_size: int = 2
    vocab_size: int = 32_000
    max_text_length: int = 128
    text_hidden_size: int = 384
    text_layers: int = 4
    text_heads: int = 6
    hidden_size: int = 768
    num_layers: int = 14
    num_attention_heads: int = 12
    num_experts: int = 4
    top_k: int = 2
    shared_width: int = 1_536
    expert_width: int = 96
    time_buckets: int = 256
    route_seed: int = 0x71D5A17
    caption_dropout: float = 0.1
    attention_dropout: float = 0.0
    layer_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
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
        text += self.text_layers * (12 * t * t + 13 * t)
        text += 2 * t + t * h + h
        time = 8 * h * h + 5 * h
        patch = self.latent_patch_features * h + h + self.image_token_count * h
        per_block = 8 * h * h + 8 * h
        per_block += 6 * h * h + 6 * h
        per_block += 3 * h * self.shared_width
        per_block += 3 * self.num_experts * h * self.expert_width
        final = 2 * h * h + 2 * h + h * self.latent_patch_features
        final += self.latent_patch_features
        return text + time + patch + self.num_layers * per_block + final

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "TRHashImageConfig":
        return cls(**values)

