"""TR-Hash vision tower with contextual Source67 routing by patch identity.

Original spatial IDs feed both the two-lag MLP and GQA head/value routing.
The legacy profile retains the previous static balanced-hash MHA backbone.
This spatial extension has no validated vision score.
"""

from __future__ import annotations

from complexity.tr_hash.attention import (
    resolve_domain_heads,
    resolve_domain_widths,
    validate_domain_profile,
)

from dataclasses import asdict, dataclass, fields, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from complexity.tr_hash.attention import DomainSelfAttention, make_domain_mlp
from complexity.tr_hash import (
    TRHashPrecision,
)


@dataclass
class TRHashVisionTowerConfig:
    architecture_profile: str = field(default="source67", kw_only=True)
    architecture_profile_version: int = field(default=1, kw_only=True)
    image_size: int = 224
    patch_size: int = 16
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
    precision: TRHashPrecision = TRHashPrecision.BF16
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
        self.precision = TRHashPrecision(self.precision)
        if self.image_size <= 0 or self.image_size % self.patch_size:
            raise ValueError("image_size must be positive and divisible by patch_size")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        values.setdefault("architecture_profile", "legacy")
        unknown = set(values) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"unknown encoder config fields: {sorted(unknown)}")
        return cls(**values)

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def route_vocab_size(self) -> int:
        return self.num_patches


class _VisionSelfAttention(DomainSelfAttention):
    pass


class TRHashVisionBlock(nn.Module):
    def __init__(self, config: TRHashVisionTowerConfig, layer_index: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = _VisionSelfAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_dropout,
            num_key_value_heads=config.num_key_value_heads,
            architecture_profile=config.architecture_profile,
            layer_index=layer_index,
        )
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = make_domain_mlp(
            hidden_size=config.hidden_size,
            vocab_size=config.route_vocab_size,
            num_experts=config.num_experts,
            top_k=config.top_k,
            shared_width=config.shared_width,
            expert_width=config.expert_width,
            layer_index=layer_index,
            route_seed=config.route_seed,
            precision=config.precision,
            architecture_profile=config.architecture_profile,
        )

    def forward(self, x: torch.Tensor, route_ids: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), route_ids)
        x = x + self.mlp(self.norm2(x), route_ids)
        return x


class TRHashVisionTower(nn.Module):
    """Image -> per-patch contextual embeddings, patches routed through real
    TR-Hash MoE experts by fixed spatial position."""

    def __init__(self, config: TRHashVisionTowerConfig):
        super().__init__()
        self.config = config
        self.patch_embed = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.num_patches, config.hidden_size)
        )
        self.blocks = nn.ModuleList(
            TRHashVisionBlock(config, layer_index=index)
            for index in range(config.num_hidden_layers)
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Each patch's expert routing is a fixed function of its own spatial
        # position — identical role to text token IDs, just addressed by
        # position instead of vocabulary identity.
        self.register_buffer(
            "route_ids", torch.arange(config.num_patches, dtype=torch.long), persistent=True
        )
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        expected = (self.config.num_channels, self.config.image_size, self.config.image_size)
        if tuple(pixel_values.shape[1:]) != expected:
            raise ValueError(f"expected pixel_values shape [batch, {expected}]")
        x = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)
        x = x + self.position_embedding
        route_ids = self.route_ids.unsqueeze(0).expand(x.size(0), -1)
        for block in self.blocks:
            x = block(x, route_ids)
        return self.norm(x)


class TRHashVisionClassifier(nn.Module):
    """Image -> class logits, built on ``TRHashVisionTower``."""

    def __init__(self, config: TRHashVisionTowerConfig, num_classes: int):
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.tower = TRHashVisionTower(config)
        self.head_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.head = nn.Linear(config.hidden_size, num_classes)

    def forward(self, pixel_values: torch.Tensor, labels: "torch.Tensor | None" = None) -> dict:
        features = self.tower(pixel_values)
        pooled = self.head_norm(features.mean(dim=1))
        logits = self.head(pooled)
        output = {"logits": logits, "pooled_features": pooled}
        if labels is not None:
            output["loss"] = F.cross_entropy(logits, labels)
        return output
