"""TR-Hash MoE video tower — tubelets route through deterministic expert IDs.

Replaces ``complexity.multimodal.video``'s ``VideoTokenRoutedMLP`` (spatial
position mod num_experts, precomputed once and applied identically at every
layer) with a real ``TRHashEngine`` per block, keyed on each tubelet's fixed
position in the flattened temporal x spatial grid — same principle as
``TRHashVisionTower`` for images, generalized from patches to tubelets, with
a per-layer route permutation (via ``TRHashEngineConfig.layer_index``)
instead of one fixed routing reused at every depth.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from complexity.multimodal.video import TubeletEmbedding, VideoConfig
from complexity.tr_hash import (
    TRHashBackend,
    TRHashEngine,
    TRHashEngineConfig,
    TRHashPrecision,
    TRHashStrategy,
)

from .config import TRHashVideoTowerConfig


class _VideoSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = float(dropout)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (value.transpose(1, 2) for value in (q, k, v))
        output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(output.transpose(1, 2).reshape(batch, length, width))


class TRHashVideoBlock(nn.Module):
    def __init__(self, config: TRHashVideoTowerConfig, layer_index: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = _VideoSelfAttention(
            config.hidden_size, config.num_attention_heads, config.attention_dropout
        )
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = TRHashEngine(
            TRHashEngineConfig(
                hidden_size=config.hidden_size,
                vocab_size=config.route_vocab_size,
                num_experts=config.num_experts,
                top_k=config.top_k,
                shared_width=config.shared_width,
                expert_width=config.expert_width,
                routing_strategy=TRHashStrategy.BALANCED_HASH,
                layer_index=layer_index,
                route_seed=config.route_seed,
                precision=TRHashPrecision.BF16,
                backend=TRHashBackend.AUTO,
            )
        )

    def forward(self, x: torch.Tensor, route_ids: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x), route_ids)
        return x


class TRHashVideoTower(nn.Module):
    """Video -> per-tubelet contextual embeddings, tubelets routed through
    real TR-Hash MoE experts by fixed spatio-temporal position."""

    def __init__(self, config: TRHashVideoTowerConfig):
        super().__init__()
        self.config = config
        self.tubelet_embed = TubeletEmbedding(
            VideoConfig(
                image_size=config.image_size,
                patch_size=config.patch_size,
                num_frames=config.num_frames,
                temporal_patch_size=config.temporal_patch_size,
                num_channels=config.num_channels,
                hidden_size=config.hidden_size,
            )
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.num_patches, config.hidden_size)
        )
        self.blocks = nn.ModuleList(
            TRHashVideoBlock(config, layer_index=index)
            for index in range(config.num_hidden_layers)
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.register_buffer(
            "route_ids", torch.arange(config.num_patches, dtype=torch.long), persistent=True
        )
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        expected = (
            self.config.num_channels,
            self.config.num_frames,
            self.config.image_size,
            self.config.image_size,
        )
        if tuple(video.shape[1:]) != expected:
            raise ValueError(f"expected video shape [batch, {expected}]")
        x = self.tubelet_embed(video)
        x = x + self.position_embedding
        route_ids = self.route_ids.unsqueeze(0).expand(x.size(0), -1)
        for block in self.blocks:
            x = block(x, route_ids)
        return self.norm(x)


class TRHashVideoClassifier(nn.Module):
    """Video -> class logits, built on ``TRHashVideoTower``."""

    def __init__(self, config: TRHashVideoTowerConfig, num_classes: int):
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.tower = TRHashVideoTower(config)
        self.head_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.head = nn.Linear(config.hidden_size, num_classes)

    def forward(
        self, video: torch.Tensor, labels: "torch.Tensor | None" = None
    ) -> dict:
        features = self.tower(video)
        pooled = self.head_norm(features.mean(dim=1))
        logits = self.head(pooled)
        output = {"logits": logits, "pooled_features": pooled}
        if labels is not None:
            output["loss"] = F.cross_entropy(logits, labels)
        return output
