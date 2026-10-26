"""TR-Hash video tower with contextual TR-HASH routing by tubelet identity.

Raw flattened temporal/spatial positions feed both GQA and the two-lag MLP.
No video accuracy result is implied by the text recipe."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from complexity.multimodal.video import TubeletEmbedding, VideoConfig
from complexity.tr_hash.attention import DomainSelfAttention, make_domain_mlp
from complexity.tr_hash import TRHashPrecision
from .config import TRHashVideoTowerConfig


class _VideoSelfAttention(DomainSelfAttention):
    pass


class TRHashVideoBlock(nn.Module):

    def __init__(self, config: TRHashVideoTowerConfig, layer_index: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = _VideoSelfAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_dropout,
            num_key_value_heads=config.num_key_value_heads,
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
            precision=TRHashPrecision.BF16,
        )

    def forward(self, x: torch.Tensor, route_ids: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), route_ids)
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
            (
                TRHashVideoBlock(config, layer_index=index)
                for index in range(config.num_hidden_layers)
            )
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

    def forward(self, video: torch.Tensor, labels: "torch.Tensor | None" = None) -> dict:
        features = self.tower(video)
        pooled = self.head_norm(features.mean(dim=1))
        logits = self.head(pooled)
        output = {"logits": logits, "pooled_features": pooled}
        if labels is not None:
            output["loss"] = F.cross_entropy(logits, labels)
        return output
