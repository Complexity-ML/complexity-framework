"""Optional level specialization and temporal motion for TR-Hash detection."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from complexity.tr_hash import (
    TRHashBackend,
    TRHashEngine,
    TRHashEngineConfig,
    TRHashPrecision,
    TRHashStrategy,
)

from .config import TRHashDetectorConfig


class ResidualLevelAdapter(nn.Module):
    """Identity-initialized spatial adapter dedicated to one pyramid level."""

    def __init__(self, channels: int, ratio: float):
        super().__init__()
        inner = max(16, int(round(channels * ratio)))
        self.layers = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.GroupNorm(1, inner),
            nn.GELU(),
            nn.Conv2d(inner, inner, 3, padding=1, groups=inner, bias=False),
            nn.GroupNorm(1, inner),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
        )
        nn.init.zeros_(self.layers[-1].weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.layers(values)


class MultiScaleLevelAdapters(nn.Module):
    """Apply an independent residual adapter to every prediction level."""

    def __init__(self, channels: int, levels: int, ratio: float):
        super().__init__()
        self.adapters = nn.ModuleList(
            ResidualLevelAdapter(channels, ratio) for _ in range(levels)
        )

    def forward(self, feature_maps: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(feature_maps) != len(self.adapters):
            raise ValueError("feature pyramid level count does not match adapters")
        return [adapter(values) for adapter, values in zip(self.adapters, feature_maps)]


class ClassLevelHashGate(nn.Module):
    """Route each class-level identity through TR-Hash before level scoring."""

    def __init__(self, config: TRHashDetectorConfig):
        super().__init__()
        self.num_levels = len(config.grid_sizes)
        self.num_classes = config.num_classes
        hidden_size = config.vision_hidden_size
        self.class_embedding = nn.Parameter(torch.empty(config.num_classes, hidden_size))
        self.level_embedding = nn.Parameter(torch.empty(self.num_levels, hidden_size))
        self.norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.mlp = TRHashEngine(
            TRHashEngineConfig(
                hidden_size=hidden_size,
                vocab_size=self.num_levels * config.num_classes,
                num_experts=config.vision_num_experts,
                top_k=config.vision_top_k,
                shared_width=config.vision_shared_width,
                expert_width=config.vision_expert_width,
                routing_strategy=TRHashStrategy.BALANCED_HASH,
                layer_index=config.vision_layers + 1,
                route_seed=config.route_seed,
                precision=TRHashPrecision(config.vision_precision),
                backend=TRHashBackend.AUTO,
            )
        )
        self.score = nn.Linear(hidden_size, 1)
        self.register_buffer(
            "route_ids",
            torch.arange(self.num_levels * config.num_classes, dtype=torch.long),
            persistent=True,
        )
        nn.init.normal_(self.class_embedding, std=0.02)
        nn.init.normal_(self.level_embedding, std=0.02)
        # A zero score gives a uniform distribution and therefore a zero
        # log-prior in the prediction head at initialization.
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, feature_maps: list[torch.Tensor]) -> torch.Tensor:
        if len(feature_maps) != self.num_levels:
            raise ValueError("feature pyramid level count does not match hash gate")
        pooled = torch.stack(
            [feature_map.mean(dim=(-2, -1)) for feature_map in feature_maps],
            dim=1,
        )
        batch = pooled.size(0)
        tokens = (
            pooled[:, :, None, :]
            + self.level_embedding[None, :, None, :]
            + self.class_embedding[None, None, :, :]
        ).reshape(batch, self.num_levels * self.num_classes, -1)
        route_ids = self.route_ids[None].expand(batch, -1)
        tokens = tokens + self.mlp(self.norm(tokens), route_ids)
        return self.score(tokens).reshape(batch, self.num_levels, self.num_classes)


class TemporalMotionPyramid(nn.Module):
    """Project frame-difference statistics into residuals for every FPN level.

    Clips use ``[batch, time, channels, height, width]``. Static images execute
    the same branch with an all-zero motion summary: predictions remain neutral
    while every parameter stays in the DDP graph.
    """

    def __init__(
        self,
        channels: int,
        levels: int,
        hidden_size: int,
        scale_init: float,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(6, hidden_size, 5, stride=4, padding=2, bias=False),
            nn.GroupNorm(1, hidden_size),
            nn.GELU(),
            nn.Conv2d(
                hidden_size,
                hidden_size,
                3,
                padding=1,
                groups=hidden_size,
                bias=False,
            ),
            nn.GroupNorm(1, hidden_size),
            nn.GELU(),
        )
        self.projections = nn.ModuleList(
            nn.Conv2d(hidden_size, channels, 1, bias=False) for _ in range(levels)
        )
        self.scales = nn.Parameter(torch.full((levels,), float(scale_init)))

    @staticmethod
    def center_frame(pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim == 4:
            return pixel_values
        if pixel_values.ndim != 5:
            raise ValueError(
                "detector input must be [B,C,H,W] or video [B,T,C,H,W]"
            )
        if pixel_values.size(1) < 2:
            raise ValueError("video detection requires at least two frames")
        if pixel_values.size(2) != 3:
            raise ValueError("video detector clips require three image channels")
        return pixel_values[:, pixel_values.size(1) // 2]

    @staticmethod
    def _motion_summary(pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim == 4:
            return torch.cat((pixel_values * 0.0, pixel_values * 0.0), dim=1)
        differences = pixel_values[:, 1:] - pixel_values[:, :-1]
        signed = differences.mean(dim=1)
        magnitude = differences.abs().mean(dim=1)
        return torch.cat((signed, magnitude), dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        target_shapes: list[tuple[int, int]],
    ) -> list[torch.Tensor]:
        if len(target_shapes) != len(self.projections):
            raise ValueError("motion pyramid target count does not match levels")
        motion = self.stem(self._motion_summary(pixel_values))
        return [
            scale * projection(F.adaptive_avg_pool2d(motion, shape))
            for scale, projection, shape in zip(
                self.scales,
                self.projections,
                target_shapes,
            )
        ]
