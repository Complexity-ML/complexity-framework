"""Lightweight cross-scale fusion necks for the TR-Hash detector."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _identity_channel_projection(channels: int) -> nn.Conv2d:
    projection = nn.Conv2d(channels, channels, 1)
    nn.init.dirac_(projection.weight)
    nn.init.zeros_(projection.bias)
    return projection


class _ResidualSpatialRefinement(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.layers(values)


class CrossScaleFusionNeck(nn.Module):
    """Identity-initialized additive FPN with an optional PAN bottom-up path.

    Every level keeps a cheap channel-wise lateral projection. Learned scalar
    gates introduce adjacent-scale context without changing transferred
    baseline predictions at initialization.
    """

    def __init__(self, channels: int, levels: int, mode: str):
        super().__init__()
        if levels <= 0:
            raise ValueError("neck requires at least one feature level")
        if mode not in {"fpn", "pan"}:
            raise ValueError("cross-scale neck mode must be fpn or pan")
        self.mode = mode
        self.laterals = nn.ModuleList(
            _identity_channel_projection(channels) for _ in range(levels)
        )
        self.top_down_gates = nn.Parameter(torch.zeros(max(levels - 1, 0)))
        self.top_down_refinements = nn.ModuleList(
            _ResidualSpatialRefinement(channels) for _ in range(levels - 1)
        )
        if mode == "pan":
            self.bottom_up_downsamples = nn.ModuleList(
                nn.Conv2d(
                    channels,
                    channels,
                    3,
                    stride=2,
                    padding=1,
                    groups=channels,
                )
                for _ in range(levels - 1)
            )
            self.bottom_up_gates = nn.Parameter(torch.zeros(max(levels - 1, 0)))
            self.bottom_up_refinements = nn.ModuleList(
                _ResidualSpatialRefinement(channels) for _ in range(levels - 1)
            )
        else:
            self.bottom_up_downsamples = nn.ModuleList()
            self.register_parameter("bottom_up_gates", None)
            self.bottom_up_refinements = nn.ModuleList()

    def forward(self, feature_maps: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(feature_maps) != len(self.laterals):
            raise ValueError("feature pyramid level count does not match neck")
        top_down = [lateral(values) for lateral, values in zip(self.laterals, feature_maps)]
        for level in range(len(top_down) - 2, -1, -1):
            context = F.interpolate(top_down[level + 1], size=top_down[level].shape[-2:])
            fused = top_down[level] + self.top_down_gates[level] * context
            top_down[level] = self.top_down_refinements[level](fused)
        if self.mode == "fpn":
            return top_down

        outputs = [top_down[0]]
        for level, (downsample, refinement) in enumerate(
            zip(self.bottom_up_downsamples, self.bottom_up_refinements)
        ):
            context = downsample(outputs[-1])
            target = top_down[level + 1]
            if context.shape[-2:] != target.shape[-2:]:
                context = F.interpolate(context, size=target.shape[-2:])
            fused = target + self.bottom_up_gates[level] * context
            outputs.append(refinement(fused))
        return outputs
