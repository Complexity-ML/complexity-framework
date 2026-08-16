"""Modality-specific tokenizers for TR-Hash sensor fusion."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _TemporalBlock(nn.Module):
    def __init__(self, width: int, dilation: int):
        super().__init__()
        self.depthwise = nn.Conv1d(
            width,
            width,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
            groups=width,
            bias=False,
        )
        self.norm = nn.GroupNorm(1, width)
        self.pointwise = nn.Sequential(
            nn.Conv1d(width, 2 * width, 1),
            nn.GELU(),
            nn.Conv1d(2 * width, width, 1),
        )
        self.scale = nn.Parameter(torch.full((width, 1), 1e-3))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.depthwise(values)
        values = self.pointwise(self.norm(values))
        return residual + values * self.scale


class SequenceTokenizerV2(nn.Module):
    """Dilated temporal encoder retaining both local and long-range dynamics."""

    def __init__(self, input_features: int, hidden_size: int, token_count: int):
        super().__init__()
        self.input_features = int(input_features)
        self.token_count = int(token_count)
        self.input = nn.Sequential(
            nn.Conv1d(2 * input_features, hidden_size, 3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_size),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *(_TemporalBlock(hidden_size, dilation) for dilation in (1, 2, 4, 8))
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.size(-1) != self.input_features:
            raise ValueError(
                f"sensor sequences must have shape [batch, time, {self.input_features}]"
            )
        if values.size(1) == 0:
            raise ValueError("sensor sequences cannot be empty")
        centered = values - values.mean(dim=1, keepdim=True)
        standardized = centered / values.std(
            dim=1,
            keepdim=True,
            unbiased=False,
        ).clamp_min(1e-5)
        values = torch.cat((values, standardized), dim=-1)
        features = self.blocks(self.input(values.transpose(1, 2)))
        return F.adaptive_avg_pool1d(features, self.token_count).transpose(1, 2)
