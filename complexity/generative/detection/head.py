"""Small decoupled prediction heads for TR-Hash detection."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import TRHashDetectorConfig


def _branch(
    input_size: int,
    hidden_size: int,
    output_size: int,
    eps: float,
) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_size, eps=eps),
        nn.Linear(input_size, hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, output_size),
    )


class DecoupledDetectionHead(nn.Module):
    """Independent classification and localization branches at every scale."""

    def __init__(self, config: TRHashDetectorConfig):
        super().__init__()
        self.config = config
        hidden = config.vision_hidden_size
        branch_hidden = config.resolved_head_hidden_size
        self.regression_heads = nn.ModuleList(
            _branch(hidden, branch_hidden, config.regression_width, config.layer_norm_eps)
            for _ in config.grid_sizes
        )
        self.classification_heads = nn.ModuleList(
            _branch(hidden, branch_hidden, config.num_classes, config.layer_norm_eps)
            for _ in config.grid_sizes
        )
        self._initialize_predictions()

    def _initialize_predictions(self) -> None:
        class_prior = math.log(0.01 / 0.99)
        for head in self.classification_heads:
            nn.init.normal_(head[-1].weight, std=0.01)
            nn.init.constant_(head[-1].bias, class_prior)
        for head in self.regression_heads:
            nn.init.normal_(head[-1].weight, std=0.01)
            nn.init.zeros_(head[-1].bias)
            if self.config.reg_max:
                bins = torch.arange(
                    self.config.dfl_bins,
                    dtype=head[-1].bias.dtype,
                    device=head[-1].bias.device,
                )
                prior = -(bins - 2.0).abs()
                with torch.no_grad():
                    head[-1].bias.copy_(prior.repeat(4))

    def forward(
        self,
        feature_maps: list[torch.Tensor],
        *,
        return_hidden: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        outputs = []
        hidden_outputs = []
        for regression, classification, feature_map in zip(
            self.regression_heads,
            self.classification_heads,
            feature_maps,
        ):
            tokens = feature_map.flatten(2).transpose(1, 2)
            regression_hidden = regression[2](regression[1](regression[0](tokens)))
            classification_hidden = classification[2](classification[1](classification[0](tokens)))
            outputs.append(
                torch.cat(
                    (regression[3](regression_hidden), classification[3](classification_hidden)),
                    dim=-1,
                )
            )
            if return_hidden:
                hidden_outputs.append(tokens)
        return torch.cat(outputs, dim=1), hidden_outputs if return_hidden else None
