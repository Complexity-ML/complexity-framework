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


class _SpatialContext(nn.Module):
    """Cheap residual spatial mixing before the token-wise prediction MLP."""

    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.activation = nn.SiLU()
        self.scale = nn.Parameter(torch.zeros(channels))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        context = self.activation(self.depthwise(values))
        return values + self.scale.view(1, -1, 1, 1) * context


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
        self.regression_spatial = nn.ModuleList(
            _SpatialContext(hidden) if config.head_spatial_mixing else nn.Identity()
            for _ in config.grid_sizes
        )
        self.classification_spatial = nn.ModuleList(
            _SpatialContext(hidden) if config.head_spatial_mixing else nn.Identity()
            for _ in config.grid_sizes
        )
        self.regression_log_scales = (
            nn.Parameter(torch.zeros(len(config.grid_sizes)))
            if config.regression_logit_scale
            else None
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
        class_level_bias: torch.Tensor | None = None,
        return_hidden: bool = False,
        return_branch_hidden: bool = False,
    ) -> tuple[torch.Tensor, list | None]:
        if return_hidden and return_branch_hidden:
            raise ValueError("request either feature tokens or branch hidden states, not both")
        outputs = []
        hidden_outputs = []
        if class_level_bias is not None and class_level_bias.shape[1:] != (
            len(feature_maps),
            self.config.num_classes,
        ):
            raise ValueError("class_level_bias must have shape [B, levels, classes]")
        for level, (
            regression,
            classification,
            regression_spatial,
            classification_spatial,
            feature_map,
        ) in enumerate(
            zip(
                self.regression_heads,
                self.classification_heads,
                self.regression_spatial,
                self.classification_spatial,
                feature_maps,
            )
        ):
            tokens = feature_map.flatten(2).transpose(1, 2)
            regression_tokens = regression_spatial(feature_map).flatten(2).transpose(1, 2)
            classification_tokens = classification_spatial(feature_map).flatten(2).transpose(1, 2)
            regression_hidden = regression[2](regression[1](regression[0](regression_tokens)))
            classification_hidden = classification[2](
                classification[1](classification[0](classification_tokens))
            )
            class_logits = classification[3](classification_hidden)
            if class_level_bias is not None:
                class_logits = class_logits + class_level_bias[:, level, None, :]
            regression_logits = regression[3](regression_hidden)
            if self.regression_log_scales is not None:
                regression_logits = regression_logits * self.regression_log_scales[level].exp()
            outputs.append(
                torch.cat(
                    (regression_logits, class_logits),
                    dim=-1,
                )
            )
            if return_hidden:
                hidden_outputs.append(tokens)
            elif return_branch_hidden:
                hidden_outputs.append((regression_hidden, classification_hidden))
        return (
            torch.cat(outputs, dim=1),
            hidden_outputs if return_hidden or return_branch_hidden else None,
        )


class OneToOnePredictionHead(nn.Module):
    """Cheap output-only branch sharing the trained decoupled head features."""

    def __init__(self, config: TRHashDetectorConfig):
        super().__init__()
        hidden = config.resolved_head_hidden_size
        self.regression_outputs = nn.ModuleList(
            nn.Linear(hidden, config.regression_width) for _ in config.grid_sizes
        )
        self.classification_outputs = nn.ModuleList(
            nn.Linear(hidden, config.num_classes) for _ in config.grid_sizes
        )
        self.regression_log_scales = (
            nn.Parameter(torch.zeros(len(config.grid_sizes)))
            if config.regression_logit_scale
            else None
        )
        class_prior = math.log(0.01 / 0.99)
        for regression, classification in zip(
            self.regression_outputs,
            self.classification_outputs,
        ):
            nn.init.normal_(regression.weight, std=0.01)
            nn.init.zeros_(regression.bias)
            if config.reg_max:
                bins = torch.arange(
                    config.dfl_bins,
                    dtype=regression.bias.dtype,
                    device=regression.bias.device,
                )
                with torch.no_grad():
                    regression.bias.copy_((-(bins - 2.0).abs()).repeat(4))
            nn.init.normal_(classification.weight, std=0.01)
            nn.init.constant_(classification.bias, class_prior)

    @torch.no_grad()
    def initialize_from(self, one_to_many: DecoupledDetectionHead) -> None:
        """Start from the compatible one-to-many output projections."""

        for target, source in zip(
            self.regression_outputs,
            one_to_many.regression_heads,
        ):
            target.load_state_dict(source[-1].state_dict())
        if self.regression_log_scales is not None:
            if one_to_many.regression_log_scales is None:
                raise ValueError(
                    "one_to_many head has no regression_log_scales to transfer; "
                    "both heads must share regression_logit_scale"
                )
            self.regression_log_scales.copy_(one_to_many.regression_log_scales)
        for target, source in zip(
            self.classification_outputs,
            one_to_many.classification_heads,
        ):
            target.load_state_dict(source[-1].state_dict())

        self.verify_initialized_from(one_to_many)

    @torch.no_grad()
    def verify_initialized_from(self, one_to_many: DecoupledDetectionHead) -> None:
        """Fail closed unless every O2O output parameter exactly matches O2M."""

        mismatches: list[str] = []
        verified_parameters: set[str] = set()
        if len(self.regression_outputs) != len(one_to_many.regression_heads):
            mismatches.append("regression_outputs.level_count")
        for level, (target, source) in enumerate(
            zip(self.regression_outputs, one_to_many.regression_heads)
        ):
            source_output = source[-1]
            for name in ("weight", "bias"):
                parameter_name = f"regression_outputs.{level}.{name}"
                verified_parameters.add(parameter_name)
                if not torch.equal(getattr(target, name), getattr(source_output, name)):
                    mismatches.append(parameter_name)
        if len(self.classification_outputs) != len(one_to_many.classification_heads):
            mismatches.append("classification_outputs.level_count")
        for level, (target, source) in enumerate(
            zip(self.classification_outputs, one_to_many.classification_heads)
        ):
            source_output = source[-1]
            for name in ("weight", "bias"):
                parameter_name = f"classification_outputs.{level}.{name}"
                verified_parameters.add(parameter_name)
                if not torch.equal(getattr(target, name), getattr(source_output, name)):
                    mismatches.append(parameter_name)

        if (self.regression_log_scales is None) != (one_to_many.regression_log_scales is None):
            mismatches.append("regression_log_scales.presence")
        elif self.regression_log_scales is not None and not torch.equal(
            self.regression_log_scales,
            one_to_many.regression_log_scales,
        ):
            mismatches.append("regression_log_scales")
        if self.regression_log_scales is not None:
            verified_parameters.add("regression_log_scales")

        unverified = sorted(set(dict(self.named_parameters())) - verified_parameters)
        mismatches.extend(f"unverified:{name}" for name in unverified)

        if mismatches:
            raise RuntimeError(
                "one-to-one head initialization is incomplete; mismatched O2M state: "
                + ", ".join(mismatches)
            )

    def forward(
        self,
        hidden_outputs: list[tuple[torch.Tensor, torch.Tensor]],
        *,
        class_level_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = []
        for level, (
            (regression_hidden, classification_hidden),
            regression,
            classification,
        ) in enumerate(
            zip(
                hidden_outputs,
                self.regression_outputs,
                self.classification_outputs,
            )
        ):
            class_logits = classification(classification_hidden)
            if class_level_bias is not None:
                class_logits = class_logits + class_level_bias[:, level, None, :]
            regression_logits = regression(regression_hidden)
            if self.regression_log_scales is not None:
                regression_logits = regression_logits * self.regression_log_scales[level].exp()
            outputs.append(
                torch.cat(
                    (
                        regression_logits,
                        class_logits,
                    ),
                    dim=-1,
                )
            )
        return torch.cat(outputs, dim=1)
