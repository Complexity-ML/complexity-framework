"""Export-based pipeline helpers for large Token-Routed runs.

This module uses ``torch.distributed.pipelining`` instead of the legacy
hand-written pipeline wrapper. It is intentionally small: build deterministic
split specs, validate layer/stage layouts, and trace a model into a Pipe.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn


def pipeline_split_spec(
    num_layers: int,
    pp_size: int,
    *,
    layers_prefix: str = "layers",
    require_even: bool = True,
) -> dict[str, Any]:
    """Return a torch.distributed.pipelining split spec for layer stacks.

    The split points are inserted at the beginning of the first layer assigned
    to each non-zero pipeline stage. For 32 layers and PP=8 this returns
    ``layers.4``, ``layers.8``, ..., ``layers.28``.
    """
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if pp_size <= 0:
        raise ValueError("pp_size must be positive")
    if pp_size > num_layers:
        raise ValueError("pp_size cannot exceed num_layers")
    if require_even and num_layers % pp_size != 0:
        raise ValueError(
            f"num_layers ({num_layers}) must be divisible by pp_size ({pp_size})"
        )

    from torch.distributed.pipelining import SplitPoint

    split_spec: dict[str, Any] = {}
    for stage_idx in range(1, pp_size):
        layer_idx = stage_idx * num_layers // pp_size
        split_spec[f"{layers_prefix}.{layer_idx}"] = SplitPoint.BEGINNING
    return split_spec


class LastHiddenStateWrapper(nn.Module):
    """Tensor-output wrapper for ``ComplexityModel`` pipeline tracing."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids, return_logits=False)
        if isinstance(outputs, Mapping):
            return outputs["last_hidden_state"]
        return outputs


def enable_pipeline_export_mode(model: nn.Module) -> None:
    """Switch Token-Routed MLPs to export-friendly dispatch in-place."""
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "static_expert_capacity"):
            config.static_expert_capacity = True


def trace_pipeline(
    model: nn.Module,
    example_input_ids: torch.Tensor,
    *,
    pp_size: int,
    layers_prefix: str = "model.layers",
    return_last_hidden: bool = True,
) -> Any:
    """Trace ``model`` into a ``torch.distributed.pipelining.Pipe``.

    ``return_last_hidden=True`` avoids dict outputs and the large LM head,
    matching the framework's chunked-loss training path.
    """
    from torch.distributed.pipelining import pipeline

    if pp_size <= 0:
        raise ValueError("pp_size must be positive")

    enable_pipeline_export_mode(model)
    traced_model = LastHiddenStateWrapper(model) if return_last_hidden else model
    num_layers = len(getattr(model, "layers"))
    split_spec = pipeline_split_spec(
        num_layers,
        pp_size,
        layers_prefix=layers_prefix if return_last_hidden else "layers",
    )
    return pipeline(traced_model, mb_args=(example_input_ids,), split_spec=split_spec)

