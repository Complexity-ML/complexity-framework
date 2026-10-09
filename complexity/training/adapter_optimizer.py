"""AdamW construction for trainable adapters and routed expert parameters."""

from __future__ import annotations

import math

import torch
from torch import nn

from .adam_tr import adamtr_param_groups


def build_finetuning_optimizer(
    model: nn.Module,
    *,
    learning_rate: float = 2e-5,
    expert_lr_scale: float = 1.0,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.95),
    num_experts: int | None = None,
) -> torch.optim.AdamW:
    """Build AdamW over unfrozen parameters, with an explicit expert LR scale.

    Expert classification is shared with AdamTR. This helper resolves that
    classification into ordinary AdamW groups; it does not apply AdamTR's
    gradient or spectral conditioning. Frozen base weights never enter a group.
    Group order is stable for optimizer checkpoint save/load with the same
    model and adaptation targets.
    """
    for name, value, allow_zero in (
        ("learning_rate", learning_rate, True),
        ("expert_lr_scale", expert_lr_scale, False),
        ("weight_decay", weight_decay, True),
    ):
        if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
            raise ValueError(
                f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}"
            )
    if not math.isfinite(learning_rate * expert_lr_scale):
        raise ValueError("Scaled expert learning rate must be finite")
    if num_experts is None:
        counts = {
            module.config.num_experts
            for module in model.modules()
            if hasattr(getattr(module, "config", None), "num_experts")
        }
        if len(counts) > 1:
            raise ValueError("Models with different expert counts require explicit num_experts")
        num_experts = counts.pop() if counts else 1
    if isinstance(num_experts, bool) or not isinstance(num_experts, int) or num_experts < 1:
        raise ValueError("num_experts must be a positive integer")

    buckets: dict[str, list[nn.Parameter]] = {
        "base_decay": [],
        "base_no_decay": [],
        "expert_decay": [],
        "expert_no_decay": [],
    }
    classified = adamtr_param_groups(
        model,
        lr=learning_rate,
        weight_decay=weight_decay,
        expert_lr_scale=expert_lr_scale,
        expert_weight_decay=weight_decay,
        num_experts=num_experts,
        no_decay_patterns=("bias", "norm"),
    )
    for group in classified:
        parameter = group["params"][0]
        name = group["param_name"].lower()
        no_decay = parameter.ndim < 2 or "bias" in name or "norm" in name
        # Single-expert TR-HASH and its LoRA factors still belong to the expert
        # branch even when AdamTR's multi-expert shape heuristic is disabled.
        expert_tensor = {"expert_gate", "expert_up", "expert_down"}.intersection(name.split("."))
        prefix = "expert" if group["param_type"] == "expert" or expert_tensor else "base"
        buckets[f"{prefix}_{'no_decay' if no_decay else 'decay'}"].append(parameter)

    groups = [
        {
            "name": name,
            "params": parameters,
            "weight_decay": 0.0 if name.endswith("no_decay") else weight_decay,
            "lr": learning_rate * (expert_lr_scale if name.startswith("expert_") else 1.0),
        }
        for name, parameters in buckets.items()
        if parameters
    ]
    if not groups:
        raise ValueError("No trainable parameters: unfreeze parameters or apply adapters first")
    return torch.optim.AdamW(groups, lr=learning_rate, betas=betas)
