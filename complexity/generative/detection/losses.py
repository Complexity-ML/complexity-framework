"""Detection losses used by the quality-aware TR-Hash head."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def quality_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    beta: float = 2.0,
) -> torch.Tensor:
    """Quality Focal Loss for joint class-presence and localization quality."""

    probabilities = logits.sigmoid()
    modulation = (targets - probabilities).abs().pow(beta)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    normalizer = (targets > 0).sum().clamp_min(1)
    return (loss * modulation).sum() / normalizer


def distribution_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    reg_max: int,
) -> torch.Tensor:
    """Interpolate cross-entropy between adjacent LTRB distance bins.

    Returns one scalar loss per positive prediction so assignment-quality
    weights can be applied by the caller.
    """

    targets = targets.clamp(0, reg_max - 1e-3)
    lower = targets.floor().long()
    upper = (lower + 1).clamp_max(reg_max)
    upper_weight = targets - lower.to(targets.dtype)
    lower_weight = 1.0 - upper_weight
    flat_logits = logits.reshape(-1, reg_max + 1)
    lower_loss = F.cross_entropy(flat_logits, lower.reshape(-1), reduction="none")
    upper_loss = F.cross_entropy(flat_logits, upper.reshape(-1), reduction="none")
    interpolated = lower_loss * lower_weight.reshape(-1) + upper_loss * upper_weight.reshape(-1)
    return interpolated.reshape(-1, 4).mean(-1)
