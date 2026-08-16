"""Detection losses used by the quality-aware TR-Hash head."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class _QualityFocalLossFunction(torch.autograd.Function):
    """Recompute QFL intermediates in backward instead of retaining them.

    Dense detection at 640 px with hundreds of classes otherwise keeps several
    full ``[batch, cells, classes]`` tensors alive for every detector branch.
    Saving only the inputs cuts the activation peak without changing the loss.
    """

    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
        beta: float,
    ) -> torch.Tensor:
        broadcast_weights = torch.broadcast_to(weights, targets.shape)
        probabilities = logits.sigmoid()
        modulation = (targets - probabilities).abs().pow(beta)
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        normalizer = (broadcast_weights * (targets > 0)).sum().clamp_min(1)
        result = (loss * modulation * broadcast_weights).sum() / normalizer
        ctx.save_for_backward(logits, targets, weights, normalizer)
        ctx.beta = beta
        return result

    @staticmethod
    def backward(ctx, output_gradient: torch.Tensor):
        logits, targets, weights, normalizer = ctx.saved_tensors
        beta = ctx.beta
        broadcast_weights = torch.broadcast_to(weights, targets.shape)
        logits_gradient = torch.empty_like(logits)
        class_chunk_size = 32
        for class_start in range(0, logits.shape[-1], class_chunk_size):
            class_end = min(class_start + class_chunk_size, logits.shape[-1])
            class_slice = slice(class_start, class_end)
            chunk_logits = logits[..., class_slice]
            chunk_targets = targets[..., class_slice]
            chunk_weights = broadcast_weights[..., class_slice]
            probabilities = chunk_logits.sigmoid()
            delta = probabilities - chunk_targets
            absolute_delta = delta.abs()
            modulation = absolute_delta.pow(beta)
            if beta == 0.0:
                modulation_gradient = torch.zeros_like(delta)
            else:
                modulation_gradient = (
                    beta
                    * absolute_delta.pow(beta - 1.0)
                    * delta.sign()
                    * probabilities
                    * (1.0 - probabilities)
                )
            binary_cross_entropy = F.binary_cross_entropy_with_logits(
                chunk_logits,
                chunk_targets,
                reduction="none",
            )
            chunk_gradient = (
                (delta * modulation + binary_cross_entropy * modulation_gradient)
                * chunk_weights
                / normalizer
            )
            logits_gradient[..., class_slice].copy_(chunk_gradient)
        logits_gradient.mul_(output_gradient.to(logits_gradient.dtype))
        return logits_gradient, None, None, None


def quality_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    beta: float = 2.0,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Quality Focal Loss for joint class-presence and localization quality."""

    if weights is None:
        weights = logits.new_ones(())
    return _QualityFocalLossFunction.apply(logits, targets, weights, beta)


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
