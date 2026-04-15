"""
Causal LM loss primitives for framework-complexity.

Combines cross-entropy with optional label smoothing and z-loss regularization.
The z-loss term penalizes the squared logsumexp of the logits, preventing the
logit norm from drifting during long pretraining runs (PaLM, Gemini).

Usage:
    from complexity.core.losses import causal_lm_loss

    outputs = model(input_ids)
    logits  = outputs["logits"]  # [B, S, V]
    loss, metrics = causal_lm_loss(
        logits, labels,
        label_smoothing=0.1,
        z_loss_coef=1e-4,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


@dataclass
class CausalLMLossMetrics:
    ce: float           # raw cross-entropy (for ppl)
    z_loss: float       # z-loss contribution (0 if disabled)
    total: float        # ce + z_loss

    def as_dict(self) -> dict:
        return {"ce": self.ce, "z_loss": self.z_loss, "total": self.total}


def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
    z_loss_coef: float = 0.0,
    ignore_index: int = -100,
    shift: bool = False,
) -> Tuple[torch.Tensor, CausalLMLossMetrics]:
    """
    Cross-entropy + optional label smoothing + optional z-loss.

    Args:
        logits: [B, S, V] or [N, V]
        labels: [B, S]    or [N]
        label_smoothing: 0.0 disables. Typical 0.1.
        z_loss_coef: 0.0 disables. Typical 1e-4. Penalizes logsumexp(logits)^2.
            Computed in float32 for numerical stability regardless of autocast dtype.
        ignore_index: label id to ignore in the CE sum (default -100, torch default).
        shift: if True, shift so that logits[:, :-1] predict labels[:, 1:]. Use when
            the caller passes full sequences aligned with input_ids. Default False
            (caller pre-shifts the labels, matching the rest of the codebase).

    Returns:
        loss: scalar tensor (ce + z_loss_coef·z_loss) — the thing to backprop on
        metrics: CausalLMLossMetrics with detached floats for logging
    """
    if shift:
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

    vocab = logits.size(-1)
    flat_logits = logits.reshape(-1, vocab)
    flat_labels = labels.reshape(-1)

    ce = F.cross_entropy(
        flat_logits,
        flat_labels,
        label_smoothing=label_smoothing,
        ignore_index=ignore_index,
    )

    if z_loss_coef > 0.0:
        # float32 for numerical stability under bf16/fp16 autocast
        lse = flat_logits.float().logsumexp(dim=-1)
        if ignore_index is not None:
            mask = flat_labels != ignore_index
            lse = lse[mask] if mask.any() else lse
        z = lse.pow(2).mean()
        loss = ce + z_loss_coef * z
        z_val = z.detach().item()
    else:
        loss = ce
        z_val = 0.0

    metrics = CausalLMLossMetrics(
        ce=ce.detach().item(),
        z_loss=z_val,
        total=loss.detach().item(),
    )
    return loss, metrics
