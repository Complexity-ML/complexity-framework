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
from torch.utils.checkpoint import checkpoint as activation_checkpoint


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
    sync_metrics: bool = True,
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
        z_val = z.detach().item() if sync_metrics else float("nan")
    else:
        loss = ce
        z_val = 0.0

    loss_val = loss.detach().item() if sync_metrics else float("nan")
    ce_val = ce.detach().item() if sync_metrics else float("nan")
    metrics = CausalLMLossMetrics(
        ce=ce_val,
        z_loss=z_val,
        total=loss_val,
    )
    return loss, metrics


def causal_lm_loss_from_hidden(
    hidden_states: torch.Tensor,
    output_weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
    z_loss_coef: float = 0.0,
    ignore_index: int = -100,
    shift: bool = False,
    chunk_tokens: int = 0,
    checkpoint_chunks: bool = True,
    sync_metrics: bool = True,
) -> Tuple[torch.Tensor, CausalLMLossMetrics]:
    """
    Cross-entropy from hidden states and a tied LM head weight.

    This is mathematically equivalent to materializing
    ``logits = F.linear(hidden_states, output_weight)`` then calling
    ``causal_lm_loss``. With large vocabularies (for example o200k) the full
    [B, S, V] logits tensor can dominate memory, so ``chunk_tokens`` computes
    the loss over flat token chunks and sums the exact CE.
    """
    if chunk_tokens is None or chunk_tokens <= 0:
        logits = F.linear(hidden_states, output_weight)
        return causal_lm_loss(
            logits,
            labels,
            label_smoothing=label_smoothing,
            z_loss_coef=z_loss_coef,
            ignore_index=ignore_index,
            shift=shift,
            sync_metrics=sync_metrics,
        )

    if shift:
        hidden_states = hidden_states[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

    flat_hidden = hidden_states.reshape(-1, hidden_states.size(-1))
    flat_labels = labels.reshape(-1)
    valid = flat_labels != ignore_index if ignore_index is not None else torch.ones_like(flat_labels, dtype=torch.bool)
    denom = valid.sum().clamp_min(1).to(dtype=torch.float32)

    def chunk_loss_sum(hidden_chunk: torch.Tensor, labels_chunk: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hidden_chunk, weight)
        ce_sum = F.cross_entropy(
            logits,
            labels_chunk,
            label_smoothing=label_smoothing,
            ignore_index=ignore_index,
            reduction="sum",
        )
        if z_loss_coef <= 0.0:
            return ce_sum
        lse = logits.float().logsumexp(dim=-1)
        if ignore_index is not None:
            mask = labels_chunk != ignore_index
            lse = lse[mask] if mask.any() else lse[:0]
        z_sum = lse.pow(2).sum()
        return ce_sum + z_loss_coef * z_sum

    total = flat_hidden.new_zeros(())
    use_checkpoint = checkpoint_chunks and torch.is_grad_enabled() and flat_hidden.requires_grad
    for start in range(0, flat_hidden.size(0), chunk_tokens):
        end = min(start + chunk_tokens, flat_hidden.size(0))
        hidden_chunk = flat_hidden[start:end]
        labels_chunk = flat_labels[start:end]
        if use_checkpoint:
            total = total + activation_checkpoint(
                chunk_loss_sum,
                hidden_chunk,
                labels_chunk,
                output_weight,
                use_reentrant=False,
            )
        else:
            total = total + chunk_loss_sum(hidden_chunk, labels_chunk, output_weight)

    loss = total / denom
    loss_val = loss.detach().item() if sync_metrics else float("nan")
    metrics = CausalLMLossMetrics(
        ce=loss_val,
        z_loss=0.0 if z_loss_coef <= 0.0 else float("nan"),
        total=loss_val,
    )
    return loss, metrics
