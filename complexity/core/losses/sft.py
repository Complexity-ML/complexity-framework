"""Explicit SFT loss policy for aligned labels and optional curriculum weights.

These helpers do not select a training workflow. The active portable 200M SFT
keeps its existing loss. CPU dispatch tests cannot qualify a Liger CUDA kernel.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F

from .causal_lm import causal_lm_loss_from_hidden
from .fused_ce import fused_linear_causal_lm_loss, has_liger_fused_linear_ce


def validate_sft_loss_backend(device, *, fp32_loss: bool, liger_loss: bool) -> None:
    """Qualify an explicitly requested Liger runtime before model/data allocation."""
    device = torch.device(device)
    if not liger_loss:
        return
    if fp32_loss:
        raise ValueError("Liger SFT requires fp32_loss=False")
    if device.type != "cuda":
        raise ValueError("Liger SFT requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("Liger SFT requires available CUDA; refusing a CPU fallback")
    if not has_liger_fused_linear_ce():
        raise RuntimeError("Liger SFT requires the fused linear cross-entropy backend")


def sft_loss_from_hidden(
    hidden_states: torch.Tensor,
    output_weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_tokens: int,
    example_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """FP32 CE on already aligned targets, normalized by visible weight mass.

    Weights may have shape B or B×S. Ignored context, padding and separators
    contribute no mass; a supervised EOS remains an ordinary target.
    """
    if hidden_states.shape[:-1] != labels.shape:
        raise ValueError("Hidden states and aligned labels must have matching token shapes")
    if output_weight.ndim != 2 or output_weight.shape[1] != hidden_states.shape[-1]:
        raise ValueError("Output weight and hidden width differ")
    if type(chunk_tokens) is not int or chunk_tokens < 0:
        raise ValueError("chunk_tokens must be a nonnegative integer")
    flat_hidden, flat_labels = hidden_states.reshape(-1, hidden_states.size(-1)), labels.reshape(-1)
    valid = flat_labels != -100
    if not valid.any():
        raise ValueError("SFT loss needs at least one supervised target")
    denominator = valid.sum().to(dtype=torch.float32)
    weights = None
    if example_weights is not None:
        if example_weights.ndim == 1 and example_weights.shape[0] == labels.shape[0]:
            weights = example_weights.to(device=hidden_states.device, dtype=torch.float32)
            weights = weights.view(-1, *([1] * (labels.ndim - 1))).expand_as(labels).reshape(-1)
        elif example_weights.shape == labels.shape:
            weights = example_weights.to(device=hidden_states.device, dtype=torch.float32).reshape(-1)
        else:
            raise ValueError("example_weights must contain one scalar per batch example or per token")
        if not torch.isfinite(weights).all() or torch.any(weights < 0):
            raise ValueError("SFT loss weights must be finite and nonnegative")
        denominator = weights[valid].sum()
        if denominator <= 0:
            raise ValueError("SFT loss requires positive visible weight mass")
    total = flat_hidden.new_zeros((), dtype=torch.float32)
    chunk = chunk_tokens or flat_hidden.shape[0]
    # Casting operands alone does not force FP32 while CUDA/CPU autocast is active.
    context = (
        torch.autocast(hidden_states.device.type, enabled=False)
        if hidden_states.device.type in {"cpu", "cuda"}
        else nullcontext()
    )
    with context:
        for start in range(0, len(flat_labels), chunk):
            end = min(start + chunk, len(flat_labels))
            visible = valid[start:end]
            if not visible.any():
                continue
            logits = F.linear(flat_hidden[start:end][visible].float(), output_weight.float())
            losses = F.cross_entropy(logits, flat_labels[start:end][visible], reduction="none")
            if weights is not None:
                losses = losses * weights[start:end][visible]
            total = total + losses.sum()
    return total / denominator


def compute_sft_loss(
    hidden_states: torch.Tensor,
    output_weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    fp32_loss: bool,
    liger_loss: bool,
    chunk_tokens: int,
    example_weights: torch.Tensor | None = None,
    sync_metrics: bool = True,
) -> tuple[torch.Tensor, float]:
    """Dispatch loss independently of runtime qualification and workflow launch.

    Call ``validate_sft_loss_backend`` before a runtime that requires Liger.
    The framework fused helper may otherwise fall back on CPU/MPS. Weighted
    loss is supported only by the explicit FP32 path and never silently lost.
    """
    if liger_loss:
        if example_weights is not None:
            raise ValueError("Liger SFT loss does not support per-example loss weights")
        loss, metrics = fused_linear_causal_lm_loss(
            hidden_states, output_weight, labels, use_liger=True, sync_metrics=sync_metrics
        )
        return loss, float(metrics.ce)
    if fp32_loss:
        loss = sft_loss_from_hidden(
            hidden_states, output_weight, labels,
            chunk_tokens=chunk_tokens, example_weights=example_weights,
        )
        return loss, float(loss.detach()) if sync_metrics else float("nan")
    if example_weights is not None:
        raise ValueError("Weighted SFT loss requires fp32_loss=True")
    loss, metrics = causal_lm_loss_from_hidden(
        hidden_states, output_weight, labels,
        shift=False, chunk_tokens=chunk_tokens, sync_metrics=sync_metrics,
    )
    return loss, float(metrics.ce)
