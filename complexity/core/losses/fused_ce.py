"""
Fused linear + cross-entropy loss, Triton-accelerated via Liger Kernel.

The naive sequence ``logits = F.linear(hidden, weight); loss = CE(logits, labels)``
materializes a ``[B, S, V]`` tensor that dominates memory and bandwidth at large
vocab. Liger Kernel's ``LigerFusedLinearCrossEntropy`` fuses the projection
and the CE into a single Triton kernel that never materializes the full
logits tensor — ~15-20% end-to-end speedup and ~50% activation memory saved
for vocab=32k, seq=4096.

Falls back to the pure-PyTorch path from :mod:`causal_lm` when Liger is not
installed or when the device is not CUDA (MPS / CPU).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .causal_lm import CausalLMLossMetrics, causal_lm_loss


def _liger_available() -> bool:
    """Cache-friendly check so we don't re-import on every step."""
    if not hasattr(_liger_available, "_cache"):
        try:
            from liger_kernel.transformers.fused_linear_cross_entropy import (
                LigerFusedLinearCrossEntropyFunction,  # noqa: F401
            )
            _liger_available._cache = True
        except Exception:
            _liger_available._cache = False
    return _liger_available._cache


def fused_linear_causal_lm_loss(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
    z_loss_coef: float = 0.0,
    ignore_index: int = -100,
    bias: Optional[torch.Tensor] = None,
    shift: bool = False,
    use_liger: bool = True,
) -> Tuple[torch.Tensor, CausalLMLossMetrics]:
    """
    Compute CE loss directly from hidden states + output projection, without
    materializing logits when Liger is available.

    Args:
        hidden_states: [B, S, H] or [N, H]
        weight:        [V, H] output projection (typically ``model.embed_tokens.weight``
                       when using tied embeddings)
        labels:        [B, S] or [N]
        label_smoothing: 0.0 disables. Typical 0.1.
        z_loss_coef:   0.0 disables. Typical 1e-4. Penalizes ``logsumexp(logits)^2``.
        ignore_index:  label id to skip in CE (default -100).
        bias:          optional [V] bias for the output projection.
        shift:         if True, shift so hidden[:, :-1] predicts labels[:, 1:].
        use_liger:     set False to force the PyTorch fallback.

    Returns:
        (loss, metrics) — same contract as :func:`causal_lm_loss`.
    """
    if shift:
        hidden_states = hidden_states[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

    # Flatten to [N, H] / [N] for Liger (or for our fallback path)
    H = hidden_states.size(-1)
    hidden_flat = hidden_states.reshape(-1, H)
    labels_flat = labels.reshape(-1)

    if use_liger and hidden_flat.is_cuda and _liger_available():
        from liger_kernel.transformers.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyFunction,
        )

        # Liger signature (positional, since keyword names have drifted across versions):
        # (_input, weight, target, bias, ce_weight, ignore_index, lse_square_scale,
        #  label_smoothing, reduction, softcap, return_z_loss)
        out = LigerFusedLinearCrossEntropyFunction.apply(
            hidden_flat,
            weight,
            labels_flat,
            bias,
            None,               # ce_weight
            ignore_index,
            float(z_loss_coef),
            float(label_smoothing),
            "mean",
            None,               # softcap
            z_loss_coef > 0.0,  # return_z_loss
        )
        if z_loss_coef > 0.0:
            loss, z_loss_tensor = out
            z_val = float(z_loss_tensor.detach().item())
            ce_val = float((loss - z_loss_coef * z_loss_tensor).detach().item())
        else:
            loss = out
            ce_val = float(loss.detach().item())
            z_val = 0.0
        metrics = CausalLMLossMetrics(
            ce=ce_val,
            z_loss=z_val,
            total=float(loss.detach().item()),
        )
        return loss, metrics

    # Fallback: materialize logits, delegate to the pure-PyTorch CE path
    logits = F.linear(hidden_flat, weight, bias)
    return causal_lm_loss(
        logits.view(*labels.shape, -1),
        labels,
        label_smoothing=label_smoothing,
        z_loss_coef=z_loss_coef,
        ignore_index=ignore_index,
        shift=False,  # already shifted above if requested
    )
