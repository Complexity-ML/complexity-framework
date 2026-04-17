"""
Telemetry helpers for Token-Routed MoE training.

Distributed-safe utilities for logging MoE internals from training scripts:

- global_expert_shares(model) — all-reduced expert utilization shares + dead count
- detect_num_experts(model)   — auto-detect num_experts from first TokenRoutedMLP

These helpers are meant to be called from a training callback registered on
ALL ranks (the all_reduce is collective). The caller decides whether to write
the results to disk (typically rank 0 only).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn


def detect_num_experts(model: nn.Module) -> Optional[int]:
    """Return the number of experts from the first TokenRoutedMLP layer, or None."""
    for m in model.modules():
        if hasattr(m, "expert_counts"):
            return int(m.expert_counts.shape[0])
    return None


def _to_local(t: torch.Tensor) -> torch.Tensor:
    """Convert DTensor to a plain local tensor (FSDP v2 compat)."""
    if hasattr(t, "to_local"):
        return t.to_local()
    return t


def global_expert_shares(
    model: nn.Module,
    num_experts: Optional[int] = None,
) -> Tuple[List[float], int]:
    """
    Aggregate ``expert_counts`` across all TokenRoutedMLP layers AND across
    all distributed ranks, then reset local counters.

    The ``expert_counts`` buffer accumulates per-rank only. Under FSDP/DDP
    each rank processes a different data shard, so a single rank's counters
    reflect only its fraction of the tokens. This function performs a
    collective ``all_reduce(SUM)`` so every rank sees the same global shares.

    MUST be called at the same step on every rank — it is a collective.

    Args:
        model: model containing one or more TokenRoutedMLP instances
        num_experts: expected number of experts (for NaN-filled fallback)

    Returns:
        shares: list of length ``num_experts``, summing to 1.0 (or NaN if no
            tokens have been seen yet)
        dead: number of experts with zero tokens in the interval
    """
    if num_experts is None:
        num_experts = detect_num_experts(model)
    if num_experts is None or not any(hasattr(m, "expert_counts") for m in model.modules()):
        return [], 0

    dev = next(model.parameters()).device
    # MPS doesn't support float64; use float32 on-device and upcast on CPU
    on_device_dtype = torch.float32
    total = torch.zeros(num_experts, dtype=on_device_dtype, device=dev)

    for m in model.modules():
        if not (hasattr(m, "expert_counts") and hasattr(m, "reset_expert_counts")):
            continue
        counts = _to_local(m.expert_counts)
        total += counts.detach().to(dev, dtype=on_device_dtype)
        m.reset_expert_counts()

    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(total, op=dist.ReduceOp.SUM)

    total_cpu = total.cpu().to(torch.float64)
    s = total_cpu.sum().item()
    if s <= 0:
        return [float("nan")] * num_experts, num_experts
    shares = (total_cpu / s).tolist()
    dead = sum(1 for x in total_cpu.tolist() if x == 0)
    return shares, dead
