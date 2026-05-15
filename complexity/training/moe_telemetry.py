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

import math
from typing import Dict, List, Optional, Tuple

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


def global_tr_diagnostics(model: nn.Module, num_experts: Optional[int] = None) -> Dict[str, float]:
    """Aggregate Token-Routed gate, activation, and gradient diagnostics.

    This is intentionally passive: it reads the model after ``loss.backward()``
    and before the next ``zero_grad()``, and never modifies gradients or routing.
    """

    if num_experts is None:
        num_experts = detect_num_experts(model)
    if num_experts is None:
        return {}

    dev = next(model.parameters()).device
    totals = torch.zeros(8 + num_experts, dtype=torch.float32, device=dev)
    # [0] modules, [1] shared_gate_sum, [2] routed_gate_sum,
    # [3] shared_rms_sum, [4] routed_rms_sum,
    # [5] shared_grad_sq, [6] routed_grad_sq, [7] routed_param_count,
    # [8:] per-expert grad_sq.
    for module in model.modules():
        if not hasattr(module, "gate_proj_w") or not hasattr(module, "down_proj_w"):
            continue
        totals[0] += 1
        if hasattr(module, "shared_output_gate") and hasattr(module, "routed_output_gate"):
            totals[1] += _to_local(module.shared_output_gate.detach()).float()
            totals[2] += _to_local(module.routed_output_gate.detach()).float()
        else:
            totals[1] += float("nan")
            totals[2] += float("nan")

        if hasattr(module, "last_shared_rms"):
            totals[3] += torch.nan_to_num(_to_local(module.last_shared_rms).detach().float(), nan=0.0)
        if hasattr(module, "last_routed_rms"):
            totals[4] += torch.nan_to_num(_to_local(module.last_routed_rms).detach().float(), nan=0.0)

        for name, param in module.named_parameters(recurse=False):
            if param.grad is None:
                continue
            grad = _to_local(param.grad.detach()).float()
            grad_sq = grad.pow(2)
            if name in {"gate_proj_w", "up_proj_w", "down_proj_w"} and grad_sq.ndim >= 1:
                per_expert = grad_sq.reshape(grad_sq.shape[0], -1).sum(dim=1)
                totals[8 : 8 + num_experts] += per_expert[:num_experts]
                totals[6] += per_expert.sum()
                totals[7] += grad.numel()
            elif name == "routed_output_gate":
                totals[6] += grad_sq.sum()
            elif name.startswith("shared_"):
                totals[5] += grad_sq.sum()

        for attr in ("shared_gate", "shared_up", "shared_down"):
            submodule = getattr(module, attr, None)
            if submodule is None:
                continue
            for param in submodule.parameters(recurse=False):
                if param.grad is not None:
                    totals[5] += _to_local(param.grad.detach()).float().pow(2).sum()

    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)

    cpu = totals.cpu().to(torch.float64)
    modules = int(cpu[0].item())
    if modules <= 0:
        return {}

    denom = max(1, modules)
    routed_grad_sq = cpu[6].item()
    shared_grad_sq = cpu[5].item()
    per_expert = cpu[8 : 8 + num_experts]
    return {
        "shared_gate": _finite_mean(cpu[1].item(), denom),
        "routed_gate": _finite_mean(cpu[2].item(), denom),
        "shared_rms": cpu[3].item() / denom,
        "routed_rms": cpu[4].item() / denom,
        "shared_grad_norm": math.sqrt(max(0.0, shared_grad_sq)),
        "routed_grad_norm": math.sqrt(max(0.0, routed_grad_sq)),
        **{
            f"expert_{idx}_grad_norm": math.sqrt(max(0.0, per_expert[idx].item()))
            for idx in range(num_experts)
        },
    }


def _finite_mean(value: float, denom: int) -> float:
    if not math.isfinite(value):
        return float("nan")
    return value / denom
