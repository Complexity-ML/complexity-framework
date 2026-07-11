"""Runtime helpers for the o200k pretraining runner."""

from __future__ import annotations

import os
import math
from dataclasses import dataclass

import torch
import torch.distributed as dist

from complexity.core.losses import causal_lm_loss_from_hidden
from complexity.utils import autocast, setup_mps


@dataclass(frozen=True)
class RuntimeControls:
    """Controls and learned scalar gates actually present in a built model."""

    token_routed_layers: int
    lexical_layers: int
    object_gate: float | None
    micro_gate: float | None


@torch.no_grad()
def runtime_controls(model) -> RuntimeControls:
    """Inspect active routing controls instead of trusting generic CLI defaults."""

    token_routed_layers = 0
    object_gates = []
    micro_gates = []
    for module in model.modules():
        if callable(getattr(module, "set_top_k_primary_weight", None)):
            token_routed_layers += 1
        object_gate = getattr(module, "object_output_gate", None)
        micro_gate = getattr(module, "micro_output_gate", None)
        if object_gate is not None:
            object_gates.append(float(object_gate.detach().float().item()))
        if micro_gate is not None:
            micro_gates.append(float(micro_gate.detach().float().item()))
    lexical_layers = max(len(object_gates), len(micro_gates))
    return RuntimeControls(
        token_routed_layers=token_routed_layers,
        lexical_layers=lexical_layers,
        object_gate=sum(object_gates) / len(object_gates) if object_gates else None,
        micro_gate=sum(micro_gates) / len(micro_gates) if micro_gates else None,
    )


@torch.no_grad()
def evaluate(
    model,
    raw_model,
    loader,
    device,
    amp_dtype,
    eval_batches,
    label_smoothing,
    z_loss,
    loss_chunk_tokens,
    distributed,
):
    was_training = model.training
    model.eval()
    loss_sum = None
    loss_count = 0
    for idx, batch in enumerate(loader):
        if idx >= eval_batches:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with autocast(device, dtype=amp_dtype, enabled=amp_dtype is not None):
            outputs = model(input_ids, return_logits=False)
            loss, _ = causal_lm_loss_from_hidden(
                outputs["last_hidden_state"],
                raw_model.embed_tokens.weight,
                labels,
                label_smoothing=label_smoothing,
                z_loss_coef=z_loss,
                chunk_tokens=loss_chunk_tokens,
                checkpoint_chunks=False,
                sync_metrics=False,
            )
        detached = loss.detach()
        loss_sum = detached if loss_sum is None else loss_sum + detached
        loss_count += 1
    if was_training:
        model.train()
    if loss_sum is None:
        loss_tensor = torch.tensor(float("nan"), device=device)
    else:
        loss_tensor = loss_sum / max(1, loss_count)
    if distributed:
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
    return loss_tensor.item()


def init_distributed(seed: int):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires a CUDA-compatible GPU backend (NVIDIA CUDA or AMD ROCm).")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        torch.manual_seed(seed + rank)
        return torch.device("cuda", local_rank), distributed, rank, local_rank, world_size

    device = setup_mps(unlimited_watermark=True, cpu_fallback=True, seed=seed)
    return device, distributed, rank, local_rank, world_size


def reduce_average(value: float, device: torch.device, distributed: bool) -> float:
    if not distributed:
        return value
    tensor = torch.tensor(float(value), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return tensor.item()


def reduce_average_tensor(value: torch.Tensor, distributed: bool) -> float:
    tensor = value.detach().float()
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return tensor.item()


def scheduled_topk_primary_weight(
    step: int,
    total_steps: int,
    start: float,
    final: float,
    schedule_ratio: float,
) -> float:
    """Cosine ramp from lexical mixing toward primary expert specialization."""
    start = min(1.0, max(0.0, float(start)))
    final = min(1.0, max(0.0, float(final)))
    ratio = min(1.0, max(0.0, float(schedule_ratio)))
    ramp_steps = max(1, int(max(1, total_steps) * ratio))
    if ratio <= 0.0 or ramp_steps <= 1:
        return final
    progress = min(1.0, max(0.0, step / ramp_steps))
    blend = 0.5 - 0.5 * math.cos(math.pi * progress)
    return start + (final - start) * blend


def apply_topk_primary_weight(model, weight: float) -> int:
    """Apply a scheduled top-k primary route weight to all Token-Routed layers."""
    count = 0
    for module in model.modules():
        setter = getattr(module, "set_top_k_primary_weight", None)
        if setter is None:
            continue
        setter(weight)
        count += 1
    return count


def scheduled_value(
    step: int,
    total_steps: int,
    start: float,
    final: float | None,
    schedule_ratio: float,
) -> float:
    """Cosine ramp helper for optional scalar curricula."""
    if final is None:
        return float(start)
    ratio = min(1.0, max(0.0, float(schedule_ratio)))
    ramp_steps = max(1, int(max(1, total_steps) * ratio))
    if ratio <= 0.0 or ramp_steps <= 1:
        return float(final)
    progress = min(1.0, max(0.0, step / ramp_steps))
    blend = 0.5 - 0.5 * math.cos(math.pi * progress)
    return float(start) + (float(final) - float(start)) * blend


@torch.no_grad()
def apply_shared_routed_gates(model, shared_gate: float | None, routed_gate: float | None) -> int:
    """Apply scheduled shared/routed scalar gates to Token-Routed layers."""
    if shared_gate is None and routed_gate is None:
        return 0
    count = 0
    for module in model.modules():
        shared_param = getattr(module, "shared_output_gate", None)
        routed_param = getattr(module, "routed_output_gate", None)
        if shared_param is None or routed_param is None:
            continue
        if shared_gate is not None:
            shared_param.fill_(float(shared_gate))
        if routed_gate is not None:
            routed_param.fill_(float(routed_gate))
        count += 1
    return count


def expert_diversity_loss(model, target: str = "down") -> torch.Tensor | None:
    """Penalize expert weight colinearity within each Token-Routed layer.

    The penalty is the mean squared off-diagonal cosine similarity between
    expert matrices. Minimizing it encourages experts to carve different
    directions while leaving the deterministic routing rule unchanged.
    """
    losses = []
    for module in model.modules():
        down_w = getattr(module, "down_proj_w", None)
        if down_w is None or down_w.ndim != 3:
            continue
        weights = [down_w]
        if target == "all":
            gate_w = getattr(module, "gate_proj_w", None)
            up_w = getattr(module, "up_proj_w", None)
            if gate_w is not None and up_w is not None:
                weights = [gate_w, up_w, down_w]
        for weight in weights:
            flat = weight.float().reshape(weight.shape[0], -1)
            flat = torch.nn.functional.normalize(flat, dim=1, eps=1e-6)
            sim = flat @ flat.transpose(0, 1)
            off_diag = sim - torch.eye(sim.shape[0], device=sim.device, dtype=sim.dtype)
            losses.append(off_diag.pow(2).sum() / max(1, sim.numel() - sim.shape[0]))
    if not losses:
        return None
    return torch.stack(losses).mean()
