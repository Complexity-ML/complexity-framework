"""Runtime helpers for the o200k pretraining runner."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from complexity.core.losses import causal_lm_loss_from_hidden
from complexity.utils import autocast, setup_mps


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
    losses = []
    for idx, batch in enumerate(loader):
        if idx >= eval_batches:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with autocast(device, dtype=amp_dtype, enabled=amp_dtype is not None):
            outputs = model(input_ids, return_logits=False)
            _, metrics = causal_lm_loss_from_hidden(
                outputs["last_hidden_state"],
                raw_model.embed_tokens.weight,
                labels,
                label_smoothing=label_smoothing,
                z_loss_coef=z_loss,
                chunk_tokens=loss_chunk_tokens,
                checkpoint_chunks=False,
            )
        losses.append(metrics.ce)
    if was_training:
        model.train()
    eval_loss = sum(losses) / max(1, len(losses))
    if distributed:
        loss_tensor = torch.tensor(eval_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        eval_loss = loss_tensor.item()
    return eval_loss


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
