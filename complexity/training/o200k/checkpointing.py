"""Checkpoint helpers for o200k pretraining."""

from __future__ import annotations

import logging

import torch.distributed as dist

from complexity.utils.device import backend_metadata
from complexity.utils.local_checkpoint import load_local_checkpoint, save_local_checkpoint

logger = logging.getLogger(__name__)


def save_checkpoint(args, raw_model, optimizer, scheduler, config, step: int, is_main: bool, distributed: bool):
    if distributed:
        dist.barrier()
    if not is_main or args.save_steps <= 0:
        if distributed:
            dist.barrier()
        return

    ckpt_dir = save_local_checkpoint(
        args.save_dir,
        step=step,
        total_limit=args.save_total_limit,
        state={
            "step": step,
            "model": {k: v.detach().cpu() for k, v in raw_model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config.to_dict(),
            "args": vars(args),
            "backend": backend_metadata(kernel_policy=getattr(args, "use_custom_kernels", "auto")),
        },
    )
    logger.info(f"Checkpoint saved: {ckpt_dir}")
    if distributed:
        dist.barrier()


def load_checkpoint(path: str, raw_model, optimizer, scheduler, device, is_main: bool) -> int:
    ckpt_dir, state = load_local_checkpoint(path, map_location=device)
    raw_model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    step = int(state["step"])
    if is_main:
        logger.info(f"Resumed from {ckpt_dir} at step {step}")
    return step
