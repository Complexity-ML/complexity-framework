#!/usr/bin/env python3
"""Train the 200M TR-Hash latent flow model on Atlas Images TAR shards."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from complexity.generative.image import (
    AtlasImageTarDataset,
    FrozenAutoencoderKL,
    TRHashImageConfig,
    TRHashTextToImage,
    collate_atlas_images,
)

LOGGER = logging.getLogger("tr_hash_text_to_image")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/tr_hash_text_to_image_200m.yaml"))
    parser.add_argument("--shards", required=True, help="Local glob for train TAR shards")
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer/tokenizer.json"))
    parser.add_argument("--vae", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint directory")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="0 consumes all requested epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per GPU")
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--save-steps", type=int, default=5_000)
    parser.add_argument("--save-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return rank, local_rank, world_size, device


def tokenize_captions(
    tokenizer: Tokenizer,
    captions: Sequence[str],
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer.encode_batch(list(captions))
    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        pad_id = 0
    lengths = [min(len(item.ids), max_length) for item in encoded]
    width = max(max(lengths, default=1), 1)
    ids = torch.full((len(encoded), width), int(pad_id), dtype=torch.long)
    mask = torch.zeros((len(encoded), width), dtype=torch.bool)
    for index, item in enumerate(encoded):
        values = item.ids[:max_length]
        if not values:
            values = [int(pad_id)]
        ids[index, : len(values)] = torch.tensor(values)
        mask[index, : len(values)] = True
    return ids.to(device, non_blocking=True), mask.to(device, non_blocking=True)


def save_checkpoint(
    output: Path,
    model: TRHashTextToImage,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: TRHashImageConfig,
    step: int,
) -> None:
    target = output / f"step_{step:07d}"
    target.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    save_file(state, str(target / "model.safetensors"))
    (target / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    torch.save(
        {"step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        target / "training_state.pt",
    )
    LOGGER.info("Checkpoint saved: %s", target)


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, device = setup_distributed()
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    values = yaml.safe_load(args.config.read_text())
    config = TRHashImageConfig.from_dict(values)
    shards = sorted(Path(path) for path in glob.glob(args.shards))
    if not shards:
        raise FileNotFoundError(f"no shards match {args.shards!r}")
    dataset = AtlasImageTarDataset(
        shards,
        image_size=config.image_size,
        rank=rank,
        world_size=world_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate_atlas_images,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    codec = FrozenAutoencoderKL(args.vae).to(device)
    model = TRHashTextToImage(config).to(device)
    model.gradient_checkpointing = args.gradient_checkpointing
    raw_model = model
    if args.resume is not None:
        resume_config = TRHashImageConfig.from_dict(
            json.loads((args.resume / "config.json").read_text())
        )
        if resume_config != config:
            raise ValueError("resume checkpoint config does not match --config")
        raw_model.load_state_dict(load_file(str(args.resume / "model.safetensors")))
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )
    estimated_batches = math.ceil(329_510 / max(world_size * args.batch_size, 1))
    estimated_steps = math.ceil(estimated_batches * args.epochs / args.gradient_accumulation)
    total_steps = args.max_steps or estimated_steps

    def lr_factor(step: int) -> float:
        if step < args.warmup_steps:
            return max(step, 1) / max(args.warmup_steps, 1)
        progress = min((step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1), 1.0)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    step = 0
    if args.resume is not None:
        state = torch.load(
            args.resume / "training_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = int(state["step"])
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Model: %.1fM parameters", raw_model.num_parameters() / 1e6)
        LOGGER.info("Dataset: %d shards; world=%d batch/GPU=%d", len(shards), world_size, args.batch_size)
        LOGGER.info("Training: AdamW lr=%g steps=%d epochs=%d", args.lr, total_steps, args.epochs)

    progress = tqdm(
        total=total_steps,
        initial=step,
        desc="TR-Hash image",
        unit="step",
        dynamic_ncols=True,
        disable=rank != 0,
    )

    micro_step = 0
    running_loss = 0.0
    running_samples = 0
    last_loss = float("nan")
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    for _epoch in range(args.epochs):
        for batch in loader:
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            caption_ids, caption_mask = tokenize_captions(
                tokenizer, batch["captions"], config.max_text_length, device
            )
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=args.bf16 and device.type in {"cuda", "cpu"},
            ):
                latents = codec.encode(pixels)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=args.bf16 and device.type in {"cuda", "cpu"},
            ):
                batch_size = latents.size(0)
                noise = torch.randn_like(latents)
                timesteps = torch.rand(batch_size, device=device, dtype=latents.dtype)
                dropped = torch.rand(batch_size, device=device) < config.caption_dropout
                training_mask = caption_mask.clone()
                training_mask[dropped] = False
                view_shape = (batch_size,) + (1,) * (latents.ndim - 1)
                interpolation = timesteps.view(view_shape)
                noisy = (1.0 - interpolation) * latents + interpolation * noise
                prediction = model(noisy, timesteps, caption_ids, training_mask)
                loss = F.mse_loss(prediction.float(), (noise - latents).float())
                scaled_loss = loss / args.gradient_accumulation
            scaled_loss.backward()
            micro_step += 1
            running_loss += float(loss.detach())
            last_loss = float(loss.detach())
            running_samples += pixels.size(0) * world_size
            if micro_step % args.gradient_accumulation:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            progress.update(1)
            if rank == 0 and step % args.log_steps == 0:
                elapsed = time.monotonic() - started
                average_loss = running_loss / args.log_steps
                throughput = running_samples / max(elapsed, 1e-6)
                progress.set_postfix(
                    loss=f"{average_loss:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    img_s=f"{throughput:.1f}",
                    refresh=True,
                )
                LOGGER.info(
                    "step=%d/%d loss=%.5f lr=%.3e images/s=%.1f",
                    step,
                    total_steps,
                    average_loss,
                    scheduler.get_last_lr()[0],
                    throughput,
                )
                running_loss = 0.0
            if rank == 0 and args.save_steps and step % args.save_steps == 0:
                save_checkpoint(args.output, raw_model, optimizer, scheduler, config, step)
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    if rank == 0:
        elapsed = time.monotonic() - started
        summary = {
            "steps": step,
            "last_loss": last_loss,
            "elapsed_seconds": elapsed,
            "images_seen": running_samples,
            "images_per_second": running_samples / max(elapsed, 1e-6),
            "world_size": world_size,
            "batch_size_per_gpu": args.batch_size,
        }
        (args.output / "training_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        LOGGER.info(
            "Training complete: step=%d loss=%.5f elapsed=%.1fs images/s=%.1f",
            step,
            last_loss,
            elapsed,
            summary["images_per_second"],
        )
        if args.save_final:
            save_checkpoint(args.output, raw_model, optimizer, scheduler, config, step)
    progress.close()
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
