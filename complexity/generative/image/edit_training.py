"""Distributed training runtime for instruction-guided TR-Hash image editing."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from complexity.training.finetuning import (
    IMAGE_EDITING_SUPERVISED_FINETUNING,
    REFINEMENT_STAGE,
    SUPERVISED_FINETUNING_STAGE,
    validate_full_parameter_finetuning,
)

from .codec import FrozenAutoencoderKL
from .config import TRHashImageConfig
from .edit_data import AtlasImageEditTarDataset, collate_atlas_image_edits
from .editing import (
    TRHashImageEditConfig,
    TRHashImageEditor,
    sample_edit_condition_dropout,
)
from .training import prune_checkpoints, setup_distributed, tokenize_captions

LOGGER = logging.getLogger("tr_hash_image_editor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tr_hash_image_editor_200m.yaml"),
    )
    parser.add_argument("--shards", required=True, help="Local glob for edit TAR shards")
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer/tokenizer.json"))
    parser.add_argument("--vae", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--init-text-to-image",
        type=Path,
        default=None,
        help="Initialize shared weights from a text-to-image checkpoint",
    )
    parser.add_argument(
        "--source-stage",
        choices=(REFINEMENT_STAGE, SUPERVISED_FINETUNING_STAGE),
        default=None,
        help=(
            "Lineage stage of --init-text-to-image. Required for edit SFT; "
            "direct pretraining -> SFT is forbidden."
        ),
    )
    parser.add_argument("--samples-per-epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4, help="Per GPU")
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--save-steps", type=int, default=5_000)
    parser.add_argument("--keep-checkpoints", type=int, default=4)
    parser.add_argument("--save-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if args.resume is not None and args.init_text_to_image is not None:
        parser.error("--resume and --init-text-to-image are mutually exclusive")
    if args.max_steps <= 0 and args.samples_per_epoch <= 0:
        parser.error("set --samples-per-epoch or a positive --max-steps")
    return args


def _base_config(config: TRHashImageEditConfig) -> TRHashImageConfig:
    names = TRHashImageConfig.__dataclass_fields__
    return TRHashImageConfig(**{name: getattr(config, name) for name in names})


def save_edit_checkpoint(
    output: Path,
    model: TRHashImageEditor,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: TRHashImageEditConfig,
    step: int,
    keep_checkpoints: int,
) -> None:
    target = output / f"step_{step:07d}"
    target.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    save_file(state, str(target / "model.safetensors"))
    (target / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        target / "training_state.pt",
    )
    LOGGER.info("Checkpoint saved: %s", target)
    prune_checkpoints(output, keep_checkpoints)


def main() -> None:
    args = parse_args()
    if args.init_text_to_image is not None:
        validate_full_parameter_finetuning(
            IMAGE_EDITING_SUPERVISED_FINETUNING,
            source_stage=args.source_stage,
        )
    rank, local_rank, world_size, device = setup_distributed()
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    config = TRHashImageEditConfig.from_dict(yaml.safe_load(args.config.read_text()))
    shards = sorted(Path(path) for path in glob.glob(args.shards))
    if not shards:
        raise FileNotFoundError(f"no shards match {args.shards!r}")
    dataset = AtlasImageEditTarDataset(
        shards,
        image_size=config.image_size,
        rank=rank,
        world_size=world_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate_atlas_image_edits,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    codec = FrozenAutoencoderKL(args.vae).to(device)
    model = TRHashImageEditor(config).to(device)
    model.gradient_checkpointing = args.gradient_checkpointing
    raw_model = model

    if args.init_text_to_image is not None:
        base_config = TRHashImageConfig.from_dict(
            json.loads((args.init_text_to_image / "config.json").read_text())
        )
        if base_config != _base_config(config):
            raise ValueError("text-to-image checkpoint architecture does not match editor config")
        missing = raw_model.load_text_to_image_state_dict(
            load_file(str(args.init_text_to_image / "model.safetensors"))
        )
        LOGGER.info(
            "Initialized from text-to-image checkpoint; %d edit tensors remain new",
            len(missing),
        )
    elif args.resume is not None:
        resume_config = TRHashImageEditConfig.from_dict(
            json.loads((args.resume / "config.json").read_text())
        )
        if resume_config != config:
            raise ValueError("resume checkpoint config does not match --config")
        raw_model.load_state_dict(load_file(str(args.resume / "model.safetensors")))

    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        global_batch = world_size * args.batch_size * args.gradient_accumulation
        total_steps = math.ceil(args.samples_per_epoch * args.epochs / global_batch)

    def lr_factor(step: int) -> float:
        if step < args.warmup_steps:
            return max(step, 1) / max(args.warmup_steps, 1)
        progress = min(
            (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1),
            1.0,
        )
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
        LOGGER.info(
            "Dataset: %d edit shards; world=%d batch/GPU=%d",
            len(shards),
            world_size,
            args.batch_size,
        )
        LOGGER.info("Training: AdamW lr=%g steps=%d", args.lr, total_steps)
        metrics_path = args.output / "metrics.jsonl"
    else:
        metrics_path = None

    progress = tqdm(
        total=total_steps,
        initial=step,
        desc="TR-Hash image edit",
        unit="step",
        dynamic_ncols=True,
        disable=rank != 0,
    )
    micro_step = 0
    running_loss = 0.0
    interval_samples = 0
    last_loss = float("nan")
    started = time.monotonic()
    interval_started = started
    optimizer.zero_grad(set_to_none=True)
    while step < total_steps:
        for batch in loader:
            source_pixels = batch["source_pixel_values"].to(device, non_blocking=True)
            target_pixels = batch["target_pixel_values"].to(device, non_blocking=True)
            instruction_ids, instruction_mask = tokenize_captions(
                tokenizer,
                batch["instructions"],
                config.max_text_length,
                device,
            )
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=args.bf16 and device.type in {"cuda", "cpu"},
            ):
                source_latents = codec.encode(source_pixels)
                target_latents = codec.encode(target_pixels)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=args.bf16 and device.type in {"cuda", "cpu"},
            ):
                batch_size = target_latents.size(0)
                noise = torch.randn_like(target_latents)
                timesteps = torch.rand(
                    batch_size,
                    device=device,
                    dtype=target_latents.dtype,
                )
                dropped_text, dropped_source = sample_edit_condition_dropout(
                    batch_size,
                    caption_dropout=config.caption_dropout,
                    source_dropout=config.source_dropout,
                    device=device,
                )
                training_ids = instruction_ids.clone()
                training_ids[dropped_text] = 0
                training_mask = instruction_mask.clone()
                training_mask[dropped_text] = False
                training_source = source_latents * (~dropped_source).to(source_latents.dtype).view(
                    batch_size, 1, 1, 1
                )
                view_shape = (batch_size,) + (1,) * (target_latents.ndim - 1)
                interpolation = timesteps.view(view_shape)
                noisy = (1.0 - interpolation) * target_latents + interpolation * noise
                prediction = model(
                    noisy,
                    training_source,
                    timesteps,
                    training_ids,
                    training_mask,
                )
                loss = F.mse_loss(
                    prediction.float(),
                    (noise - target_latents).float(),
                )
                scaled_loss = loss / args.gradient_accumulation
            scaled_loss.backward()
            micro_step += 1
            running_loss += float(loss.detach())
            last_loss = float(loss.detach())
            interval_samples += target_pixels.size(0) * world_size
            if micro_step % args.gradient_accumulation:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            progress.update(1)
            if rank == 0 and step % args.log_steps == 0:
                now = time.monotonic()
                average_loss = running_loss / args.log_steps
                throughput = interval_samples / max(now - interval_started, 1e-6)
                progress.set_postfix(
                    loss=f"{average_loss:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    pair_s=f"{throughput:.1f}",
                    refresh=True,
                )
                with metrics_path.open("a") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "step": step,
                                "total_steps": total_steps,
                                "loss": average_loss,
                                "lr": scheduler.get_last_lr()[0],
                                "pairs_per_second": throughput,
                                "elapsed_seconds": now - started,
                            }
                        )
                        + "\n"
                    )
                running_loss = 0.0
                interval_samples = 0
                interval_started = now
            if rank == 0 and args.save_steps and step % args.save_steps == 0:
                save_edit_checkpoint(
                    args.output,
                    raw_model,
                    optimizer,
                    scheduler,
                    config,
                    step,
                    args.keep_checkpoints,
                )
            if step >= total_steps:
                break

    if rank == 0:
        elapsed = time.monotonic() - started
        summary = {
            "steps": step,
            "last_loss": last_loss,
            "elapsed_seconds": elapsed,
            "world_size": world_size,
            "batch_size_per_gpu": args.batch_size,
        }
        (args.output / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if args.save_final:
            save_edit_checkpoint(
                args.output,
                raw_model,
                optimizer,
                scheduler,
                config,
                step,
                args.keep_checkpoints,
            )
    progress.close()
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
