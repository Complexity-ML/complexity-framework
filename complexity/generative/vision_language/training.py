"""Distributed training runtime for TR-Hash image+text-to-text models."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import shutil
import time
from pathlib import Path
from typing import Sequence

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from complexity.training.finetuning import (
    IMAGE_TEXT_TO_TEXT_SUPERVISED_FINETUNING,
    REFINEMENT_STAGE,
    SUPERVISED_FINETUNING_STAGE,
    validate_full_parameter_finetuning,
)

from .config import TRHashVisionLanguageConfig
from .data import DEFAULT_PROMPT, VisionLanguageTarDataset, collate_vision_language
from .model import TRHashImageTextToText

LOGGER = logging.getLogger("tr_hash_image_text_to_text")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Optional TRHashVisionLanguageConfig YAML/JSON")
    parser.add_argument("--shards", required=True, help="Local glob for stage-1 (alignment) TAR shards")
    parser.add_argument(
        "--sft-shards",
        default=None,
        help=(
            "Optional glob for a curated image-grounded conversation shard set "
            "(questions, answers, corrections, comparisons, multi-turn). "
            "Inspired by TR-HASH Vision's noisy-pretrain -> clean full-parameter "
            "SFT recipe: the final --sft-steps steps train on this set instead "
            "of --shards, per the two-stage recipe in "
            "docs/tr-hash-image-text-to-text.md. Requires --sft-steps > 0."
        ),
    )
    parser.add_argument(
        "--sft-steps",
        type=int,
        default=0,
        help="Final N steps trained on --sft-shards instead of --shards. 0 disables the SFT stage.",
    )
    parser.add_argument(
        "--sft-lr-scale",
        type=float,
        default=1.0,
        help="LR multiplier applied during the --sft-steps stage (typically < 1 for a clean SFT).",
    )
    parser.add_argument(
        "--source-stage",
        choices=(REFINEMENT_STAGE, SUPERVISED_FINETUNING_STAGE),
        default=None,
        help=(
            "Lineage stage of the checkpoint entering --sft-shards. Required "
            "for SFT; direct alignment/pretraining -> SFT is forbidden."
        ),
    )
    parser.add_argument("--default-prompt", default=DEFAULT_PROMPT, help="Prompt used when a shard has none")
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer/tokenizer.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint directory")
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4, help="Per GPU")
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-text-length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--save-steps", type=int, default=5_000)
    parser.add_argument("--keep-checkpoints", type=int, default=4)
    parser.add_argument("--save-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
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


def tokenize_prompt_response(
    tokenizer: Tokenizer,
    prompts: Sequence[str],
    responses: Sequence[str],
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate prompt + response tokens; mask prompt and padding in labels.

    Mirrors the contract in docs/tr-hash-image-text-to-text.md: the language-model
    loss applies only to answer positions.
    """

    encoded_prompts = tokenizer.encode_batch(list(prompts))
    encoded_responses = tokenizer.encode_batch(list(responses))
    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        pad_id = 0
    eos_id = tokenizer.token_to_id("</s>")

    sequences: list[list[int]] = []
    prompt_lengths: list[int] = []
    for prompt_item, response_item in zip(encoded_prompts, encoded_responses):
        response_ids = list(response_item.ids)
        if eos_id is not None:
            response_ids = response_ids + [int(eos_id)]
        combined = (list(prompt_item.ids) + response_ids)[:max_length]
        if not combined:
            combined = [int(pad_id)]
        sequences.append(combined)
        prompt_lengths.append(min(len(prompt_item.ids), len(combined)))

    width = max((len(seq) for seq in sequences), default=1)
    input_ids = torch.full((len(sequences), width), int(pad_id), dtype=torch.long)
    labels = torch.full((len(sequences), width), -100, dtype=torch.long)
    for index, (seq, prompt_length) in enumerate(zip(sequences, prompt_lengths)):
        ids = torch.tensor(seq, dtype=torch.long)
        input_ids[index, : len(seq)] = ids
        if prompt_length < len(seq):
            labels[index, prompt_length : len(seq)] = ids[prompt_length:]
    return input_ids.to(device, non_blocking=True), labels.to(device, non_blocking=True)


def stage_is_sft(step: int, *, total_steps: int, sft_steps: int) -> bool:
    """True once training has entered the final ``sft_steps`` steps of the run."""

    if sft_steps <= 0:
        return False
    return step >= total_steps - sft_steps


def freeze_decoder_for_vision_only_sft(model: TRHashImageTextToText) -> dict[str, int]:
    """Freeze the language decoder; leave the vision tower, resampler, and
    visual projection trainable.

    The curated stage-2 SFT corpus is image-grounded QA/dialogue -- a
    language-instruction shape the framework restricts to LoRA-only for text
    models (complexity/training/finetuning.py). Freezing the decoder outright
    keeps this pipeline compliant without a LoRA path: the exemption covers
    full-parameter *vision* adaptation only, and the language decoder is
    never trained during this stage, LoRA or otherwise.
    """

    model.decoder.requires_grad_(False)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {"trainable": trainable, "total": total, "frozen": total - trainable}


def save_checkpoint(
    output: Path,
    model: TRHashImageTextToText,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: TRHashVisionLanguageConfig,
    step: int,
    keep_checkpoints: int,
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
    prune_checkpoints(output, keep_checkpoints)


def prune_checkpoints(output: Path, keep_checkpoints: int) -> None:
    """Keep only the newest complete step directories."""

    if keep_checkpoints <= 0:
        raise ValueError("keep_checkpoints must be positive")
    checkpoints = sorted(
        (
            path
            for path in output.glob("step_*")
            if path.is_dir() and path.name.removeprefix("step_").isdigit()
        ),
        key=lambda path: int(path.name.removeprefix("step_")),
    )
    for stale in checkpoints[:-keep_checkpoints]:
        shutil.rmtree(stale)
        LOGGER.info("Pruned checkpoint: %s", stale)


def _build_loader(
    shards_glob: str,
    *,
    config: TRHashVisionLanguageConfig,
    default_prompt: str,
    rank: int,
    world_size: int,
    batch_size: int,
    workers: int,
    device_type: str,
) -> tuple[DataLoader, list[Path]]:
    shard_paths = sorted(Path(path) for path in glob.glob(shards_glob))
    if not shard_paths:
        raise FileNotFoundError(f"no shards match {shards_glob!r}")
    dataset = VisionLanguageTarDataset(
        shard_paths,
        image_size=config.image_size,
        rank=rank,
        world_size=world_size,
        default_prompt=default_prompt,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=collate_vision_language,
        pin_memory=device_type == "cuda",
        persistent_workers=workers > 0,
    )
    return loader, shard_paths


def main() -> None:
    args = parse_args()
    if args.sft_steps < 0:
        raise ValueError("--sft-steps must be non-negative")
    if args.sft_steps > 0 and not args.sft_shards:
        raise ValueError("--sft-steps requires --sft-shards")
    if args.sft_shards and args.sft_steps <= 0:
        raise ValueError("--sft-shards requires --sft-steps > 0")
    if args.sft_lr_scale <= 0.0:
        raise ValueError("--sft-lr-scale must be positive")
    if args.sft_steps > args.max_steps:
        raise ValueError("--sft-steps cannot exceed --max-steps")
    if args.sft_shards:
        validate_full_parameter_finetuning(
            IMAGE_TEXT_TO_TEXT_SUPERVISED_FINETUNING,
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

    if args.config is not None:
        values = json.loads(args.config.read_text())
        config = TRHashVisionLanguageConfig.from_dict(values)
    else:
        config = TRHashVisionLanguageConfig()

    loader, shard_paths = _build_loader(
        args.shards,
        config=config,
        default_prompt=args.default_prompt,
        rank=rank,
        world_size=world_size,
        batch_size=args.batch_size,
        workers=args.workers,
        device_type=device.type,
    )
    sft_loader = None
    sft_shard_paths: list[Path] = []
    if args.sft_shards:
        sft_loader, sft_shard_paths = _build_loader(
            args.sft_shards,
            config=config,
            default_prompt=args.default_prompt,
            rank=rank,
            world_size=world_size,
            batch_size=args.batch_size,
            workers=args.workers,
            device_type=device.type,
        )

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    model = TRHashImageTextToText(config).to(device)
    raw_model = model
    if args.resume is not None:
        resume_config = TRHashVisionLanguageConfig.from_dict(
            json.loads((args.resume / "config.json").read_text())
        )
        if resume_config != config:
            raise ValueError("resume checkpoint config does not match --config")
        raw_model.load_state_dict(load_file(str(args.resume / "model.safetensors")))
    if world_size > 1:
        # find_unused_parameters is required once --sft-shards is set: the
        # decoder's requires_grad flips to False mid-run when the SFT stage
        # starts, and DDP's default gradient-bucket reduction assumes a fixed
        # set of grad-producing parameters for the whole run.
        model = DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=bool(args.sft_shards)
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )
    total_steps = args.max_steps

    def lr_factor(step: int) -> float:
        if step < args.warmup_steps:
            base = max(step, 1) / max(args.warmup_steps, 1)
        else:
            progress = min((step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1), 1.0)
            base = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
        if stage_is_sft(step, total_steps=total_steps, sft_steps=args.sft_steps):
            base *= args.sft_lr_scale
        return base

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
        LOGGER.info("Stage 1: %d shards; world=%d batch/GPU=%d", len(shard_paths), world_size, args.batch_size)
        if sft_loader is not None:
            LOGGER.info(
                "Stage 2 (SFT): %d shards, final %d steps @ lr_scale=%g",
                len(sft_shard_paths),
                args.sft_steps,
                args.sft_lr_scale,
            )
        LOGGER.info("Training: AdamW lr=%g steps=%d", args.lr, total_steps)
        metrics_path = args.output / "metrics.jsonl"
    else:
        metrics_path = None

    progress = tqdm(
        total=total_steps,
        initial=step,
        desc="TR-Hash VLM",
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
    pretrain_iterator = iter(loader)
    sft_iterator = iter(sft_loader) if sft_loader is not None else None
    entered_sft_stage = False
    # Iterable shard assignments can contain different numbers of samples per
    # DDP rank. Cycle each local loader until the shared optimizer-step target
    # is reached; bounding by local iterator passes can strand shorter ranks at
    # a barrier while longer ranks are still reducing gradients.
    while step < total_steps:
        in_sft_stage = stage_is_sft(step, total_steps=total_steps, sft_steps=args.sft_steps)
        if in_sft_stage and not entered_sft_stage:
            entered_sft_stage = True
            freeze_stats = freeze_decoder_for_vision_only_sft(raw_model)
            if rank == 0:
                LOGGER.info(
                    "Entering vision-only SFT stage at step=%d (%d steps remaining): "
                    "decoder frozen, %.1fM/%.1fM params trainable (vision tower/resampler/projection)",
                    step,
                    args.sft_steps,
                    freeze_stats["trainable"] / 1e6,
                    freeze_stats["total"] / 1e6,
                )
        if in_sft_stage:
            try:
                batch = next(sft_iterator)
            except StopIteration:
                sft_iterator = iter(sft_loader)
                batch = next(sft_iterator)
        else:
            try:
                batch = next(pretrain_iterator)
            except StopIteration:
                pretrain_iterator = iter(loader)
                batch = next(pretrain_iterator)

        pixels = batch["pixel_values"].to(device, non_blocking=True)
        input_ids, labels = tokenize_prompt_response(
            tokenizer, batch["prompts"], batch["responses"], args.max_text_length, device
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=args.bf16 and device.type in {"cuda", "cpu"},
        ):
            output = model(pixels, input_ids, labels=labels)
            loss = output["loss"]
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
                ex_s=f"{throughput:.1f}",
                stage="sft" if in_sft_stage else "pretrain",
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
                            "examples_per_second": throughput,
                            "elapsed_seconds": elapsed,
                            "stage": "sft" if in_sft_stage else "pretrain",
                        }
                    )
                    + "\n"
                )
            running_loss = 0.0
        if rank == 0 and args.save_steps and step % args.save_steps == 0:
            save_checkpoint(
                args.output,
                raw_model,
                optimizer,
                scheduler,
                config,
                step,
                args.keep_checkpoints,
            )

    if rank == 0:
        elapsed = time.monotonic() - started
        summary = {
            "steps": step,
            "last_loss": last_loss,
            "elapsed_seconds": elapsed,
            "examples_seen": running_samples,
            "examples_per_second": running_samples / max(elapsed, 1e-6),
            "world_size": world_size,
            "batch_size_per_gpu": args.batch_size,
        }
        (args.output / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        LOGGER.info(
            "Training complete: step=%d loss=%.5f elapsed=%.1fs examples/s=%.1f",
            step,
            last_loss,
            elapsed,
            summary["examples_per_second"],
        )
        if args.save_final:
            save_checkpoint(
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
