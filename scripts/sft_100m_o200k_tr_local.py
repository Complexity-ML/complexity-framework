#!/usr/bin/env python3
"""SFT runner for local o200k Token-Routed checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from complexity.config import ModelConfig
from complexity.core.losses import causal_lm_loss_from_hidden
from complexity.models import ComplexityModel
from complexity.tokenizer import Tokenizer
from complexity.training.o200k_pretrain import init_distributed
from complexity.utils import autocast, autocast_dtype, empty_cache, setup_mps, synchronize
from complexity.utils.device import backend_metadata, configure_torch_acceleration
from complexity.utils.local_checkpoint import save_local_checkpoint


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
for noisy_logger in ("httpx", "httpcore", "huggingface_hub", "datasets"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


TOY_RECORDS = [
    {
        "messages": [
            {"role": "user", "content": "Explique Token-Routed MLP en une phrase."},
            {
                "role": "assistant",
                "content": "Token-Routed MLP envoie chaque token vers des experts fixes tout en gardant un chemin partagé dense.",
            },
        ]
    },
    {
        "instruction": "Donne une réponse courte.",
        "input": "Pourquoi masquer le prompt en SFT ?",
        "output": "Pour apprendre seulement la réponse assistant, pas recopier l'instruction.",
    },
    {
        "prompt": "User:\nQuel est le but du SFT ?\n\nAssistant:\n",
        "completion": "Adapter le modèle à un style de réponse utile sans refaire tout le pré-entraînement.",
    },
]


def load_checkpoint_state(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[Path, dict[str, Any]]:
    ckpt = Path(path)
    if ckpt.is_file():
        return ckpt.parent, torch.load(ckpt, map_location=map_location)
    ckpt_file = ckpt / "checkpoint.pt"
    if ckpt_file.exists():
        return ckpt, torch.load(ckpt_file, map_location=map_location)
    latest = ckpt / "latest"
    if latest.exists():
        target = latest.read_text(encoding="utf-8").strip()
        if target:
            return load_checkpoint_state(ckpt / target, map_location=map_location)
    raise FileNotFoundError(f"No checkpoint.pt found under {ckpt}")


def checkpoint_config(state: dict[str, Any]) -> ModelConfig:
    if "config" not in state:
        raise KeyError("Checkpoint does not contain a 'config' entry")
    return ModelConfig.from_dict(state["config"])


def format_record(record: dict[str, Any]) -> tuple[str, str]:
    if "messages" in record:
        messages = record["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        assistant_idx = None
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "assistant":
                assistant_idx = idx
                break
        if assistant_idx is None:
            raise ValueError("messages record has no assistant message")

        prompt_parts: list[str] = []
        for msg in messages[:assistant_idx]:
            role = str(msg.get("role", "user")).strip().title()
            content = str(msg.get("content", "")).strip()
            if content:
                prompt_parts.append(f"{role}:\n{content}")
        prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:\n"
        completion = str(messages[assistant_idx].get("content", "")).strip()
        return prompt, completion

    if "instruction" in record or "output" in record:
        instruction = str(record.get("instruction", "")).strip()
        extra_input = str(record.get("input", "")).strip()
        output = str(record.get("output", record.get("response", ""))).strip()
        user = instruction if not extra_input else f"{instruction}\n\n{extra_input}"
        return f"User:\n{user}\n\nAssistant:\n", output

    if "prompt" in record and ("completion" in record or "response" in record):
        return str(record["prompt"]), str(record.get("completion", record.get("response", "")))

    raise ValueError("Supported JSONL formats: messages, instruction/output, or prompt/completion")


def encode_sft_example(
    tokenizer: Tokenizer,
    record: dict[str, Any],
    seq_len: int,
    min_completion_tokens: int,
) -> dict[str, torch.Tensor]:
    prompt, completion = format_record(record)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        completion_ids = completion_ids + [eos_id]

    if not completion_ids:
        completion_ids = [eos_id if eos_id is not None else 0]

    max_prompt = max(1, seq_len + 1 - max(min_completion_tokens, 1))
    if len(prompt_ids) > max_prompt:
        prompt_ids = prompt_ids[-max_prompt:]

    max_completion = max(1, seq_len + 1 - len(prompt_ids))
    completion_ids = completion_ids[:max_completion]
    full = prompt_ids + completion_ids
    if len(full) < 2:
        full = full + [eos_id if eos_id is not None else 0]
    full = full[: seq_len + 1]

    input_ids = full[:-1]
    labels = full[1:]
    prompt_targets = max(0, min(len(labels), len(prompt_ids) - 1))
    labels[:prompt_targets] = [-100] * prompt_targets

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_id if eos_id is not None else 0
    pad = seq_len - len(input_ids)
    if pad > 0:
        input_ids = input_ids + [pad_id] * pad
        labels = labels + [-100] * pad

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class SFTJsonlDataset(IterableDataset):
    def __init__(
        self,
        path: str | None,
        tokenizer_path: str,
        seq_len: int,
        seed: int,
        rank: int,
        world_size: int,
        min_completion_tokens: int = 32,
    ):
        self.records = load_records(path)
        self.tokenizer_path = tokenizer_path
        self.seq_len = seq_len
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.min_completion_tokens = min_completion_tokens

    def __iter__(self):
        tokenizer = Tokenizer.load(self.tokenizer_path)
        rng = random.Random(self.seed)
        records = list(self.records)
        while True:
            rng.shuffle(records)
            for idx, record in enumerate(records):
                if idx % self.world_size != self.rank:
                    continue
                yield encode_sft_example(tokenizer, record, self.seq_len, self.min_completion_tokens)


def load_records(path: str | None) -> list[dict[str, Any]]:
    if path is None:
        return list(TOY_RECORDS)
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    if not records:
        raise ValueError(f"No SFT records found in {path}")
    return records


def build_optimizer(args, raw_model):
    decay, no_decay = [], []
    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "bias" in name or "norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )


def label_stats(labels: torch.Tensor, vocab_size: int) -> dict[str, int]:
    valid = labels != -100
    if not valid.any():
        return {
            "supervised_tokens": 0,
            "min_label": -1,
            "max_label": -1,
            "bad_labels": 0,
        }
    valid_labels = labels[valid]
    bad = (valid_labels < 0) | (valid_labels >= vocab_size)
    return {
        "supervised_tokens": int(valid.sum().item()),
        "min_label": int(valid_labels.min().item()),
        "max_label": int(valid_labels.max().item()),
        "bad_labels": int(bad.sum().item()),
    }


def sft_loss_from_hidden(
    hidden_states: torch.Tensor,
    output_weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_tokens: int,
) -> torch.Tensor:
    """Chunked CE for SFT, computing logits/CE in fp32 for stability."""

    flat_hidden = hidden_states.reshape(-1, hidden_states.size(-1))
    flat_labels = labels.reshape(-1)
    valid = flat_labels != -100
    denom = valid.sum().clamp_min(1).to(dtype=torch.float32)
    total = flat_hidden.new_zeros((), dtype=torch.float32)
    chunk = max(1, int(chunk_tokens or flat_hidden.size(0)))
    for start in range(0, flat_hidden.size(0), chunk):
        end = min(start + chunk, flat_hidden.size(0))
        labels_chunk = flat_labels[start:end]
        valid_chunk = labels_chunk != -100
        if not valid_chunk.any():
            continue
        hidden_chunk = flat_hidden[start:end][valid_chunk].float()
        labels_chunk = labels_chunk[valid_chunk]
        logits = F.linear(hidden_chunk, output_weight.float())
        total = total + F.cross_entropy(
            logits,
            labels_chunk,
            reduction="sum",
        )
    return total / denom


def save_checkpoint(args, raw_model, optimizer, scheduler, config, source_checkpoint: str, step: int, is_main: bool, distributed: bool):
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
            "sft_source_checkpoint": source_checkpoint,
            "backend": backend_metadata(kernel_policy=getattr(args, "use_custom_kernels", "auto")),
        },
    )
    logger.info(f"Checkpoint saved: {ckpt_dir}")
    if distributed:
        dist.barrier()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SFT local o200k Token-Routed checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint dir or checkpoint.pt to fine-tune")
    parser.add_argument("--tokenizer", default="./tokenizer-o200k")
    parser.add_argument("--jsonl", default=None, help="SFT JSONL. If omitted, uses a tiny toy dataset.")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--min-completion-tokens", type=int, default=32)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--use-custom-kernels",
        choices=["auto", "true", "false"],
        default="auto",
        help="Custom Triton/CUDA kernels. auto enables NVIDIA CUDA, disables ROCm by default.",
    )
    parser.add_argument("--grad-ckpt", action="store_true")
    parser.add_argument("--loss-chunk-tokens", type=int, default=1024)
    parser.add_argument(
        "--sft-fp32-loss",
        action="store_true",
        default=True,
        help="Compute the tied o200k SFT loss in fp32 chunks for stability.",
    )
    parser.add_argument(
        "--no-sft-fp32-loss",
        dest="sft_fp32_loss",
        action="store_false",
        help="Use the generic causal_lm_loss_from_hidden path.",
    )
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-dir", default="checkpoints/sft-100m-o200k-tr")
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--run-name", default="sft-100m-o200k-tr")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--empty-cache-every", type=int, default=50)
    parser.add_argument("--cpu", action="store_true", help="Force CPU for smoke tests")
    return parser


def main():
    args = build_parser().parse_args()
    if args.cpu:
        device = torch.device("cpu")
        distributed = False
        rank = local_rank = 0
        world_size = 1
        torch.manual_seed(args.seed)
    else:
        device, distributed, rank, local_rank, world_size = init_distributed(args.seed)
    is_main = rank == 0
    kernel_policy = (
        True if args.use_custom_kernels == "true"
        else False if args.use_custom_kernels == "false"
        else "auto"
    )
    args.use_custom_kernels = kernel_policy
    configure_torch_acceleration(kernel_policy=kernel_policy, log=is_main)

    ckpt_dir, state = load_checkpoint_state(args.checkpoint, map_location="cpu")
    config = checkpoint_config(state)
    config.use_custom_kernels = kernel_policy
    raw_model = ComplexityModel(config).to(device)
    missing, unexpected = raw_model.load_state_dict(state["model"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    if args.grad_ckpt:
        raw_model.gradient_checkpointing_enable()

    model = raw_model
    if distributed:
        model = DDP(
            raw_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    train_ds = SFTJsonlDataset(
        args.jsonl,
        args.tokenizer,
        args.seq_len,
        args.seed + rank,
        rank,
        world_size,
        args.min_completion_tokens,
    )
    loader_kwargs = {"batch_size": args.batch_size, "pin_memory": False}
    if args.num_workers > 0:
        loader_kwargs.update(num_workers=args.num_workers, persistent_workers=True)
    train_loader = DataLoader(train_ds, **loader_kwargs)

    optimizer = build_optimizer(args, raw_model)
    warmup = max(1, int(args.steps * args.warmup_ratio))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, args.steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    amp_dtype = autocast_dtype(device) if args.bf16 else None

    run_dir = Path("runs") / args.run_name
    csv_file = None
    writer = None
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"SFT source: {ckpt_dir} (pretrain step={state.get('step', 'unknown')})")
        logger.info(f"Model: {raw_model.num_parameters() / 1e6:.1f}M params")
        backend = backend_metadata(kernel_policy=kernel_policy)
        logger.info(
            "Backend: "
            f"{backend['backend']} device={backend['device_name']} "
            f"matmul={backend['matmul']} distributed={backend['distributed']} "
            f"sdpa={backend['sdpa']} flash={backend['flash_attention']} "
            f"custom_triton={backend['custom_triton']}"
        )
        logger.info(
            f"Config: vocab={config.vocab_size}, hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
            f"GQA={config.num_attention_heads}/{config.num_key_value_heads}, "
            f"TR experts={config.num_experts}, top_k={config.top_k}, use_mu={config.use_mu_guidance}"
        )
        if args.jsonl is None:
            logger.info("Dataset: built-in toy SFT records")
        else:
            logger.info(f"Dataset: {args.jsonl} ({len(train_ds.records)} records)")
        csv_file = (run_dir / "metrics.csv").open("w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow([
            "step", "train_loss", "train_ppl", "lr", "tok_s",
            "supervised_tokens", "min_label", "max_label", "bad_labels",
        ])
        csv_file.flush()

    model.train()
    pbar = tqdm(total=args.steps, desc="SFT o200k TR", unit="step", dynamic_ncols=True) if is_main else None
    t_log = time.perf_counter()
    tokens_since_log = 0
    last_step = 0

    for step, batch in enumerate(train_loader, start=1):
        if step > args.steps:
            break
        last_step = step
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        stats = label_stats(labels, config.vocab_size)
        if stats["supervised_tokens"] == 0:
            continue
        if stats["bad_labels"] > 0:
            raise ValueError(
                "SFT batch has labels outside model vocab: "
                f"min={stats['min_label']} max={stats['max_label']} "
                f"vocab={config.vocab_size} bad={stats['bad_labels']}"
            )
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, dtype=amp_dtype, enabled=amp_dtype is not None):
            outputs = model(input_ids, return_logits=False)
            if args.sft_fp32_loss:
                loss = sft_loss_from_hidden(
                    outputs["last_hidden_state"],
                    raw_model.embed_tokens.weight,
                    labels,
                    chunk_tokens=args.loss_chunk_tokens,
                )
            else:
                loss, metrics = causal_lm_loss_from_hidden(
                    outputs["last_hidden_state"],
                    raw_model.embed_tokens.weight,
                    labels,
                    chunk_tokens=args.loss_chunk_tokens,
                )
        if args.sft_fp32_loss:
            metrics_ce = float(loss.detach().item())
        else:
            metrics_ce = float(metrics.ce)
        if not math.isfinite(metrics_ce):
            raise FloatingPointError(
                "Non-finite SFT loss before backward: "
                f"loss={metrics_ce} supervised_tokens={stats['supervised_tokens']} "
                f"min_label={stats['min_label']} max_label={stats['max_label']} "
                f"vocab={config.vocab_size}"
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        tokens_since_log += args.batch_size * args.seq_len * world_size
        if pbar is not None:
            pbar.update(1)

        should_log = step == 1 or step % args.log_steps == 0
        if should_log:
            synchronize(device)
            now = time.perf_counter()
            tok_s = tokens_since_log / max(1e-9, now - t_log)
            train_loss = metrics_ce
            if distributed:
                loss_tensor = torch.tensor(train_loss, device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                train_loss = float(loss_tensor.item())
            train_ppl = math.exp(min(train_loss, 20))
            lr_now = scheduler.get_last_lr()[0]
            if is_main:
                writer.writerow([
                    step, f"{train_loss:.6f}", f"{train_ppl:.2f}", f"{lr_now:.6e}", f"{tok_s:.0f}",
                    stats["supervised_tokens"], stats["min_label"], stats["max_label"], stats["bad_labels"],
                ])
                csv_file.flush()
                pbar.set_postfix(loss=f"{train_loss:.4f}", tok_s=f"{tok_s:.0f}")
            t_log = now
            tokens_since_log = 0

        if args.empty_cache_every > 0 and step % args.empty_cache_every == 0:
            empty_cache(device)
        if args.save_steps > 0 and step % args.save_steps == 0:
            save_checkpoint(args, raw_model, optimizer, scheduler, config, str(ckpt_dir), step, is_main, distributed)

    if args.save_steps > 0 and last_step > 0 and last_step % args.save_steps != 0:
        save_checkpoint(args, raw_model, optimizer, scheduler, config, str(ckpt_dir), last_step, is_main, distributed)
    if pbar is not None:
        pbar.close()
    if csv_file is not None:
        csv_file.close()
        logger.info(f"Metrics saved: {run_dir / 'metrics.csv'}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
