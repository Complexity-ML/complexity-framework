"""
Local ~300M dense SwiGLU baseline.

Runs on CUDA, MPS, or CPU with a tiny local loop and CSV logging. Use this as
the dense counterpart to train_300m_tr_local.py.

Examples:
    python3 scripts/train_300m_dense_local.py --steps 20 --dataset random
    python3 scripts/train_300m_dense_local.py --dataset text --text-file data/sample.txt --tokenizer ./tokenizer --bf16
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from complexity.config import ModelConfig
from complexity.core.losses import causal_lm_loss
from complexity.models import ComplexityModel
from complexity.tokenizer import Tokenizer
from complexity.utils import autocast, autocast_dtype, empty_cache, setup_mps, synchronize


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("train_300m_dense_local")
for noisy_logger in ("httpx", "httpcore", "huggingface_hub", "datasets"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def make_config() -> ModelConfig:
    return ModelConfig(
        hidden_size=1024,
        num_hidden_layers=18,
        num_attention_heads=16,
        num_key_value_heads=4,
        intermediate_size=4096,
        vocab_size=32000,
        max_position_embeddings=2048,
        attention_type="gqa",
        mlp_type="swiglu",
        num_experts=1,
        shared_expert=False,
        norm_type="rmsnorm",
        use_qk_norm=True,
        use_mu_guidance=False,
    )


class RandomTokenDataset(IterableDataset):
    def __init__(self, vocab_size: int, seq_len: int, seed: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.seed = seed

    def __iter__(self):
        gen = torch.Generator().manual_seed(self.seed)
        while True:
            ids = torch.randint(0, self.vocab_size, (self.seq_len + 1,), generator=gen)
            yield {"input_ids": ids[:-1], "labels": ids[1:]}


class LocalTextDataset(IterableDataset):
    def __init__(self, tokens: list[int], seq_len: int, seed: int):
        if len(tokens) < seq_len + 2:
            raise ValueError(f"Need at least {seq_len + 2} tokens, got {len(tokens)}")
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.seq_len = seq_len
        self.seed = seed

    def __iter__(self):
        gen = torch.Generator().manual_seed(self.seed)
        high = self.tokens.numel() - self.seq_len - 1
        while True:
            start = torch.randint(0, high + 1, (1,), generator=gen).item()
            chunk = self.tokens[start : start + self.seq_len + 1]
            yield {"input_ids": chunk[:-1], "labels": chunk[1:]}


class FineWebDataset(IterableDataset):
    def __init__(self, tokenizer, seq_len: int, rank: int, world_size: int):
        from datasets import load_dataset

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )

    def __iter__(self):
        buffer: list[int] = []
        for idx, example in enumerate(self.dataset):
            if idx % self.world_size != self.rank:
                continue
            text = example.get("text", "")
            if not text:
                continue
            buffer.extend(self.tokenizer.encode(text))
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len :]
                yield {
                    "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                    "labels": torch.tensor(chunk[1:], dtype=torch.long),
                }


def load_text_tokens(path: str, tokenizer_path: str) -> list[int]:
    tokenizer = Tokenizer.load(tokenizer_path)
    text = Path(path).read_text(encoding="utf-8")
    tokens = tokenizer.encode(text)
    logger.info(f"Text dataset: {path} ({len(tokens):,} tokens)")
    return tokens


def split_tokens(tokens: list[int], eval_ratio: float) -> tuple[list[int], list[int]]:
    n_eval = max(2048, int(len(tokens) * eval_ratio))
    n_eval = min(n_eval, max(1, len(tokens) // 5))
    return tokens[:-n_eval], tokens[-n_eval:]


@torch.no_grad()
def evaluate(model, raw_model, loader, device, amp_dtype, eval_batches, label_smoothing, z_loss, distributed):
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
            logits = F.linear(outputs["last_hidden_state"], raw_model.embed_tokens.weight)
            _, metrics = causal_lm_loss(
                logits,
                labels,
                label_smoothing=label_smoothing,
                z_loss_coef=z_loss,
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


def build_loaders(args, config, rank: int, world_size: int):
    if args.dataset == "fineweb":
        tokenizer = Tokenizer.load(args.tokenizer)
        if rank == 0:
            logger.info("Dataset: FineWeb-Edu sample-10BT streaming")
        train_ds = FineWebDataset(tokenizer, args.seq_len, rank, world_size)
        eval_ds = FineWebDataset(tokenizer, args.seq_len, rank, world_size) if args.eval_steps > 0 else None
    elif args.dataset == "text":
        if not args.text_file:
            raise ValueError("--text-file is required when --dataset text")
        tokens = load_text_tokens(args.text_file, args.tokenizer)
        train_tokens, eval_tokens = split_tokens(tokens, args.eval_ratio)
        train_ds = LocalTextDataset(train_tokens, args.seq_len, args.seed + rank)
        eval_ds = LocalTextDataset(eval_tokens, args.seq_len, args.seed + 10_000 + rank)
    else:
        train_ds = RandomTokenDataset(config.vocab_size, args.seq_len, args.seed + rank)
        eval_ds = RandomTokenDataset(config.vocab_size, args.seq_len, args.seed + 10_000 + rank)

    loader_kwargs = {"batch_size": args.batch_size, "pin_memory": False}
    if args.num_workers > 0:
        loader_kwargs.update(num_workers=args.num_workers, persistent_workers=True)
    eval_loader = DataLoader(eval_ds, **loader_kwargs) if eval_ds is not None else None
    return DataLoader(train_ds, **loader_kwargs), eval_loader


def init_distributed(seed: int):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA. Run single-process for CPU/MPS.")
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


def main():
    parser = argparse.ArgumentParser(description="Local ~300M dense baseline")
    parser.add_argument("--dataset", choices=["random", "text", "fineweb"], default="random")
    parser.add_argument("--text-file", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="./tokenizer")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--grad-ckpt", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--empty-cache-every", type=int, default=50)
    parser.add_argument("--run-name", type=str, default="300m-dense-local")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--z-loss", type=float, default=0.0)
    args = parser.parse_args()

    device, distributed, rank, local_rank, world_size = init_distributed(args.seed)
    is_main = rank == 0
    config = make_config()
    raw_model = ComplexityModel(config).to(device)
    if args.grad_ckpt:
        raw_model.gradient_checkpointing_enable()

    params = raw_model.num_parameters()
    if is_main:
        logger.info(f"Model: {params / 1e6:.1f}M params")
        logger.info("Config: dense SwiGLU, hidden=1024, layers=18, GQA=16/4, inter=4096")
        if distributed:
            logger.info(f"DDP: world_size={world_size}, per_gpu_batch={args.batch_size}")

    model = raw_model
    if distributed:
        model = DDP(raw_model, device_ids=[local_rank], output_device=local_rank)

    amp_dtype = autocast_dtype(device) if args.bf16 else None
    train_loader, eval_loader = build_loaders(args, config, rank, world_size)

    decay, no_decay = [], []
    for name, p in raw_model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "bias" in name or "norm" in name else decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    warmup = max(1, int(args.steps * 0.05))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, args.steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    run_dir = Path("runs") / args.run_name
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    csv_path = run_dir / "metrics.csv"
    csv_file = None
    writer = None
    if is_main:
        csv_file = csv_path.open("w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["step", "train_loss", "train_ppl", "eval_loss", "eval_ppl", "lr", "tok_s"])
        csv_file.flush()

    model.train()
    pbar = tqdm(total=args.steps, desc="300M dense", unit="step", dynamic_ncols=True) if is_main else None
    t_log = time.perf_counter()
    tokens_since_log = 0

    for step, batch in enumerate(train_loader, start=1):
        if step > args.steps:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, dtype=amp_dtype, enabled=amp_dtype is not None):
            outputs = model(input_ids, return_logits=False)
            logits = F.linear(outputs["last_hidden_state"], raw_model.embed_tokens.weight)
            loss, metrics = causal_lm_loss(
                logits,
                labels,
                label_smoothing=args.label_smoothing,
                z_loss_coef=args.z_loss,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        tokens_since_log += args.batch_size * args.seq_len * world_size
        if pbar is not None:
            pbar.update(1)

        should_eval = args.eval_steps > 0 and step % args.eval_steps == 0
        should_log = step == 1 or step % args.log_steps == 0 or should_eval
        if should_log:
            synchronize(device)
            now = time.perf_counter()
            tok_s = tokens_since_log / max(1e-9, now - t_log)
            eval_loss = float("nan")
            if should_eval and eval_loader is not None:
                eval_loss = evaluate(
                    model, raw_model, eval_loader, device, amp_dtype, args.eval_batches,
                    args.label_smoothing, args.z_loss, distributed,
                )
            train_loss = reduce_average(metrics.ce, device, distributed)
            train_ppl = math.exp(min(train_loss, 20))
            eval_ppl = math.exp(min(eval_loss, 20)) if math.isfinite(eval_loss) else float("nan")
            lr_now = scheduler.get_last_lr()[0]
            if is_main:
                writer.writerow([
                    step, f"{train_loss:.6f}", f"{train_ppl:.2f}",
                    f"{eval_loss:.6f}", f"{eval_ppl:.2f}",
                    f"{lr_now:.6e}", f"{tok_s:.0f}",
                ])
                csv_file.flush()
                pbar.set_postfix(loss=f"{train_loss:.4f}", eval=f"{eval_loss:.4f}", tok_s=f"{tok_s:.0f}")
            t_log = now
            tokens_since_log = 0

        if args.empty_cache_every > 0 and step % args.empty_cache_every == 0:
            empty_cache(device)

    if pbar is not None:
        pbar.close()
    if csv_file is not None:
        csv_file.close()
        logger.info(f"Metrics saved: {csv_path}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
