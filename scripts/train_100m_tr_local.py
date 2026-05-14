"""
Local ~100M residual Token-Routed run.

Runs on CUDA, MPS, or CPU with the same CLI/log schema as the 300M scaling
script. This is the long-horizon 100M research run: Mu-Guidance is disabled by
default, the shared SwiGLU path carries the dense-compatible trunk, and a small
deterministic Token-Routed residual branch provides specialization.

Examples:
    python3 scripts/train_100m_tr_local.py --steps 20 --dataset random
    python3 scripts/train_100m_tr_local.py --dataset text --text-file data/sample.txt --tokenizer ./tokenizer --bf16
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import shutil
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
from complexity.training import global_expert_shares
from complexity.utils import autocast, autocast_dtype, empty_cache, setup_mps, synchronize


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("train_100m_tr_local")
logging.getLogger("complexity.core.mlp.token_routed").setLevel(logging.WARNING)
for noisy_logger in ("httpx", "httpcore", "huggingface_hub", "datasets"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def make_config(args) -> ModelConfig:
    return ModelConfig(
        hidden_size=640,
        num_hidden_layers=10,
        num_attention_heads=10,
        num_key_value_heads=2,
        intermediate_size=args.intermediate_size,
        vocab_size=args.vocab_size,
        max_position_embeddings=2048,
        attention_type="gqa",
        mlp_type="token_routed",
        num_experts=4,
        shared_expert=True,
        shared_intermediate_size=args.shared_intermediate_size,
        norm_type="rmsnorm",
        use_qk_norm=True,
        use_mu_guidance=args.use_mu_guidance,
        use_shared_routed_gates=args.learn_shared_routed_gates,
        shared_gate_init=args.shared_gate_init,
        routed_gate_init=args.routed_gate_init,
        top_k=args.top_k,
        top_k_primary_weight=args.top_k_primary_weight,
        clamp_mu_contextual=args.mu_clamp,
        use_mu_norm=args.mu_norm,
        mu_alpha_init=args.mu_alpha_init,
        mu_init_value=args.mu_init_value,
        mu_context_min=args.mu_context_min,
        mu_context_max=args.mu_context_max,
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
            buffer.extend(self.tokenizer.encode(text, add_special_tokens=False))
            if self.tokenizer.eos_token_id is not None:
                buffer.append(self.tokenizer.eos_token_id)
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


def infer_vocab_size(args) -> int:
    if args.vocab_size is not None:
        return args.vocab_size
    if args.dataset == "random":
        return 32000
    return Tokenizer.load(args.tokenizer).vocab_size


def text_token_frequencies(path: str, tokenizer_path: str, vocab_size: int) -> torch.Tensor:
    tokens = load_text_tokens(path, tokenizer_path)
    ids = torch.tensor(tokens, dtype=torch.long)
    ids = ids[(ids >= 0) & (ids < vocab_size)]
    freqs = torch.zeros(vocab_size, dtype=torch.float32)
    if ids.numel() > 0:
        freqs.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))
    logger.info(
        f"Zipf routing frequencies: {int(freqs.sum().item()):,} tokens, "
        f"{int((freqs > 0).sum().item()):,} vocab entries"
    )
    return freqs


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


def resolve_checkpoint_path(path: str) -> Path:
    ckpt = Path(path)
    if ckpt.name == "latest":
        parent = ckpt.parent
        candidates = sorted(parent.glob("step_*"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoints found in {parent}")
        return candidates[-1]
    return ckpt


def save_checkpoint(args, raw_model, optimizer, scheduler, config, step: int, is_main: bool, distributed: bool):
    if distributed:
        dist.barrier()
    if not is_main or args.save_steps <= 0:
        if distributed:
            dist.barrier()
        return

    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir = save_root / f"step_{step:06d}"
    tmp_dir = save_root / f".step_{step:06d}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    torch.save(
        {
            "step": step,
            "model": {k: v.detach().cpu() for k, v in raw_model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config.to_dict(),
            "args": vars(args),
        },
        tmp_dir / "checkpoint.pt",
    )
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    tmp_dir.rename(ckpt_dir)

    checkpoints = sorted(save_root.glob("step_*"))
    excess = len(checkpoints) - max(1, args.save_total_limit)
    for old in checkpoints[:max(0, excess)]:
        shutil.rmtree(old)
    logger.info(f"Checkpoint saved: {ckpt_dir}")
    if distributed:
        dist.barrier()


def load_checkpoint(path: str, raw_model, optimizer, scheduler, device, is_main: bool) -> int:
    ckpt_dir = resolve_checkpoint_path(path)
    ckpt_file = ckpt_dir / "checkpoint.pt"
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_file}")
    state = torch.load(ckpt_file, map_location=device)
    raw_model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    step = int(state["step"])
    if is_main:
        logger.info(f"Resumed from {ckpt_dir} at step {step}")
    return step


def main():
    parser = argparse.ArgumentParser(description="Local ~100M residual Token-Routed run")
    parser.add_argument("--dataset", choices=["random", "text", "fineweb"], default="random")
    parser.add_argument("--text-file", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="./tokenizer")
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    # Iso-param residual TR profile: no Mu in the strict dense comparison, so
    # parameter budget goes to a dense-compatible shared SwiGLU trunk plus a
    # small deterministic Token-Routed residual path. This keeps the model at
    # ~99.1M params like dense while preserving TR specialization capacity.
    parser.add_argument("--intermediate-size", type=int, default=256)
    parser.add_argument("--shared-intermediate-size", type=int, default=3328)
    parser.add_argument("--shared-gate-init", type=float, default=1.0)
    parser.add_argument("--routed-gate-init", type=float, default=0.1)
    parser.add_argument("--learn-shared-routed-gates", dest="learn_shared_routed_gates", action="store_true", default=True)
    parser.add_argument("--no-learn-shared-routed-gates", dest="learn_shared_routed_gates", action="store_false")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--top-k-primary-weight", type=float, default=0.5)
    parser.add_argument("--use-mu-guidance", action="store_true")
    parser.add_argument("--mu-clamp", action="store_true")
    parser.add_argument("--mu-norm", action="store_true")
    parser.add_argument("--mu-alpha-init", type=float, default=1.0)
    parser.add_argument("--mu-init-value", type=float, default=0.0)
    parser.add_argument("--mu-context-min", type=float, default=-2.0)
    parser.add_argument("--mu-context-max", type=float, default=2.0)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--grad-ckpt", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--empty-cache-every", type=int, default=50)
    parser.add_argument("--run-name", type=str, default="100m-tr-local")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--z-loss", type=float, default=0.0)
    parser.add_argument("--save-steps", type=int, default=5000)
    parser.add_argument("--save-dir", type=str, default="checkpoints/100m-tr-local")
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--no-zipf-from-text",
        action="store_true",
        help="Disable token-frequency balanced routing when --dataset text.",
    )
    args = parser.parse_args()

    device, distributed, rank, local_rank, world_size = init_distributed(args.seed)
    is_main = rank == 0
    args.vocab_size = infer_vocab_size(args)
    config = make_config(args)
    if args.dataset == "text" and not args.no_zipf_from_text:
        config.token_frequencies = text_token_frequencies(
            args.text_file,
            args.tokenizer,
            config.vocab_size,
        )
    raw_model = ComplexityModel(config).to(device)
    if args.grad_ckpt:
        raw_model.gradient_checkpointing_enable()

    params = raw_model.num_parameters()
    if is_main:
        logger.info(f"Model: {params / 1e6:.1f}M params")
        logger.info(
            "Config: Token-Routed residual, hidden=640, layers=10, GQA=10/2, "
            f"inter={args.intermediate_size}, shared_inter={args.shared_intermediate_size}, "
            f"experts=4, top_k={args.top_k}, primary_w={args.top_k_primary_weight}, "
            f"learn_gates={args.learn_shared_routed_gates}, "
            f"gates=({args.shared_gate_init},{args.routed_gate_init}), "
            f"use_mu={args.use_mu_guidance}, mu_clamp={args.mu_clamp}, mu_norm={args.mu_norm}, "
            f"mu_alpha={args.mu_alpha_init}, mu_init={args.mu_init_value}"
        )
        if distributed:
            logger.info(f"DDP: world_size={world_size}, per_gpu_batch={args.batch_size}")

    model = raw_model
    if distributed:
        model = DDP(
            raw_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

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
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, raw_model, optimizer, scheduler, device, is_main)
    if distributed:
        dist.barrier()

    run_dir = Path("runs") / args.run_name
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    csv_path = run_dir / "metrics.csv"
    csv_file = None
    writer = None
    if is_main:
        csv_mode = "a" if args.resume and csv_path.exists() else "w"
        csv_file = csv_path.open(csv_mode, newline="")
        writer = csv.writer(csv_file)
        if csv_mode == "w":
            writer.writerow([
                "step", "train_loss", "train_ppl", "eval_loss", "eval_ppl", "lr", "tok_s",
                "expert_0_share", "expert_1_share", "expert_2_share", "expert_3_share",
                "expert_dead_count",
            ])
        csv_file.flush()

    model.train()
    pbar = (
        tqdm(total=args.steps, initial=start_step, desc="100M TR", unit="step", dynamic_ncols=True)
        if is_main else None
    )
    t_log = time.perf_counter()
    tokens_since_log = 0
    last_step = start_step

    for step, batch in enumerate(train_loader, start=start_step + 1):
        if step > args.steps:
            break
        last_step = step
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
            shares, dead = global_expert_shares(raw_model, config.num_experts)
            if not shares:
                shares = [float("nan")] * config.num_experts
            if is_main:
                writer.writerow([
                    step, f"{train_loss:.6f}", f"{train_ppl:.2f}",
                    f"{eval_loss:.6f}", f"{eval_ppl:.2f}",
                    f"{lr_now:.6e}", f"{tok_s:.0f}",
                    *[f"{s:.4f}" for s in shares], dead,
                ])
                csv_file.flush()
                pbar.set_postfix(loss=f"{train_loss:.4f}", eval=f"{eval_loss:.4f}", tok_s=f"{tok_s:.0f}")
            t_log = now
            tokens_since_log = 0

        if args.empty_cache_every > 0 and step % args.empty_cache_every == 0:
            empty_cache(device)

        if args.save_steps > 0 and step % args.save_steps == 0:
            save_checkpoint(args, raw_model, optimizer, scheduler, config, step, is_main, distributed)

    if args.save_steps > 0 and last_step > start_step and last_step % args.save_steps != 0:
        save_checkpoint(args, raw_model, optimizer, scheduler, config, last_step, is_main, distributed)

    if pbar is not None:
        pbar.close()
    if csv_file is not None:
        csv_file.close()
        logger.info(f"Metrics saved: {csv_path}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
