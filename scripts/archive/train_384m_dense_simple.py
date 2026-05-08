"""
Dense SwiGLU baseline — iso-params A/B counterpart to train_70m_mps.py.

Same backbone (hidden=640, 10 layers, GQA 10/2, RMSNorm, QK-norm), but :
  - mlp_type="swiglu" (no MoE, no shared expert)
  - intermediate_size=3840 → matches ~105M total params of the MoE recipe

This lets you compare loss curves at IDENTICAL param budget to see whether
the MoE + shared-expert + α-gate recipe actually beats a pure dense model.

CSV metrics written to runs/{--run-name}/metrics.csv (same schema as MoE).

Usage:
    python scripts/train_384m_dense_mps.py --steps 1000 --bf16 --dataset fineweb

Complexity-ML — 2026
"""

import argparse
import csv
import logging
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from complexity.config import ModelConfig
from complexity.core.losses import causal_lm_loss
from complexity.models import ComplexityModel
from complexity.tokenizer import Tokenizer
from complexity.utils import (
    autocast,
    autocast_dtype,
    empty_cache,
    mps_memory_stats,
    setup_mps,
    synchronize,
)


class FineWebDataset(IterableDataset):
    """FineWeb-Edu streaming — tokenized chunks, next-token prediction."""

    def __init__(self, tokenizer, seq_len: int = 512):
        from datasets import load_dataset
        self.tokenizer = tokenizer
        self.seq_len   = seq_len
        self.dataset   = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )

    def __iter__(self):
        buffer = []
        for example in self.dataset:
            text = example.get("text", "")
            if not text:
                continue
            buffer.extend(self.tokenizer.encode(text))
            while len(buffer) >= self.seq_len + 1:
                chunk  = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len :]
                yield {
                    "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                    "labels":    torch.tensor(chunk[1:],  dtype=torch.long),
                }


class RandomTokenDataset(IterableDataset):
    def __init__(self, vocab_size: int = 32000, seq_len: int = 512):
        self.vocab_size = vocab_size
        self.seq_len    = seq_len

    def __iter__(self):
        while True:
            ids = torch.randint(0, self.vocab_size, (self.seq_len + 1,))
            yield {"input_ids": ids[:-1], "labels": ids[1:]}

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("train_384m_dense")

for lib in ("httpx", "httpcore", "huggingface_hub", "datasets", "transformers"):
    logging.getLogger(lib).setLevel(logging.WARNING)


def make_config() -> ModelConfig:
    # intermediate_size=4358 tuned to match ~384M (iso-params with MoE)
    return ModelConfig(
        hidden_size=1024,
        num_hidden_layers=20,
        num_attention_heads=16,
        num_key_value_heads=4,
        vocab_size=32000,
        max_position_embeddings=4096,
        attention_type="gqa",
        mlp_type="swiglu",
        intermediate_size=4358,
        num_experts=1,
        shared_expert=False,
        norm_type="rmsnorm",
        use_qk_norm=True,
        use_mu_guidance=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Dense baseline on MPS/CPU")
    parser.add_argument("--tokenizer",   type=str,   default="./tokenizer")
    parser.add_argument("--dataset",     type=str,   default="fineweb",
                        help="fineweb | random")
    parser.add_argument("--steps",       type=int,   default=1000)
    parser.add_argument("--batch-size",  type=int,   default=8)
    parser.add_argument("--seq-len",     type=int,   default=512)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--log-steps",   type=int,   default=10)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--bf16",        action="store_true")
    parser.add_argument("--grad-ckpt",   action="store_true")
    parser.add_argument("--num-workers", type=int,   default=2)
    parser.add_argument("--empty-cache-every", type=int, default=50)
    parser.add_argument("--run-name",    type=str,   default="dense")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--lr-schedule", type=str, default="cosine",
                        choices=["cosine", "wsd"])
    parser.add_argument("--z-loss",      type=float, default=0.0)
    args = parser.parse_args()

    device = setup_mps(unlimited_watermark=True, cpu_fallback=True, seed=args.seed)

    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "metrics.csv"
    csv_file = csv_path.open("w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "loss", "ppl", "lr", "tok_s", "alpha_mean"])
    csv_file.flush()
    logger.info(f"CSV: {csv_path}")

    config = make_config()
    model = ComplexityModel(config).to(device)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing: enabled")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {total_params / 1e6:.1f}M params (dense baseline)")
    logger.info(
        f"Config: hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
        f"heads={config.num_attention_heads}/{config.num_key_value_heads} (GQA), "
        f"mlp={config.mlp_type}, inter={config.intermediate_size}"
    )

    amp_dtype = autocast_dtype(device) if args.bf16 else None
    if amp_dtype is not None:
        logger.info(f"Autocast: {amp_dtype}")

    # Optimizer — GPT-3 style
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "bias" in name:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": 0.1},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    warmup    = max(1, int(args.steps * 0.05))
    min_ratio = 0.1

    if args.lr_schedule == "cosine":
        def lr_lambda(step):
            if step < warmup:
                return step / warmup
            progress = (step - warmup) / max(1, args.steps - warmup)
            return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    else:  # wsd: warmup → stable (until 75%) → 1-sqrt decay
        decay_start = int(args.steps * 0.75)
        def lr_lambda(step):
            if step < warmup:
                return step / warmup
            if step < decay_start:
                return 1.0
            p = (step - decay_start) / max(1, args.steps - decay_start)
            return min_ratio + (1.0 - min_ratio) * (1.0 - math.sqrt(p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if args.dataset == "fineweb":
        tokenizer = Tokenizer.load(args.tokenizer)
        logger.info(f"Tokenizer: {args.tokenizer} (vocab={tokenizer.vocab_size})")
        dataset = FineWebDataset(tokenizer, seq_len=args.seq_len)
        logger.info("Dataset: FineWeb-Edu sample-10BT (streaming)")
    else:
        dataset = RandomTokenDataset(vocab_size=config.vocab_size, seq_len=args.seq_len)
        logger.info("Dataset: random tokens (dev mode)")

    loader_kwargs = dict(batch_size=args.batch_size, pin_memory=False)
    if args.num_workers > 0:
        loader_kwargs.update(num_workers=args.num_workers, persistent_workers=True)
    loader = DataLoader(dataset, **loader_kwargs)

    model.train()
    t_start          = time.perf_counter()
    t_log            = t_start
    tokens_since_log = 0

    logger.info(f"Training {args.steps} steps | batch={args.batch_size} | seq={args.seq_len}")
    pbar = tqdm(total=args.steps, desc="train 384M dense", unit="step", dynamic_ncols=True)

    try:
        for step, batch in enumerate(loader):
            if step >= args.steps:
                break

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels    = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device, dtype=amp_dtype, enabled=amp_dtype is not None):
                outputs = model(input_ids)
                hidden  = outputs["last_hidden_state"]
                logits  = F.linear(hidden, model.embed_tokens.weight)
                loss, loss_metrics = causal_lm_loss(
                    logits, labels,
                    label_smoothing=args.label_smoothing,
                    z_loss_coef=args.z_loss,
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            tokens_since_log += args.batch_size * args.seq_len
            pbar.update(1)

            if (step + 1) % args.log_steps == 0 or step == 0:
                synchronize(device)
                now       = time.perf_counter()
                dt        = now - t_log
                tok_s     = tokens_since_log / dt if dt > 0 else 0
                last_loss = loss.item()
                ppl       = math.exp(min(loss_metrics.ce, 20))
                lr_now    = scheduler.get_last_lr()[0]

                csv_writer.writerow([
                    step + 1, f"{last_loss:.6f}", f"{ppl:.2f}",
                    f"{lr_now:.6e}", f"{tok_s:.0f}", "nan",  # no α in dense
                ])
                csv_file.flush()

                postfix = dict(
                    loss=f"{last_loss:.4f}",
                    ppl=f"{ppl:.1f}",
                    lr=f"{lr_now:.2e}",
                    tok_s=f"{tok_s:,.0f}",
                )
                stats = mps_memory_stats()
                if stats is not None:
                    postfix["mem"] = f"{stats.driver_allocated_mb:.0f}/{stats.recommended_max_mb:.0f}MB"
                pbar.set_postfix(postfix)
                t_log            = now
                tokens_since_log = 0

            if args.empty_cache_every > 0 and (step + 1) % args.empty_cache_every == 0:
                empty_cache(device)

    except KeyboardInterrupt:
        logger.info(f"Interrupted at step {step + 1}")

    pbar.close()

    synchronize(device)
    elapsed      = time.perf_counter() - t_start
    total_tokens = args.steps * args.batch_size * args.seq_len
    logger.info(
        f"Done — {args.steps} steps in {elapsed:.1f}s | "
        f"{total_tokens / elapsed:,.0f} tok/s overall"
    )
    stats = mps_memory_stats()
    if stats is not None:
        logger.info(str(stats))

    csv_file.close()
    logger.info(f"Metrics saved: {csv_path}")


if __name__ == "__main__":
    main()
