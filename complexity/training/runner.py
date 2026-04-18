"""
TrainRunner — deduplicated training pipeline for FSDP multi-GPU runs.

Replaces the ~350 lines of boilerplate copy-pasted across scripts/train_*.py
with a single class that handles:

- argparse (standard flags; extend via `add_args`)
- distributed init + CUDA backend flags
- tokenizer load + vocab sync
- model construction from a user `make_config` factory
- Zipf token-frequency pre-pass (if num_experts > 1)
- dataset (FineWebStreamingDataset by default; override `build_dataset`)
- TrainingConfig + Trainer wiring
- compute_loss (fused Liger CE by default)
- TqdmCallback + optional WandB + CSV writer
- SIGTERM → KeyboardInterrupt trap
- save_pretrained on all ranks (collective) + config dump
- clean shutdown (barrier + cleanup)

Minimal script (30 lines):

    from complexity.config import ModelConfig
    from complexity.training.runner import TrainRunner

    def make_config():
        return ModelConfig(hidden_size=1024, num_hidden_layers=20, ...)

    if __name__ == "__main__":
        TrainRunner(
            make_config=make_config,
            run_name="400m-v1",
            checkpoint_dir="./checkpoints/400m-v1",
            default_lr=2.1e-4,
            default_batch_size=128,
            default_seq_len=2048,
        ).run()

For per-script customization, subclass and override `add_args`, `build_dataset`,
`build_compute_loss`, or `extra_callbacks`.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import signal
import time
from itertools import islice
from pathlib import Path
from typing import Any, Callable, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

from ..config import ModelConfig
from ..models import ComplexityModel
from ..parallel import (
    cleanup,
    get_rank,
    get_world_size,
    init_distributed,
    is_main_process,
)
from .callbacks import TqdmCallback, WandBCallback
from .config import TrainingConfig
from .trainer import Trainer

logger = logging.getLogger(__name__)


class FineWebStreamingDataset(IterableDataset):
    """Streaming tokenized chunks from FineWeb-Edu, sharded per rank."""

    def __init__(self, tokenizer, seq_len: int, rank: int = 0, world_size: int = 1):
        from datasets import load_dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
        if world_size > 1:
            ds = ds.shard(num_shards=world_size, index=rank)
        self.dataset = ds

    def __iter__(self):
        buffer: List[int] = []
        for example in self.dataset:
            text = example.get("text", "")
            if not text:
                continue
            buffer.extend(self.tokenizer.encode(text))
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len:]
                yield {
                    "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                    "labels": torch.tensor(chunk[1:], dtype=torch.long),
                }


class TrainRunner:
    """FSDP multi-GPU training pipeline."""

    def __init__(
        self,
        make_config: Callable[[], ModelConfig],
        run_name: str,
        checkpoint_dir: str,
        default_lr: float = 3e-4,
        default_batch_size: int = 32,
        default_seq_len: int = 2048,
        default_target_tokens: int = 8_000_000_000,
        default_gradient_accumulation: int = 1,
        default_gradient_checkpointing: bool = True,
        default_save_steps: int = 1000,
        optimizer_type: str = "adamw",
        label_smoothing: float = 0.1,
        z_loss: float = 0.0,
    ):
        self.make_config = make_config
        self.run_name = run_name
        self.checkpoint_dir = checkpoint_dir
        self.default_lr = default_lr
        self.default_batch_size = default_batch_size
        self.default_seq_len = default_seq_len
        self.default_target_tokens = default_target_tokens
        self.default_gradient_accumulation = default_gradient_accumulation
        self.default_gradient_checkpointing = default_gradient_checkpointing
        self.default_save_steps = default_save_steps
        self.optimizer_type = optimizer_type
        self.label_smoothing = label_smoothing
        self.z_loss = z_loss

    # ── Hooks: override in subclasses ────────────────────────────────────

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        """Override to add script-specific CLI flags."""
        return None

    def build_dataset(self, tokenizer, args, rank: int, world_size: int) -> IterableDataset:
        """Default: FineWeb-Edu streaming, sharded per rank."""
        return FineWebStreamingDataset(
            tokenizer=tokenizer, seq_len=args.seq_len,
            rank=rank, world_size=world_size,
        )

    def build_compute_loss(self, trainer: Trainer, model: nn.Module) -> Callable:
        """Default: fused linear + cross-entropy (Liger Triton kernel)."""
        from ..core.losses import fused_linear_causal_lm_loss
        label_smoothing = self.label_smoothing
        z_loss = self.z_loss

        def compute_loss(m, batch):
            input_ids = batch["input_ids"].to(trainer.device)
            labels = batch["labels"].to(trainer.device)
            outputs = m(input_ids)
            hidden = outputs["last_hidden_state"] if isinstance(outputs, dict) else outputs
            base = m
            while hasattr(base, "model") or hasattr(base, "module"):
                base = getattr(base, "model", None) or getattr(base, "module", None)
            weight = base.embed_tokens.weight
            shift_hidden = hidden[:, :-1, :].contiguous()
            shift_labels = labels[:, :shift_hidden.size(1)].contiguous()
            loss, _ = fused_linear_causal_lm_loss(
                shift_hidden, weight, shift_labels,
                label_smoothing=label_smoothing, z_loss_coef=z_loss,
            )
            return loss

        return compute_loss

    def extra_callbacks(self, trainer: Trainer, args, is_main: bool) -> List[Callable]:
        """Override to add script-specific callbacks."""
        return []

    # ── Standard pipeline ────────────────────────────────────────────────

    def _build_parser(self) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(description=f"Train {self.run_name}")
        p.add_argument("--tokenizer", type=str, default="./tokenizer")
        p.add_argument("--target-tokens", type=int, default=self.default_target_tokens)
        p.add_argument("--batch-size", type=int, default=self.default_batch_size)
        p.add_argument("--seq-len", type=int, default=self.default_seq_len)
        p.add_argument("--gradient-accumulation", type=int,
                       default=self.default_gradient_accumulation)
        p.add_argument("--lr", type=float, default=self.default_lr)
        p.add_argument("--warmup-steps", type=int, default=None,
                       help="Default: 5%% of max_steps")
        p.add_argument("--lr-scheduler", type=str, default="auto",
                       choices=["auto", "cosine", "linear", "constant", "wsd"])
        p.add_argument("--max-steps", type=int, default=None,
                       help="Override max_steps (bypasses --target-tokens)")
        p.add_argument("--save-steps", type=int, default=self.default_save_steps)
        p.add_argument("--log-steps", type=int, default=10)
        p.add_argument("--checkpoint-dir", type=str, default=self.checkpoint_dir)
        p.add_argument("--resume", type=str, default=None)
        p.add_argument("--num-workers", type=int, default=4)
        p.add_argument("--wandb", type=str, default=None)
        p.add_argument("--precision", type=str, default="bf16",
                       choices=["fp32", "fp16", "bf16"])
        p.add_argument("--gradient-checkpointing", action="store_true",
                       default=self.default_gradient_checkpointing)
        p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                       action="store_false")
        p.add_argument("--optimizer", type=str, default=self.optimizer_type,
                       choices=["adamw", "adamw_mup", "muon", "muon_tr", "adam_tr"],
                       help="Overrides the script's default optimizer (e.g. adam_tr for MoE ablations)")
        p.add_argument("--top-k", type=int, default=None,
                       help="Token-Routed top-K deterministic (overrides config.top_k). K=1 classic Zipf.")
        p.add_argument("--top-k-primary-weight", type=float, default=None,
                       help="Weight of primary expert in top-K combo (0.5=uniform, 1.0=degenerate top-1). Default 0.75.")
        self.add_args(p)
        return p

    def _load_tokenizer(self, path: str):
        from transformers import PreTrainedTokenizerFast
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Tokenizer not found: {path}\n"
                f"Train one first or point --tokenizer to an existing HF directory."
            )
        return PreTrainedTokenizerFast.from_pretrained(path)

    def _compute_zipf_frequencies(self, tokenizer, vocab_size: int) -> torch.Tensor:
        """One-pass token frequency count for Zipf-balanced routing init."""
        logger.info("Computing token frequencies for Zipf-balanced routing...")
        freq_dataset = FineWebStreamingDataset(
            tokenizer=tokenizer, seq_len=512, rank=0, world_size=1,
        )
        freq_loader = DataLoader(freq_dataset, batch_size=64, num_workers=2)
        freqs = torch.zeros(vocab_size, dtype=torch.float32)
        for batch in islice(freq_loader, 1000):
            ids = batch["input_ids"].flatten()
            ids = ids[ids < vocab_size]
            freqs.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))
        logger.info(f"  {freqs.sum():.0f} tokens sampled")
        return freqs

    def run(self) -> None:
        args = self._build_parser().parse_args()

        # Without basicConfig the root logger defaults to WARNING, swallowing
        # every logger.info() in the framework — including the TRAINING CONFIG
        # banner, "Saved checkpoint: ...", and FSDP init messages. Set once
        # here so users see what's happening.
        if not logging.getLogger().handlers:
            logging.basicConfig(
                format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
                level=logging.INFO,
            )
        for lib in ("httpx", "httpcore", "huggingface_hub", "datasets", "transformers"):
            logging.getLogger(lib).setLevel(logging.WARNING)

        distributed = init_distributed()
        rank = get_rank()
        world_size = get_world_size()
        is_main = is_main_process()

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        tokenizer = self._load_tokenizer(args.tokenizer)

        # Model
        config = self.make_config()
        config.vocab_size = min(len(tokenizer), config.vocab_size)
        if args.top_k is not None:
            config.top_k = args.top_k
        if args.top_k_primary_weight is not None:
            config.top_k_primary_weight = args.top_k_primary_weight

        if config.num_experts > 1 and is_main:
            config.token_frequencies = self._compute_zipf_frequencies(
                tokenizer, config.vocab_size,
            )

        model = ComplexityModel(config)
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        if is_main:
            logger.info(f"Model: {model.num_parameters():,} params "
                        f"({model.num_parameters() / 1e6:.1f}M)")
            logger.info(f"  hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
                        f"heads={config.num_attention_heads}/{config.num_key_value_heads} (GQA), "
                        f"inter={config.intermediate_size}, experts={config.num_experts}")

        # Steps + warmup
        if args.max_steps is not None:
            max_steps = args.max_steps
        else:
            tokens_per_step = (args.batch_size * world_size
                               * args.gradient_accumulation * args.seq_len)
            max_steps = math.ceil(args.target_tokens / tokens_per_step)
        warmup_steps = (args.warmup_steps if args.warmup_steps is not None
                        else max(1, int(max_steps * 0.05)))
        tokens_per_step = (args.batch_size * world_size
                           * args.gradient_accumulation * args.seq_len)
        if is_main:
            logger.info(f"Training: {max_steps:,} steps "
                        f"(~{max_steps * tokens_per_step / 1e9:.1f}B tokens)")
            logger.info(f"  tokens/step: {tokens_per_step:,}   warmup: {warmup_steps}")

        # Dataset + loader
        dataset = self.build_dataset(tokenizer, args, rank, world_size)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        # Trainer
        train_config = TrainingConfig(
            max_steps=max_steps,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            optimizer_type=args.optimizer,
            learning_rate=args.lr,
            warmup_steps=warmup_steps,
            lr_scheduler=args.lr_scheduler,
            precision=args.precision,
            save_steps=args.save_steps,
            log_steps=args.log_steps,
            checkpoint_dir=args.checkpoint_dir,
            resume_from=args.resume,
            use_fsdp=True,
            sharding_mode="full_shard",
            num_workers=args.num_workers,
        )
        trainer = Trainer(model=model, config=train_config, train_dataloader=dataloader)
        trainer.compute_loss = self.build_compute_loss(trainer, model)

        # Callbacks — TqdmCallback issues a collective; must be on ALL ranks.
        tqdm_cb = TqdmCallback(total_steps=max_steps, desc=self.run_name)
        trainer.callbacks.append(tqdm_cb)

        csv_file: Optional[Any] = None
        csv_writer = None
        if is_main:
            if args.wandb:
                trainer.callbacks.append(WandBCallback(project=args.wandb, name=self.run_name))

            os.makedirs(args.checkpoint_dir, exist_ok=True)
            csv_path = os.path.join(args.checkpoint_dir, "training_log.csv")
            file_mode = "a" if args.resume and os.path.exists(csv_path) else "w"
            csv_file = open(csv_path, file_mode, newline="")
            csv_writer = csv.writer(csv_file)
            n_experts = config.num_experts
            if file_mode == "w":
                csv_writer.writerow([
                    "step", "loss", "ppl", "lr", "tokens_seen",
                    *[f"expert_{e}_share" for e in range(n_experts)],
                    "expert_dead_count", "elapsed_s",
                ])
            csv_file.flush()

        t_start = time.time()

        def csv_callback(trainer_obj, step, loss_val):
            if not is_main or csv_writer is None:
                return
            ppl = math.exp(min(loss_val, 20))
            lr = trainer_obj.optimizer.param_groups[0]["lr"]
            shares = getattr(tqdm_cb, "last_shares", [float("nan")] * config.num_experts)
            dead = getattr(tqdm_cb, "last_dead", config.num_experts)
            csv_writer.writerow([
                step, f"{loss_val:.6f}", f"{ppl:.2f}",
                f"{lr:.6e}", step * tokens_per_step,
                *[f"{s:.4f}" for s in shares], dead,
                f"{time.time() - t_start:.1f}",
            ])
            if step % 100 == 0:
                csv_file.flush()
        trainer.callbacks.append(csv_callback)

        for cb in self.extra_callbacks(trainer, args, is_main):
            trainer.callbacks.append(cb)

        # SIGTERM → KeyboardInterrupt (torchrun sends SIGTERM on Ctrl+C).
        signal.signal(signal.SIGTERM, lambda s, f: (_ for _ in ()).throw(KeyboardInterrupt()))

        summary = None
        try:
            summary = trainer.train()
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            tqdm_cb.close()
            if csv_file is not None:
                csv_file.flush()
                csv_file.close()

        if is_main and summary is not None:
            logger.info(f"Training complete: {summary}")

        # save_pretrained must run on ALL ranks (internal full_tensor() collective).
        if summary is not None:
            base = model
            while not hasattr(base, "save_pretrained"):
                nxt = getattr(base, "module", None) or getattr(base, "model", None)
                if nxt is None or nxt is base:
                    break
                base = nxt
            if hasattr(base, "save_pretrained"):
                final_dir = os.path.join(args.checkpoint_dir, "final")
                base.save_pretrained(final_dir)
                if is_main:
                    config.save(os.path.join(final_dir, "model_config.yaml"))
                    logger.info(f"Model saved to {final_dir}")

        if distributed:
            import torch.distributed as dist
            dist.barrier()
            cleanup()
