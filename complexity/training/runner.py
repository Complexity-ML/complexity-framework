"""
TrainRunner — deduplicated training pipeline for FSDP multi-GPU runs.

Replaces the ~350 lines of boilerplate copy-pasted across scripts/train_*.py
with a single class that handles:

- argparse (standard flags; extend via `add_args`)
- distributed init + CUDA backend flags
- tokenizer load + vocab sync
- model construction from a user `make_config` factory
- fixed token-ID routing with no corpus-frequency pre-pass
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
import json
import logging
import math
import os
import signal
import time
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
from ..utils.device import configure_torch_acceleration
from .callbacks import TqdmCallback, WandBCallback
from .config import TrainingConfig
from .packing import resolve_token_pack_schedule
from .trainer import Trainer

logger = logging.getLogger(__name__)


def require_cuda_available(required: bool) -> None:
    """Fail closed when a GPU-only run cannot initialize CUDA.

    ``nvidia-smi`` can report an eGPU shortly before PyTorch is able to create
    a CUDA context, especially during boot.  GPU jobs must not silently fall
    back to CPU in that window.
    """

    if required and not torch.cuda.is_available():
        raise RuntimeError(
            "--require-cuda was set, but PyTorch cannot initialize CUDA; "
            "refusing to fall back to CPU"
        )


def routing_diagnostic(config: ModelConfig) -> dict[str, object]:
    """Describe the effective route family without changing model policy."""

    strategy_value = getattr(config.routing_strategy, "value", config.routing_strategy)
    strategy = str(strategy_value)
    is_multi_hash = strategy == "token_id_multi_hash"
    configured_hashes = int(config.route_hash_count)
    return {
        "strategy": strategy,
        "multi_hash": is_multi_hash,
        "configured_hashes": configured_hashes,
        "effective_hashes": configured_hashes if is_multi_hash else 1,
    }


def log_routing_diagnostic(config: ModelConfig) -> dict[str, object]:
    """Emit an unambiguous startup banner and return its structured values."""

    routing = routing_diagnostic(config)
    logger.info(
        "TR-Hash routing: strategy=%s multi_hash=%s "
        "effective_hashes=%d configured_route_hash_count=%d top_k=%d",
        routing["strategy"],
        str(routing["multi_hash"]).lower(),
        routing["effective_hashes"],
        routing["configured_hashes"],
        config.top_k,
    )
    return routing


def resolve_warmup_steps(
    *,
    max_steps: int,
    tokens_per_step: int,
    warmup_steps: int | None,
    warmup_tokens: int | None,
) -> int:
    """Resolve warmup without making it depend accidentally on GPU count."""

    if warmup_steps is not None and warmup_tokens is not None:
        raise ValueError("--warmup-steps and --warmup-tokens are mutually exclusive")
    if warmup_tokens is not None:
        if warmup_tokens < 0:
            raise ValueError("--warmup-tokens must be non-negative")
        return math.ceil(warmup_tokens / tokens_per_step)
    if warmup_steps is not None:
        return warmup_steps
    return max(1, int(max_steps * 0.05))


class FineWebStreamingDataset(IterableDataset):
    """Streaming tokenized chunks from FineWeb-Edu, sharded per rank."""

    packing_contract = "text-pretraining"

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
            buffer.extend(self.tokenizer.encode(text, add_special_tokens=False))
            if self.tokenizer.eos_token_id is not None:
                buffer.append(self.tokenizer.eos_token_id)
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len:]
                yield {
                    "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                    "labels": torch.tensor(chunk[1:], dtype=torch.long),
                }


class TrainRunner:
    """FSDP multi-GPU training pipeline."""

    token_pack_pipeline = "text-pretraining"

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
        default_distributed_mode: str = "fsdp",
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
        if default_distributed_mode not in {"fsdp", "ddp"}:
            raise ValueError("default_distributed_mode must be 'fsdp' or 'ddp'")
        self.default_distributed_mode = default_distributed_mode
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
            outputs = m(input_ids, return_logits=False)
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
                       help="Optimizer-step warmup. Mutually exclusive with --warmup-tokens.")
        p.add_argument("--warmup-tokens", type=int, default=None,
                       help="Warm up for this many globally trained tokens.")
        p.add_argument("--lr-scheduler", type=str, default="auto",
                       choices=["auto", "cosine", "linear", "constant", "wsd"])
        p.add_argument("--max-steps", type=int, default=None,
                       help="Override max_steps (bypasses --target-tokens)")
        p.add_argument("--save-steps", type=int, default=self.default_save_steps)
        p.add_argument(
            "--token-packs",
            type=int,
            default=1,
            help="Split the token budget into checkpoint packs; this is not an epoch count.",
        )
        p.add_argument("--log-steps", type=int, default=10)
        p.add_argument("--checkpoint-dir", type=str, default=self.checkpoint_dir)
        p.add_argument(
            "--resume",
            type=str,
            default=None,
            help="checkpoint directory to resume from, or 'auto' to resume "
            "from the highest-step checkpoint under --checkpoint-dir if one "
            "exists (silently starts fresh otherwise)",
        )
        p.add_argument(
            "--init-checkpoint",
            type=str,
            default=None,
            help="HF-style export (config.json + model.safetensors, e.g. a "
            "prior run's 'final' checkpoint) to load model weights from, "
            "leaving optimizer/scheduler/step fresh -- for starting a new "
            "training phase from completed weights instead of resuming. "
            "Use with a fresh --checkpoint-dir and --resume auto so "
            "crash-resume within the new phase still works.",
        )
        p.add_argument("--num-workers", type=int, default=4)
        p.add_argument(
            "--distributed-mode",
            choices=["fsdp", "ddp"],
            default=self.default_distributed_mode,
            help="FSDP shards the model; DDP replicates it and is faster for small models.",
        )
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
                       help="Token-Routed deterministic top-K (overrides config.top_k).")
        p.add_argument("--use-custom-kernels", type=str, default="auto",
                       choices=["auto", "true", "false"],
                       help="Custom Triton/CUDA kernels. auto enables NVIDIA CUDA, disables ROCm by default.")
        p.add_argument(
            "--require-cuda",
            action="store_true",
            help="Abort before loading data or model weights if CUDA is unavailable.",
        )
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

        require_cuda_available(args.require_cuda)

        distributed = init_distributed()
        rank = get_rank()
        world_size = get_world_size()
        is_main = is_main_process()

        custom_kernel_policy = (
            True if args.use_custom_kernels == "true"
            else False if args.use_custom_kernels == "false"
            else "auto"
        )
        configure_torch_acceleration(kernel_policy=custom_kernel_policy)

        # Never let a CUDA text run silently lose the fused output projection
        # and CE kernel.  The loss itself repeats this guard at dispatch time
        # for callers that do not use TrainRunner.
        from ..core.losses import log_liger_fused_linear_ce_status

        if is_main:
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            log_liger_fused_linear_ce_status(device_type)

        tokenizer = self._load_tokenizer(args.tokenizer)

        # Model
        config = self.make_config()
        config.vocab_size = min(len(tokenizer), config.vocab_size)
        if args.top_k is not None:
            config.top_k = args.top_k
        config.use_custom_kernels = custom_kernel_policy

        model = ComplexityModel(config)
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        if is_main:
            logger.info(f"Model: {model.num_parameters():,} params "
                        f"({model.num_parameters() / 1e6:.1f}M)")
            logger.info(f"  hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
                        f"heads={config.num_attention_heads}/{config.num_key_value_heads} (GQA), "
                        f"inter={config.intermediate_size}, experts={config.num_experts}")
            log_routing_diagnostic(config)

        # Steps + warmup
        if args.token_packs < 1:
            raise ValueError("--token-packs must be positive")
        tokens_per_step = (args.batch_size * world_size
                           * args.gradient_accumulation * args.seq_len)
        if args.max_steps is not None:
            max_steps = args.max_steps
        else:
            max_steps = math.ceil(args.target_tokens / tokens_per_step)
        warmup_steps = resolve_warmup_steps(
            max_steps=max_steps,
            tokens_per_step=tokens_per_step,
            warmup_steps=args.warmup_steps,
            warmup_tokens=args.warmup_tokens,
        )
        token_pack_schedule = resolve_token_pack_schedule(
            target_tokens=max_steps * tokens_per_step,
            tokens_per_step=tokens_per_step,
            token_packs=args.token_packs,
            pipeline=self.token_pack_pipeline,
        )
        token_pack_steps = max(token_pack_schedule.pack_step_counts)
        automatic_token_pack_saves = args.save_steps <= 0
        # Exact pack boundaries can differ by one step. Disable the periodic
        # saver and let the boundary callback below write every token pack.
        save_steps = max_steps + 1 if automatic_token_pack_saves else args.save_steps
        if is_main:
            logger.info(
                f"Training: {max_steps:,} steps "
                f"(~{max_steps * tokens_per_step / 1e9:.1f}B trained tokens)"
            )
            logger.info(f"  tokens/step: {tokens_per_step:,}")
            logger.info(
                f"  token packs: {args.token_packs}   "
                f"max steps/token pack: {token_pack_steps:,}   "
                f"warmup: {warmup_steps}"
            )

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
            save_steps=save_steps,
            log_steps=args.log_steps,
            checkpoint_dir=args.checkpoint_dir,
            resume_from=args.resume,
            init_checkpoint=args.init_checkpoint,
            use_fsdp=args.distributed_mode == "fsdp",
            sharding_mode="full_shard",
            num_workers=args.num_workers,
        )
        trainer = Trainer(model=model, config=train_config, train_dataloader=dataloader)
        trainer.compute_loss = self.build_compute_loss(trainer, model)

        if automatic_token_pack_saves:
            boundary_to_pack = {
                boundary: pack
                for pack, boundary in enumerate(token_pack_schedule.boundaries, start=1)
            }

            def token_pack_checkpoint_callback(trainer_obj, step, loss_val):
                del loss_val
                pack = boundary_to_pack.get(step)
                if pack is not None:
                    trainer_obj._save_checkpoint(tag=f"token_pack_{pack:03d}")

            # CheckpointManager uses collectives under FSDP, so this callback
            # must run on every rank rather than only on the logging rank.
            trainer.callbacks.append(token_pack_checkpoint_callback)

        # Callbacks — TqdmCallback issues a collective; must be on ALL ranks.
        tqdm_cb = TqdmCallback(
            total_steps=max_steps, desc=self.run_name, tokens_per_step=tokens_per_step
        )
        trainer.callbacks.append(tqdm_cb)

        csv_file: Optional[Any] = None
        csv_writer = None
        if is_main:
            if args.wandb:
                trainer.callbacks.append(WandBCallback(project=args.wandb, name=self.run_name))

            os.makedirs(args.checkpoint_dir, exist_ok=True)
            csv_path = os.path.join(args.checkpoint_dir, "training_log.csv")
            metrics_path = os.path.join(args.checkpoint_dir, "metrics.jsonl")
            file_mode = "a" if args.resume and os.path.exists(csv_path) else "w"
            csv_file = open(csv_path, file_mode, newline="")
            csv_writer = csv.writer(csv_file)
            n_experts = config.num_experts
            if file_mode == "w":
                csv_writer.writerow([
                    "step", "loss", "ppl", "lr", "tokens_trained",
                    *[f"expert_{e}_share" for e in range(n_experts)],
                    "expert_dead_count", "elapsed_s",
                ])
            csv_file.flush()
            if file_mode == "w":
                open(metrics_path, "w", encoding="utf-8").close()

        t_start = time.time()

        def csv_callback(trainer_obj, step, loss_val):
            if not is_main or csv_writer is None:
                return
            ppl = math.exp(min(loss_val, 20))
            lr = trainer_obj.optimizer.param_groups[0]["lr"]
            shares = getattr(tqdm_cb, "last_shares", [float("nan")] * config.num_experts)
            dead = getattr(tqdm_cb, "last_dead", None)
            csv_writer.writerow([
                step, f"{loss_val:.6f}", f"{ppl:.2f}",
                f"{lr:.6e}", step * tokens_per_step,
                *[f"{s:.4f}" for s in shares],
                "nan" if dead is None else dead,
                f"{time.time() - t_start:.1f}",
            ])
            if step == 1 or step % args.log_steps == 0 or step == max_steps:
                csv_file.flush()
                record = {
                    "step": step,
                    "max_steps": max_steps,
                    "loss": round(float(loss_val), 6),
                    "ppl": round(float(ppl), 2),
                    "lr": float(lr),
                    "tokens_trained": step * tokens_per_step,
                    "tokens_per_step": tokens_per_step,
                    "expert_shares": [round(float(share), 4) for share in shares],
                    "expert_dead_count": dead,
                    "elapsed_s": round(time.time() - t_start, 1),
                }
                with open(metrics_path, "a", encoding="utf-8") as metrics_file:
                    metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
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
