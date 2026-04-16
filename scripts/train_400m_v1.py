"""
Pre-training 400M v1 — Token-Routed MLP + Mu-Guidance.

Full architecture v1 at 400M scale for continuous pre-training.
Uses FSDP full_shard for 2× RTX PRO 6000 multi-GPU training.
Optimized for batch_size=128 on 96GB VRAM per GPU.

Model: hidden=1024, layers=20, heads=16, kv_heads=4, inter=2816, 4 experts
       → ~400M params (Token-Routed + Mu)

Usage:
    # 2× RTX PRO 6000
    torchrun --nproc_per_node=2 scripts/train_750m_v1.py

    # Resume
    torchrun --nproc_per_node=2 scripts/train_750m_v1.py --resume checkpoints/400m-v1/step_10000

INL / Complexity-ML — 2026
"""

from complexity.gpu import setup_gpu
setup_gpu()

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from torch.utils.data import DataLoader, IterableDataset
import torch
import torch.nn as nn
import argparse
import os
import math
import logging
import time
import csv

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("train_400m")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

from complexity.config import ModelConfig
from complexity.models import ComplexityModel
from complexity.training import (
    Trainer, TrainingConfig, WandBCallback, TqdmCallback,
    gamma_mean, global_expert_shares,
)
from complexity.parallel import init_distributed, get_rank, get_world_size, is_main_process, cleanup, simple_ddp


# ── Model config (~400M params) ─────────────────────────────────────────

def make_config() -> ModelConfig:
    """Full v1: Token-Routed MLP + full-width Shared + Mu-Guidance + γ-gate.
    hidden=1024, layers=20, heads=16, kv_heads=4, inter=2008, 4 experts
    → ~384.4M, iso-params with train_400m_dense (dense inter=4358).
    Shared expert = full intermediate_size (2008); routed experts = inter/4 (502).
    """
    return ModelConfig(
        hidden_size=1024,
        num_hidden_layers=20,
        num_attention_heads=16,
        num_key_value_heads=4,
        intermediate_size=2008,        # tuned to match dense params exactly
        vocab_size=32000,
        max_position_embeddings=4096,
        attention_type="gqa",
        mlp_type="token_routed",
        num_experts=4,
        shared_expert=True,
        routed_gate=False,             # γ gate removed: stalls at bf16 precision, no benefit
        use_attn_scale=True,           # LayerScale on attn.o_proj output
        attn_scale_init=1.0,           # identity init, learns to re-weight per-channel
        norm_type="rmsnorm",
        use_qk_norm=True,
        use_mu_guidance=True,
    )


# ── Dataset ───────────────────────────────────────────────────────────────

class FineWebStreamingDataset(IterableDataset):
    """Streaming tokenized chunks from FineWeb-Edu.

    Multi-GPU: each rank takes every world_size-th chunk.
    Continuous pre-training: dataset loops forever.
    """

    def __init__(self, tokenizer, max_length=2048, rank=0, world_size=1):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rank = rank
        self.world_size = world_size
        logger.info(f"Connecting to FineWeb-Edu (streaming) [rank {rank}/{world_size}]...")
        t0 = time.time()
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            split="train",
            streaming=True,
        )
        # Shard at document level — each GPU gets its own documents
        if world_size > 1:
            ds = ds.shard(num_shards=world_size, index=rank)
        self.dataset = ds
        logger.info(f"Dataset ready in {time.time() - t0:.1f}s")

    def __iter__(self):
        buffer = []
        first_yield = True
        for example in self.dataset:
            text = example.get("text", "")
            if not text:
                continue
            tokens = self.tokenizer.encode(text)
            buffer.extend(tokens)

            while len(buffer) >= self.max_length + 1:
                chunk = buffer[:self.max_length + 1]
                buffer = buffer[self.max_length:]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                if first_yield:
                    logger.info(f"First batch tokenized [rank {self.rank}]")
                    first_yield = False
                yield {"input_ids": input_ids, "labels": labels}


# ── Training ──────────────────────────────────────────────────────────────

def compute_steps_for_tokens(target_tokens: int, batch_size: int,
                              grad_accum: int, seq_len: int) -> int:
    tokens_per_step = batch_size * grad_accum * seq_len
    return math.ceil(target_tokens / tokens_per_step)


def main():
    parser = argparse.ArgumentParser(description="Train 400M v1 (Token-Routed + Mu)")
    parser.add_argument("--tokenizer", type=str, default="./tokenizer")
    parser.add_argument("--target-tokens", type=int, default=8_000_000_000,
                        help="Target token count (default: 8B)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size per GPU")
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2.1e-4)
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="Warmup steps (default: 5%% of max_steps)")
    parser.add_argument("--lr-scheduler", type=str, default="auto",
                        choices=["auto", "cosine", "linear", "constant"])
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max_steps directly (bypasses --target-tokens calc). "
                             "Use when resuming with different gradient-accumulation.")
    parser.add_argument("--save-steps", type=int, default=5000)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/400m-v1")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--wandb", type=str, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True,
                        help="Gradient checkpointing (default: enabled)")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                        action="store_false")
    args = parser.parse_args()

    # Initialize distributed
    distributed = init_distributed()
    rank = get_rank()
    world_size = get_world_size()
    is_main = is_main_process()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Tokenizer
    if not os.path.exists(args.tokenizer):
        raise FileNotFoundError(
            f"Tokenizer not found: {args.tokenizer}\n"
            f"Train one first or point to an existing HF tokenizer directory."
        )
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)

    # Model
    config = make_config()
    config.vocab_size = min(len(tokenizer), 32000)

    # Zipf-balanced routing
    if config.num_experts > 1 and is_main:
        from itertools import islice
        logger.info("Computing token frequencies for Zipf-balanced routing...")
        freq_dataset = FineWebStreamingDataset(tokenizer=tokenizer, rank=0, world_size=1)
        freq_loader = DataLoader(freq_dataset, batch_size=64, num_workers=2)
        freqs = torch.zeros(config.vocab_size, dtype=torch.float32)
        for batch in islice(freq_loader, 1000):
            ids = batch["input_ids"].flatten()
            ids = ids[ids < config.vocab_size]
            freqs.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))
        config.token_frequencies = freqs
        logger.info(f"  {freqs.sum():.0f} tokens sampled")

    model = ComplexityModel(config)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if is_main:
            logger.info("Gradient checkpointing enabled")

    if is_main:
        logger.info(f"Model: {model.num_parameters():,} params "
                    f"({model.num_parameters()/1e6:.1f}M)")
        logger.info(f"  hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
                    f"heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}")
        logger.info(f"  inter={config.intermediate_size}, experts={config.num_experts}")
        logger.info(f"  attn={config.attention_type}, mlp={config.mlp_type}, "
                    f"mu_guidance={config.use_mu_guidance}")

    # Steps
    if args.max_steps is not None:
        max_steps = args.max_steps
    else:
        max_steps = compute_steps_for_tokens(
            target_tokens=args.target_tokens,
            batch_size=args.batch_size * world_size,
            grad_accum=args.gradient_accumulation,
            seq_len=2048,
        )
    # Auto warmup: 5% of max_steps
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else max(1, int(max_steps * 0.05))

    tokens_per_step = args.batch_size * world_size * args.gradient_accumulation * 2048
    if is_main:
        logger.info(f"Training: {max_steps:,} steps (~{args.target_tokens/1e9:.1f}B tokens)")
        logger.info(f"  Tokens/step: {tokens_per_step:,} "
                    f"(batch={args.batch_size} × {world_size} GPUs × accum={args.gradient_accumulation} × seq=2048)")
        logger.info(f"  Warmup: {warmup_steps} steps (5%)")

    # Dataset
    dataset = FineWebStreamingDataset(
        tokenizer=tokenizer, rank=rank, world_size=world_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Trainer with FSDP
    train_config = TrainingConfig(
        max_steps=max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        optimizer_type="adamw",
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        lr_scheduler=args.lr_scheduler,
        precision="bf16",
        save_steps=args.save_steps,
        log_steps=args.log_steps,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume,
        use_fsdp=True,
        sharding_mode="full_shard",
        num_workers=args.num_workers,
    )

    trainer = Trainer(model=model, config=train_config, train_dataloader=dataloader)

    if is_main:
        logger.info(f"FSDP enabled ({world_size} GPUs, full_shard)")
        logger.info(f"  Device: {trainer.device}")

    # Fused linear + cross-entropy: Liger Triton kernel when available,
    # otherwise pure-PyTorch causal_lm_loss. Supports label_smoothing + z_loss.
    from complexity.core.losses import fused_linear_causal_lm_loss

    def compute_loss(model, batch):
        input_ids = batch["input_ids"].to(trainer.device)
        labels = batch["labels"].to(trainer.device)
        outputs = model(input_ids)
        hidden = outputs["last_hidden_state"] if isinstance(outputs, dict) else outputs
        m = model
        while hasattr(m, "model") or hasattr(m, "module"):
            m = getattr(m, "model", None) or getattr(m, "module", None)
        weight = m.embed_tokens.weight
        shift_hidden = hidden[:, :-1, :].contiguous()
        shift_labels = labels[:, :shift_hidden.size(1)].contiguous()
        loss, _ = fused_linear_causal_lm_loss(
            shift_hidden, weight, shift_labels,
            label_smoothing=0.1,  # matches causal_lm paper recipe
            z_loss_coef=0.0,      # bump to 1e-4 if you want PaLM-style logit stabilization
        )
        return loss

    trainer.compute_loss = compute_loss

    # Logging — tqdm/wandb/CSV writes are rank-0 only, but the expert-share
    # reducer runs on ALL ranks (all_reduce collective).
    csv_file = None
    csv_writer = None
    tqdm_cb = None
    n_experts = config.num_experts
    tokens_per_step_local = tokens_per_step
    t_start = time.time()

    # TqdmCallback runs a distributed collective internally (global_expert_shares)
    # → MUST be registered on ALL ranks. The bar displays only on rank 0.
    tqdm_cb = TqdmCallback(total_steps=max_steps, desc="400M v1")
    trainer.callbacks.append(tqdm_cb)

    if is_main:
        if args.wandb:
            wandb_cb = WandBCallback(project=args.wandb, name="400m-v1")
            trainer.callbacks.append(wandb_cb)

        os.makedirs(args.checkpoint_dir, exist_ok=True)
        csv_path = os.path.join(args.checkpoint_dir, "training_log.csv")
        file_mode = "a" if args.resume and os.path.exists(csv_path) else "w"
        csv_file = open(csv_path, file_mode, newline="")
        csv_writer = csv.writer(csv_file)
        expert_cols = [f"expert_{e}_share" for e in range(n_experts)]
        if file_mode == "w":
            csv_writer.writerow([
                "step", "loss", "ppl", "lr", "gamma_mean", "tokens_seen",
                *expert_cols, "expert_dead_count", "elapsed_s",
            ])
        csv_file.flush()

    def csv_callback(trainer_obj, step, loss_val):
        # Collective — all ranks must call. TqdmCallback already did the
        # reduce this step, so the counters are zero now; we re-read gamma
        # (local, cheap) and rely on TqdmCallback having absorbed the shares.
        # To get shares into the CSV, we do a cheap second all_reduce here
        # (counters were already reset, so this gives zeros → we capture
        # shares from tqdm_cb instead via a shared ref below).
        gamma = gamma_mean(trainer_obj.model)
        if not is_main:
            return
        real_loss = loss_val
        ppl = math.exp(min(real_loss, 20))
        lr = trainer_obj.optimizer.param_groups[0]["lr"]
        tokens_seen = step * tokens_per_step_local
        # Shares come from the TqdmCallback cache set during the same step
        shares = getattr(tqdm_cb, "last_shares", [float("nan")] * n_experts)
        dead = getattr(tqdm_cb, "last_dead", n_experts)
        csv_writer.writerow([
            step, f"{real_loss:.6f}", f"{ppl:.2f}",
            f"{lr:.6e}", f"{gamma:.6f}", tokens_seen,
            *[f"{s:.4f}" for s in shares], dead,
            f"{time.time() - t_start:.1f}",
        ])
        if step % 100 == 0:
            csv_file.flush()
    trainer.callbacks.append(csv_callback)

    logging.getLogger("complexity.training.trainer").setLevel(logging.WARNING)

    if is_main:
        logger.info("Starting training...")
    # SIGTERM handler (torchrun sends SIGTERM on Ctrl+C, not KeyboardInterrupt)
    import signal
    signal.signal(signal.SIGTERM, lambda s, f: (_ for _ in ()).throw(KeyboardInterrupt()))

    summary = None
    try:
        summary = trainer.train()
    except (KeyboardInterrupt, SystemExit):
        pass  # Trainer already saved checkpoint
    finally:
        if tqdm_cb is not None:
            tqdm_cb.close()
        if csv_file is not None:
            csv_file.flush()
            csv_file.close()
        logging.getLogger("complexity.training.trainer").setLevel(logging.INFO)

    if summary is not None and is_main:
        logger.info(f"Training complete: {summary}")
        # Unwrap to base model for correct save (DDP/FSDP wrappers shard weights)
        base = model
        while not hasattr(base, 'save_pretrained'):
            next_base = getattr(base, 'module', None) or getattr(base, 'model', None); base = next_base if (next_base is not None and next_base is not base) else base
            if not hasattr(base, 'save_pretrained'): break
        base.save_pretrained(os.path.join(args.checkpoint_dir, "final"))
        config.save(os.path.join(args.checkpoint_dir, "final", "model_config.yaml"))
        logger.info(f"Model saved to {args.checkpoint_dir}/final/")

    if distributed:
        cleanup()


if __name__ == "__main__":
    from complexity.gpu.distributed_cleanup import safe_main
    safe_main(main)
