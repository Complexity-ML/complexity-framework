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
from complexity.training import Trainer, TrainingConfig, WandBCallback, TqdmCallback
from complexity.parallel import init_distributed, get_rank, get_world_size, is_main_process, cleanup, simple_ddp


# ── Model config (~400M params) ─────────────────────────────────────────

def make_config() -> ModelConfig:
    """Full v1: Token-Routed MLP + Mu-Guidance + INL Dynamics.
    hidden=1024, layers=20, heads=16, kv_heads=4, inter=3200, 4 experts
    shared=expert_inter=800 → ~384M (run2 architecture scaled up).
    """
    return ModelConfig(
        hidden_size=1024,
        num_hidden_layers=20,
        num_attention_heads=16,
        num_key_value_heads=4,
        intermediate_size=3200,
        vocab_size=32000,
        max_position_embeddings=4096,
        attention_type="gqa",
        mlp_type="token_routed",
        num_experts=4,
        shared_expert=True,
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

    if is_main:
        tokens_per_step = args.batch_size * world_size * args.gradient_accumulation * 2048
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

    # Fused cross-entropy: never materializes full [B*S, vocab] logits
    from complexity_cuda.fused_cross_entropy import fused_cross_entropy

    def compute_loss(model, batch):
        input_ids = batch["input_ids"].to(trainer.device)
        labels = batch["labels"].to(trainer.device)
        outputs = model(input_ids)
        hidden = outputs["last_hidden_state"] if isinstance(outputs, dict) else outputs
        m = model
        while hasattr(m, 'model') or hasattr(m, 'module'):
            m = getattr(m, 'model', None) or getattr(m, 'module', None)
        weight = m.embed_tokens.weight
        shift_hidden = hidden[:, :-1, :].contiguous()
        shift_labels = labels[:, :shift_hidden.size(1)].contiguous()
        return fused_cross_entropy(shift_hidden, weight, shift_labels)

    trainer.compute_loss = compute_loss

    # Logging — rank 0 only
    csv_file = None
    tqdm_cb = None
    if is_main:
        if args.wandb:
            wandb_cb = WandBCallback(project=args.wandb, name="400m-v1")
            trainer.callbacks.append(wandb_cb)

        tqdm_cb = TqdmCallback(total_steps=max_steps, desc="400M v1")
        trainer.callbacks.append(tqdm_cb)

        os.makedirs(args.checkpoint_dir, exist_ok=True)
        csv_path = os.path.join(args.checkpoint_dir, "training_log.csv")
        file_mode = "a" if args.resume and os.path.exists(csv_path) else "w"
        csv_file = open(csv_path, file_mode, newline="")
        csv_writer = csv.writer(csv_file)
        if file_mode == "w":
            csv_writer.writerow(["step", "loss", "ppl", "elapsed_s"])
        csv_file.flush()
        t_start = time.time()

        def csv_callback(trainer_obj, step, loss_val):
            real_loss = loss_val
            ppl = math.exp(min(real_loss, 20))
            csv_writer.writerow([step, f"{real_loss:.6f}", f"{ppl:.2f}", f"{time.time() - t_start:.1f}"])
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
        model.save_pretrained(os.path.join(args.checkpoint_dir, "final"))
        config.save(os.path.join(args.checkpoint_dir, "final", "model_config.yaml"))
        logger.info(f"Model saved to {args.checkpoint_dir}/final/")

    if distributed:
        cleanup()


if __name__ == "__main__":
    from complexity.gpu.distributed_cleanup import safe_main
    safe_main(main)
