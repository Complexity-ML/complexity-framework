"""
Pre-training 3B ComplexityModel — Token-Routed MoE + Mu-Guidance.

Overtraining recipe: 3B params × 200B tokens (67× overtrain).
Uses FSDP full_shard for 8× B200 multi-GPU training.

Model: hidden=2048, layers=26, heads=32, kv_heads=8, inter=7552, 8 experts
       → ~3.02B params (Token-Routed + Mu + Shared Expert)

Usage:
    torchrun --nproc_per_node=8 scripts/train_3b_v1.py
    torchrun --nproc_per_node=8 scripts/train_3b_v1.py --resume checkpoints/3b-v1/step_10000

Complexity-ML — 2026
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
logger = logging.getLogger("train_3b")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

from complexity.config import ModelConfig
from complexity.models import ComplexityModel
from complexity.training import Trainer, TrainingConfig, WandBCallback, TqdmCallback
from complexity.training import gamma_mean, global_expert_shares
from complexity.parallel import init_distributed, get_rank, get_world_size, is_main_process, cleanup


def make_config() -> ModelConfig:
    """3B MoE: hidden=2048, 26 layers, GQA 32h/8kv, 8 experts, Mu-Guidance.
    Shared expert = full intermediate_size (7552). Each routed expert = 944.
    → ~3.02B total params.
    """
    return ModelConfig(
        hidden_size=2048,
        num_hidden_layers=26,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=7552,
        vocab_size=32000,
        max_position_embeddings=4096,
        attention_type="gqa",
        mlp_type="token_routed",
        num_experts=8,
        shared_expert=True,
        routed_gate=False,
        norm_type="rmsnorm",
        use_qk_norm=True,
        use_mu_guidance=True,
    )


# ── Dataset ───────────────────────────────────────────────────────────────

class FineWebStreamingDataset(IterableDataset):
    """FineWeb-Edu streaming with multi-GPU sharding."""

    def __init__(self, tokenizer, max_length=2048, rank=0, world_size=1):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rank = rank
        self.world_size = world_size
        logger.info(f"Connecting to FineWeb-Edu (streaming) [rank {rank}/{world_size}]...")
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-100BT",
                          split="train", streaming=True)
        if world_size > 1:
            ds = ds.shard(num_shards=world_size, index=rank)
        self.dataset = ds

    def __iter__(self):
        buffer = []
        for example in self.dataset:
            text = example.get("text", "")
            if not text:
                continue
            buffer.extend(self.tokenizer.encode(text))
            while len(buffer) >= self.max_length + 1:
                chunk = buffer[:self.max_length + 1]
                buffer = buffer[self.max_length:]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}


def compute_steps_for_tokens(target_tokens, batch_size, grad_accum, seq_len):
    tokens_per_step = batch_size * grad_accum * seq_len
    return math.ceil(target_tokens / tokens_per_step)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train 3B v1 (Token-Routed + Mu)")
    parser.add_argument("--tokenizer", type=str, default="./tokenizer")
    parser.add_argument("--target-tokens", type=int, default=200_000_000_000,
                        help="Target token count (default: 200B)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size per GPU")
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--lr-scheduler", type=str, default="auto",
                        choices=["auto", "cosine", "linear", "constant"])
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=5000)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/3b-v1")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--wandb", type=str, default=None)
    args = parser.parse_args()

    distributed = init_distributed()
    rank = get_rank()
    world_size = get_world_size()
    is_main = is_main_process()

    config = make_config()
    if is_main:
        logger.info(f"Config: {config.hidden_size}h, {config.num_hidden_layers}L, "
                     f"{config.num_attention_heads}/{config.num_key_value_heads} GQA, "
                     f"inter={config.intermediate_size}, experts={config.num_experts}")

    # Token frequencies for Zipf-balanced routing
    if is_main:
        logger.info("Computing token frequencies for Zipf-balanced routing...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
    freq_dataset = FineWebStreamingDataset(tokenizer=tokenizer, rank=0, world_size=1)
    freq_loader = DataLoader(freq_dataset, batch_size=64, num_workers=2)
    token_counts = torch.zeros(config.vocab_size, dtype=torch.long)
    for i, batch in enumerate(freq_loader):
        token_counts.scatter_add_(0, batch["input_ids"].view(-1),
                                   torch.ones_like(batch["input_ids"].view(-1), dtype=torch.long))
        if i >= 200:
            break
    config.token_frequencies = token_counts.float()

    model = ComplexityModel(config)
    total_params = sum(p.numel() for p in model.parameters())
    if is_main:
        logger.info(f"Model: {total_params / 1e9:.2f}B params")

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
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else max(1, int(max_steps * 0.05))

    tokens_per_step = args.batch_size * world_size * args.gradient_accumulation * 2048
    if is_main:
        logger.info(f"Training: {max_steps:,} steps (~{args.target_tokens/1e9:.0f}B tokens)")
        logger.info(f"  Tokens/step: {tokens_per_step:,} "
                     f"(batch={args.batch_size} × {world_size} GPUs × accum={args.gradient_accumulation} × seq=2048)")
        logger.info(f"  Warmup: {warmup_steps} steps")

    # Dataset
    dataset = FineWebStreamingDataset(tokenizer=tokenizer, rank=rank, world_size=world_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=True, persistent_workers=True)

    # Trainer
    train_config = TrainingConfig(
        max_steps=max_steps,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        lr_scheduler=args.lr_scheduler,
        save_steps=args.save_steps,
        log_steps=args.log_steps,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        use_fsdp=True,
        bf16=True,
    )

    trainer = Trainer(model=model, config=train_config, train_dataloader=dataloader)

    if is_main:
        logger.info(f"FSDP enabled ({world_size} GPUs, full_shard)")

    # Fused cross-entropy
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
            label_smoothing=0.1,
        )
        return loss

    trainer.compute_loss = compute_loss

    # Callbacks
    n_experts = config.num_experts
    t_start = time.time()
    csv_file = None
    csv_writer = None

    # TqdmCallback on all ranks (collective inside)
    tqdm_cb = TqdmCallback(total_steps=max_steps, desc="3B v1")
    trainer.callbacks.append(tqdm_cb)

    if is_main:
        if args.wandb:
            wandb_cb = WandBCallback(project=args.wandb, name="3b-v1")
            trainer.callbacks.append(wandb_cb)

        os.makedirs(args.checkpoint_dir, exist_ok=True)
        csv_path = os.path.join(args.checkpoint_dir, "training_log.csv")
        file_mode = "a" if args.resume and os.path.exists(csv_path) else "w"
        csv_file = open(csv_path, file_mode, newline="")
        csv_writer = csv.writer(csv_file)
        expert_cols = [f"expert_{e}_share" for e in range(n_experts)]
        if file_mode == "w":
            csv_writer.writerow([
                "step", "loss", "ppl", "lr", "tokens_seen",
                *expert_cols, "expert_dead_count", "elapsed_s",
            ])
        csv_file.flush()

    def csv_callback(trainer_obj, step, loss_val):
        if not is_main:
            return
        real_loss = loss_val
        ppl = math.exp(min(real_loss, 20))
        lr = trainer_obj.optimizer.param_groups[0]["lr"]
        shares = getattr(tqdm_cb, "last_shares", [float("nan")] * n_experts)
        dead = getattr(tqdm_cb, "last_dead", n_experts)
        tokens_seen = step * tokens_per_step
        csv_writer.writerow([
            step, f"{real_loss:.6f}", f"{ppl:.2f}",
            f"{lr:.6e}", tokens_seen,
            *[f"{s:.4f}" for s in shares], dead,
            f"{time.time() - t_start:.1f}",
        ])
        if step % 100 == 0:
            csv_file.flush()
    trainer.callbacks.append(csv_callback)

    if args.resume:
        trainer.resume_from_checkpoint(args.resume)

    logging.getLogger("complexity.training.trainer").setLevel(logging.WARNING)

    if is_main:
        logger.info("Starting training...")
    import signal
    signal.signal(signal.SIGTERM, lambda s, f: (_ for _ in ()).throw(KeyboardInterrupt()))

    try:
        summary = trainer.train()
    except KeyboardInterrupt:
        summary = None
        if is_main:
            logger.info("Interrupted — saving checkpoint...")
    finally:
        if is_main and csv_file:
            csv_file.flush()
            csv_file.close()
        logging.getLogger("complexity.training.trainer").setLevel(logging.INFO)

    if is_main and summary is not None:
        logger.info(f"Training complete: {summary}")

    # Save — all ranks call (collective full_tensor)
    if summary is not None:
        base = model
        while not hasattr(base, 'save_pretrained'):
            next_base = getattr(base, 'module', None) or getattr(base, 'model', None)
            base = next_base if (next_base is not None and next_base is not base) else base
            if not hasattr(base, 'save_pretrained'):
                break
        base.save_pretrained(os.path.join(args.checkpoint_dir, "final"))
        if is_main:
            config.save(os.path.join(args.checkpoint_dir, "final", "model_config.yaml"))
            logger.info(f"Model saved to {args.checkpoint_dir}/final/")

    if distributed:
        import torch.distributed as dist
        dist.barrier()
        cleanup()


if __name__ == "__main__":
    from complexity.gpu.distributed_cleanup import safe_main
    safe_main(main)
