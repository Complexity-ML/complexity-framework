"""
Quick 400M model training — single node, 8 GPUs, FSDP.

Tests cluster setup with a ~400M param model on 1 node.
Batch=64 per GPU, no accumulation → 1M tokens/step on 8 GPUs.

Usage:
    torchrun --nproc_per_node=8 scripts/train_400m_cluster.py
    torchrun --nproc_per_node=8 scripts/train_400m_cluster.py -- --target-tokens 8_000_000_000
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
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("train-400m")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

from complexity.config import ModelConfig
from complexity.models import ComplexityModel
from complexity.training import Trainer, TrainingConfig
from complexity.parallel import init_distributed, get_rank, get_world_size, is_main_process
from complexity.parallel.cluster import ClusterConfig, ClusterModel


def make_config_400m() -> ModelConfig:
    """~400M params: Token-Routed + Mu-Guidance + Zipf-balanced."""
    return ModelConfig(
        hidden_size=1024,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=4,
        intermediate_size=3072,
        vocab_size=32000,
        max_position_embeddings=2048,
        attention_type="gqa",
        mlp_type="token_routed",
        num_experts=4,
        norm_type="rmsnorm",
        use_qk_norm=True,
        use_mu_guidance=True,
    )


class FineWebStreamingDataset(IterableDataset):
    def __init__(self, tokenizer, max_length=2048, rank=0, world_size=1):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rank = rank
        self.world_size = world_size
        logger.info(f"Connecting to FineWeb-Edu (streaming) [rank {rank}/{world_size}]...")
        t0 = time.time()
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            split="train",
            streaming=True,
        )
        logger.info(f"Dataset ready in {time.time() - t0:.1f}s")

    def __iter__(self):
        buffer = []
        first_yield = True
        chunk_idx = 0
        for example in self.dataset:
            text = example.get("text", "")
            if not text:
                continue
            tokens = self.tokenizer.encode(text)
            buffer.extend(tokens)
            while len(buffer) >= self.max_length + 1:
                chunk = buffer[:self.max_length + 1]
                buffer = buffer[self.max_length:]
                if chunk_idx % self.world_size == self.rank:
                    input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                    labels = torch.tensor(chunk[1:], dtype=torch.long)
                    if first_yield:
                        logger.info(f"First batch tokenized [rank {self.rank}]")
                        first_yield = False
                    yield {"input_ids": input_ids, "labels": labels}
                chunk_idx += 1


def compute_steps_for_tokens(target_tokens, batch_size, grad_accum, seq_len):
    tokens_per_step = batch_size * grad_accum * seq_len
    return math.ceil(target_tokens / tokens_per_step)


def main():
    parser = argparse.ArgumentParser(description="Train 400M model — single node cluster test")
    parser.add_argument("--tokenizer", type=str, default="./tokenizer")
    parser.add_argument("--target-tokens", type=int, default=8_000_000_000,
                        help="Target tokens (default: 8B)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="Warmup steps (0 = auto 5%% of total steps)")
    parser.add_argument("--lr-scheduler", type=str, default="cosine",
                        choices=["cosine", "wsd", "linear", "constant"])
    parser.add_argument("--save-steps", type=int, default=2000)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/400m-cluster")
    parser.add_argument("--num-workers", type=int, default=0)
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
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)

    # Model
    config = make_config_400m()
    config.vocab_size = min(len(tokenizer), 32000)

    # Zipf-balanced routing (compute token frequencies)
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
        top5 = freqs.topk(5)
        logger.info(f"  {freqs.sum():.0f} tokens sampled, "
                    f"top-5 IDs={top5.indices.tolist()} counts={top5.values.long().tolist()}")

    model = ComplexityModel(config)

    if is_main:
        n = model.num_parameters()
        logger.info(f"Model: {n:,} params ({n/1e6:.1f}M)")
        logger.info(f"  hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
                    f"heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}")
        logger.info(f"  mlp={config.mlp_type}, experts={config.num_experts}, "
                    f"mu_guidance={config.use_mu_guidance}")

    # ClusterModel — DP only, batch=64 per GPU, accum=1
    # Scale séquences/step via nb de GPUs (DP)
    cluster_config = ClusterConfig(
        tp_size=1,
        pp_size=1,
        dp_size=world_size,
        micro_batch_size=args.batch_size,
    )
    if is_main:
        logger.info(f"  Cluster: TP={cluster_config.tp_size}, PP={cluster_config.pp_size}, "
                    f"DP={cluster_config.dp_size}")
        logger.info(f"  Effective batch: {cluster_config.effective_batch_size} sequences")
    model = ClusterModel(model, cluster_config)

    # Steps
    max_steps = compute_steps_for_tokens(
        target_tokens=args.target_tokens,
        batch_size=args.batch_size * world_size,
        grad_accum=args.gradient_accumulation,
        seq_len=2048,
    )
    # Auto warmup: 5% of total steps
    if args.warmup_steps == 0:
        args.warmup_steps = max(1, int(max_steps * 0.05))
    if is_main:
        tokens_per_step = args.batch_size * world_size * args.gradient_accumulation * 2048
        logger.info(f"Training: {max_steps:,} steps (~{args.target_tokens/1e9:.1f}B tokens)")
        logger.info(f"  Tokens/step: {tokens_per_step:,} "
                    f"(batch={args.batch_size} × {world_size} GPUs × accum={args.gradient_accumulation} × seq=2048)")
        logger.info(f"  LR: {args.lr}, warmup: {args.warmup_steps} (auto 5%), scheduler: {args.lr_scheduler}")

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

    # Trainer (FSDP handled by ClusterModel, disable trainer's own FSDP)
    train_config = TrainingConfig(
        max_steps=max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        lr_scheduler=args.lr_scheduler,
        precision="bf16",
        save_steps=args.save_steps,
        log_steps=args.log_steps,
        checkpoint_dir=args.checkpoint_dir,
        use_fsdp=False,  # ClusterModel handles parallelism
    )

    trainer = Trainer(model=model, config=train_config, train_dataloader=dataloader)

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

    # Progress bar (rank 0 only)
    if is_main:
        pbar = tqdm(total=max_steps, desc="400M cluster", unit="step", dynamic_ncols=True)
        def tqdm_callback(trainer_obj, step, loss_val):
            real_loss = loss_val
            ppl = math.exp(min(real_loss, 20))
            lr = trainer_obj.optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{real_loss:.4f}", ppl=f"{ppl:.1f}", lr=f"{lr:.2e}", ordered=True)
            pbar.update(1)
        trainer.callbacks.append(tqdm_callback)
        logging.getLogger("complexity.training.trainer").setLevel(logging.WARNING)

    # Train
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
        if is_main:
            if hasattr(pbar, 'close'):
                pbar.close()

    if summary is not None and is_main:
        logger.info(f"Training complete: {summary}")
        model.save_pretrained(os.path.join(args.checkpoint_dir, "final"))
        logger.info(f"Model saved to {args.checkpoint_dir}/final/")


if __name__ == "__main__":
    from complexity.gpu.distributed_cleanup import safe_main
    safe_main(main)
