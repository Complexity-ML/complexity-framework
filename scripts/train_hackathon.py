"""
Hackathon B300 — Token-Routed 400M + Mu Projection + Muon.

GPU MODE IRL Hackathon · PyTorch Conference Europe 2026
Cluster: B300 (360 PFLOP/s BF16)

Usage:
    # Single node, 2 GPU (test)
    torchrun --nproc_per_node=2 scripts/train_hackathon.py

    # Single node, 8 GPU
    torchrun --nproc_per_node=8 scripts/train_hackathon.py

    # Multi-node (e.g. 2 nodes × 8 GPU)
    torchrun --nnodes=2 --nproc_per_node=8 \\
        --master_addr=$MASTER_ADDR --master_port=29500 \\
        --node_rank=$NODE_RANK \\
        scripts/train_hackathon.py

    # Resume
    torchrun --nproc_per_node=8 scripts/train_hackathon.py --resume checkpoints/hackathon/step_5000

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
logger = logging.getLogger("hackathon")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

from complexity.config import ModelConfig, get_preset
from complexity.models import ComplexityModel
from complexity.training import Trainer, TrainingConfig, WandBCallback, TqdmCallback
from complexity.parallel import init_distributed, get_rank, get_world_size, is_main_process, cleanup
from complexity.parallel.cluster import ClusterConfig, ClusterModel


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
    parser = argparse.ArgumentParser(description="Hackathon B300 — Token-Routed 400M + Mu + Muon")
    parser.add_argument("--tokenizer", type=str, default="./tokenizer")
    parser.add_argument("--target-tokens", type=int, default=25_000_000_000,
                        help="Target token count (default: 25B — 125%% Chinchilla for 1B)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size per GPU (64 × DP GPUs = seq/step)")
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--muon-lr", type=float, default=0.02,
                        help="Muon LR for 2D weights (Newton-Schulz)")
    parser.add_argument("--adam-lr", type=float, default=3e-4,
                        help="AdamW LR for embeddings/norms/mu")
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="Warmup steps (0 = auto 5%% of total steps)")
    parser.add_argument("--lr-scheduler", type=str, default="wsd",
                        choices=["auto", "cosine", "wsd", "linear", "constant"])
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=2000)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/hackathon")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--wandb", type=str, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=False)
    parser.add_argument("--no-compile", action="store_true", default=False,
                        help="Disable torch.compile")
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

    if is_main:
        logger.info(f"World size: {world_size} GPU(s)")

    # Tokenizer
    if not os.path.exists(args.tokenizer):
        raise FileNotFoundError(
            f"Tokenizer not found: {args.tokenizer}\n"
            f"Train one first or point to an existing HF tokenizer directory."
        )
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)

    # Model — hackathon-b300 preset (400M, token-routed 4 experts, mu projection)
    config = get_preset("hackathon-b300")
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
        top5 = freqs.topk(5)
        logger.info(f"  {freqs.sum():.0f} tokens sampled, "
                    f"top-5 IDs={top5.indices.tolist()} counts={top5.values.long().tolist()}")

    model = ComplexityModel(config)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if is_main:
            logger.info("Gradient checkpointing enabled")

    if is_main:
        logger.info(f"Model: {model.num_parameters():,} params "
                    f"({model.num_parameters()/1e6:.0f}M)")
        logger.info(f"  hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
                    f"heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}")
        logger.info(f"  inter={config.intermediate_size}, experts={config.num_experts}")
        logger.info(f"  mlp={config.mlp_type}, mu_guidance={getattr(config, 'use_mu_guidance', False)}")
        logger.info(f"  Optimizer: MuonTR (lr={args.muon_lr}) + AdamW (lr={args.adam_lr})")

    # ClusterModel — DP only, batch per GPU, accum=1
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
        logger.info(f"  Effective batch: {cluster_config.effective_batch_size} sequences "
                    f"({cluster_config.effective_batch_size * 2048 / 1e6:.1f}M tokens/step)")
    model = ClusterModel(model, cluster_config)

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
    if args.warmup_steps == 0:
        args.warmup_steps = max(1, int(max_steps * 0.05))
    if is_main:
        tokens_per_step = args.batch_size * world_size * args.gradient_accumulation * 2048
        logger.info(f"Training: {max_steps:,} steps (~{args.target_tokens/1e9:.1f}B tokens)")
        logger.info(f"  Tokens/step: {tokens_per_step:,} "
                    f"(batch={args.batch_size} × {world_size} GPUs × accum={args.gradient_accumulation} × seq=2048)")
        logger.info(f"  Warmup: {args.warmup_steps} steps (auto 5%)")

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

    # Trainer (ClusterModel handles parallelism)
    train_config = TrainingConfig(
        max_steps=max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        optimizer_type="muon_tr",
        learning_rate=args.adam_lr,
        muon_lr=args.muon_lr,
        expert_lr_scale=1.5,
        expert_weight_decay=0.005,
        warmup_steps=args.warmup_steps,
        lr_scheduler=args.lr_scheduler,
        precision="bf16",
        save_steps=args.save_steps,
        log_steps=args.log_steps,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume,
        use_fsdp=False,  # ClusterModel handles parallelism
        num_workers=args.num_workers,
    )

    trainer = Trainer(model=model, config=train_config, train_dataloader=dataloader)

    # torch.compile
    if not args.no_compile and hasattr(torch, 'compile'):
        try:
            trainer.model = torch.compile(trainer.model)
            if is_main:
                logger.info("torch.compile enabled")
        except Exception as e:
            if is_main:
                logger.warning(f"torch.compile failed: {e}")

    # Fused cross-entropy: never materializes full [B*S, vocab] logits
    from complexity_cuda.fused_cross_entropy import fused_cross_entropy

    def compute_loss(model, batch):
        input_ids = batch["input_ids"].to(trainer.device)
        labels = batch["labels"].to(trainer.device)
        outputs = model(input_ids)
        hidden = outputs["last_hidden_state"] if isinstance(outputs, dict) else outputs
        # Get lm_head weight (tied with embeddings)
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
            wandb_cb = WandBCallback(project=args.wandb, name="hackathon-b300")
            trainer.callbacks.append(wandb_cb)

        tqdm_cb = TqdmCallback(total_steps=max_steps, desc="hackathon")
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

        accum = args.gradient_accumulation

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
