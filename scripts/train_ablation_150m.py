"""
TMLR Ablation Study — 150M models, 32k vocab, 8B tokens.

Fair comparison: Dense baseline vs Full Complexity (Token-Routed + Mu + Zipf).
Iso-param (<0.2% diff), same data, same scheduler, same budget.

Run 1: Dense baseline    — SwiGLU standard, no routing, no Mu (~170M)
Run 2: Full Complexity   — Token-Routed + Mu-Guidance + Zipf-balanced (~187M)
Run 3: No-Mu ablation    — Token-Routed only, Mu disabled (~187M)
Run 4: Mixtral baseline  — Learned router + top-1 + load balancing (~187M)

Usage:
    # Single GPU
    python scripts/train_ablation_150m.py --run 1 2

    # Multi-GPU (FSDP)
    torchrun --nproc_per_node=8 scripts/train_ablation_150m.py -- --run 1 2

    # Resume
    torchrun --nproc_per_node=8 scripts/train_ablation_150m.py -- --run 2 --resume checkpoints/ablation-150m/run2-full/step_5000

Complexity-ML / INL — 2026
"""

from complexity.gpu import setup_gpu
setup_gpu()

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from torch.utils.data import DataLoader, IterableDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import math
import logging
import time
import csv
from itertools import islice
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("ablation-150m")

for name in ("httpx", "httpcore", "huggingface_hub", "datasets"):
    logging.getLogger(name).setLevel(logging.WARNING)

from complexity.config import ModelConfig
from complexity.models import ComplexityModel
from complexity.training import Trainer, TrainingConfig, WandBCallback
from complexity.parallel import init_distributed, get_rank, get_world_size, is_main_process, cleanup
from complexity_cuda.fused_cross_entropy import fused_cross_entropy


# ── Architecture configs (all ~170M, 32k vocab) ─────────────────────────

def make_config_dense() -> ModelConfig:
    """Run 1: Dense SwiGLU baseline (~170.8M params)."""
    return ModelConfig(
        hidden_size=768,
        num_hidden_layers=18,
        num_attention_heads=12,
        num_key_value_heads=4,
        intermediate_size=2416,
        vocab_size=32000,
        max_position_embeddings=2048,
        attention_type="gqa",
        mlp_type="swiglu",
        num_experts=1,
        norm_type="rmsnorm",
        use_qk_norm=True,
        use_mu_guidance=False,
    )


def make_config_full() -> ModelConfig:
    """Run 2: Full Complexity — Token-Routed + Mu + Zipf + Shared Expert."""
    return ModelConfig(
        hidden_size=768,
        num_hidden_layers=18,
        num_attention_heads=12,
        num_key_value_heads=4,
        intermediate_size=2048,
        vocab_size=32000,
        max_position_embeddings=2048,
        attention_type="gqa",
        mlp_type="token_routed",
        num_experts=4,
        norm_type="rmsnorm",
        use_qk_norm=True,
        use_mu_guidance=True,
        shared_expert=True,
    )


def make_config_mixtral() -> ModelConfig:
    """Run 4: Mixtral-style MoE baseline — learned router + top-1 + load balancing.

    Same architecture as Run 2 but with a learned router instead of
    deterministic token routing. For fair comparison with our approach.
    """
    return ModelConfig(
        hidden_size=768,
        num_hidden_layers=18,
        num_attention_heads=12,
        num_key_value_heads=4,
        intermediate_size=2048,
        vocab_size=32000,
        max_position_embeddings=2048,
        attention_type="gqa",
        mlp_type="mixtral",
        num_experts=4,
        shared_expert=True,
        norm_type="rmsnorm",
        use_qk_norm=True,
        use_mu_guidance=True,
    )


def make_config_no_mu() -> ModelConfig:
    """Run 3: Token-Routed + Shared + Zipf WITHOUT Mu-Guidance (ablation)."""
    return ModelConfig(
        hidden_size=768,
        num_hidden_layers=18,
        num_attention_heads=12,
        num_key_value_heads=4,
        intermediate_size=2048,
        vocab_size=32000,
        max_position_embeddings=2048,
        attention_type="gqa",
        mlp_type="token_routed",
        num_experts=4,
        norm_type="rmsnorm",
        use_qk_norm=True,
        use_mu_guidance=False,
        shared_expert=True,
    )


RUN_CONFIGS = {
    1: ("run1-dense",       "Dense SwiGLU baseline (170M)",                 make_config_dense),
    2: ("run2-iso-param",   "Token-Routed + Mu + Shared + Zipf (176M)",     make_config_full),
    3: ("run3-no-mu",       "Token-Routed + Shared + Zipf, NO Mu (170M)",   make_config_no_mu),
    4: ("run4-mixtral",     "Mixtral-style MoE (learned router, top-1)",    make_config_mixtral),
}


# ── Dataset ──────────────────────────────────────────────────────────────

class FineWebStreamingDataset(IterableDataset):
    """Streaming tokenized chunks from FineWeb-Edu."""

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


# ── Training ─────────────────────────────────────────────────────────────

def compute_steps_for_tokens(target_tokens, batch_size, grad_accum, seq_len):
    tokens_per_step = batch_size * grad_accum * seq_len
    return math.ceil(target_tokens / tokens_per_step)


def train_run(run_id, args, rank, world_size, is_main):
    """Train a single run."""
    name, desc, config_fn = RUN_CONFIGS[run_id]
    if is_main:
        logger.info("=" * 70)
        logger.info(f"  Run {run_id}: {desc}")
        logger.info(f"  Output: {args.checkpoint_dir}/{name}")
        logger.info(f"  Target: {args.target_tokens/1e9:.0f}B tokens, from scratch")
        logger.info(f"  FSDP: {world_size} GPUs, full_shard")
        logger.info("=" * 70)

    # Tokenizer
    if not os.path.exists(args.tokenizer):
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)

    # Model config
    config = config_fn()
    config.vocab_size = min(len(tokenizer), 32000)

    # Zipf-balanced routing (only for runs with experts)
    if config.num_experts > 1 and is_main:
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

    # Model (from scratch)
    model = ComplexityModel(config)
    model.gradient_checkpointing_enable()
    if is_main:
        n_params = model.num_parameters()
        logger.info(f"Model: {n_params:,} params ({n_params/1e6:.1f}M)")
        logger.info(f"  hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
                    f"heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}")
        logger.info(f"  mlp={config.mlp_type}, experts={config.num_experts}, "
                    f"mu_guidance={config.use_mu_guidance}")

    # Steps
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
        logger.info(f"  Training for {max_steps:,} steps (~{args.target_tokens/1e9:.1f}B tokens)")
        logger.info(f"  Tokens/step: {tokens_per_step:,} "
                    f"(batch={args.batch_size} × {world_size} GPUs × accum={args.gradient_accumulation} × seq=2048)")
        logger.info(f"  LR: {args.lr}, warmup: {args.warmup_steps} (auto 5%), scheduler: cosine")

    # Dataset (sharded per rank)
    dataset = FineWebStreamingDataset(tokenizer=tokenizer, rank=rank, world_size=world_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Trainer with FSDP
    checkpoint_dir = os.path.join(args.checkpoint_dir, name)
    train_config = TrainingConfig(
        max_steps=max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        lr_scheduler="cosine",
        precision="bf16",
        save_steps=args.save_steps,
        log_steps=args.log_steps,
        checkpoint_dir=checkpoint_dir,
        resume_from=args.resume if run_id == args.resume_run else None,
        use_fsdp=True,
        sharding_mode="full_shard",
    )

    trainer = Trainer(model=model, config=train_config, train_dataloader=dataloader)

    # Fused cross-entropy (never materializes full logits tensor)
    # Per-expert loss tracking (only for token-routed models)
    _expert_losses = {}  # Shared between compute_loss and callback

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

        # Per-expert loss (for token-routed models)
        if config.num_experts > 1 and is_main:
            with torch.no_grad():
                # Compute full logits for per-expert measurement
                shift_logits = F.linear(shift_hidden.float(), weight.float())
                per_token_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='none',
                )
                # Get expert assignment for each token
                shift_ids = input_ids[:, 1:shift_hidden.size(1)+1].contiguous()
                token_to_expert = None
                for module in m.modules():
                    if hasattr(module, 'token_to_expert'):
                        token_to_expert = module.token_to_expert
                        break
                if token_to_expert is not None:
                    flat_ids = shift_ids.view(-1).clamp(0, config.vocab_size - 1)
                    flat_experts = token_to_expert[flat_ids]
                    valid = shift_labels.view(-1) != -100
                    for e in range(config.num_experts):
                        mask = (flat_experts == e) & valid
                        if mask.any():
                            _expert_losses[e] = per_token_loss[mask].mean().item()

        return fused_cross_entropy(shift_hidden, weight, shift_labels)

    trainer.compute_loss = compute_loss

    # W&B
    if args.wandb and is_main:
        wandb_cb = WandBCallback(project=args.wandb, name=name)
        trainer.callbacks.append(wandb_cb)

    # CSV + tqdm (rank 0 only)
    csv_file = None
    pbar = None
    if is_main:
        os.makedirs(checkpoint_dir, exist_ok=True)
        csv_path = os.path.join(checkpoint_dir, "training_log.csv")
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        expert_cols = [f"loss_e{e}" for e in range(config.num_experts)] if config.num_experts > 1 else []
        csv_writer.writerow(["step", "loss", "ppl", "lr", "elapsed_s"] + expert_cols)
        csv_file.flush()
        t_start = time.time()
        logger.info(f"CSV log: {csv_path}")

        pbar = tqdm(total=max_steps, desc=f"Run {run_id}: {name}", unit="step", dynamic_ncols=True)

        def tqdm_callback(trainer_obj, step, loss_val):
            real_loss = loss_val
            ppl = math.exp(min(real_loss, 20))
            lr = trainer_obj.optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{real_loss:.4f}", ppl=f"{ppl:.1f}", lr=f"{lr:.2e}", ordered=True)
            pbar.update(1)
            # Per-expert losses
            expert_vals = [f"{_expert_losses.get(e, 0):.6f}" for e in range(config.num_experts)] if config.num_experts > 1 else []
            csv_writer.writerow([step, f"{real_loss:.6f}", f"{ppl:.2f}", f"{lr:.2e}", f"{time.time() - t_start:.1f}"] + expert_vals)
            if step % 100 == 0:
                csv_file.flush()

        trainer.callbacks.append(tqdm_callback)
        logging.getLogger("complexity.training.trainer").setLevel(logging.WARNING)

    # SIGTERM → KeyboardInterrupt so Trainer catches it and saves checkpoint.
    # The Trainer's own except KeyboardInterrupt calls _save_checkpoint("interrupted").
    # We do NOT re-save here (FSDP save after NCCL death → broken pipe loop).
    import signal
    signal.signal(signal.SIGTERM, lambda s, f: (_ for _ in ()).throw(KeyboardInterrupt()))

    # Train
    if is_main:
        logger.info("Starting training...")
    summary = None
    try:
        summary = trainer.train()
    except (KeyboardInterrupt, SystemExit):
        pass  # Trainer already saved checkpoint
    finally:
        if pbar is not None:
            pbar.close()
        if csv_file is not None:
            csv_file.flush()
            csv_file.close()
        logging.getLogger("complexity.training.trainer").setLevel(logging.INFO)

    if summary is not None and is_main:
        logger.info(f"Run {run_id} complete: {summary}")
        model.save_pretrained(os.path.join(checkpoint_dir, "final"))
        config.save(os.path.join(checkpoint_dir, "final", "model_config.yaml"))
        logger.info(f"Model saved to {checkpoint_dir}/final/")

    return summary


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TMLR Ablation: 150M models, 8B tokens, fair comparison"
    )
    parser.add_argument("--run", type=str, default="all", nargs="+",
                        help="Run ID(s): 1, 2, 3, 4, 'all'")
    parser.add_argument("--tokenizer", type=str, default="./tokenizer")
    parser.add_argument("--target-tokens", type=int, default=8_000_000_000,
                        help="Target token count (default: 8B)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation", type=int, default=1,
                        help="Gradient accumulation (effective batch = batch × accum × GPUs)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="Warmup steps (0 = auto 5%% of total steps)")
    parser.add_argument("--save-steps", type=int, default=5000)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/ablation-150m")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--wandb", type=str, default=None)
    args = parser.parse_args()

    # Initialize distributed
    distributed = init_distributed()
    rank = get_rank()
    world_size = get_world_size()
    is_main = is_main_process()

    # Figure out which run to resume
    args.resume_run = None
    if args.resume:
        for rid in RUN_CONFIGS:
            if RUN_CONFIGS[rid][0] in args.resume:
                args.resume_run = rid
                break

    # Overview
    if is_main:
        tokens_per_step = args.batch_size * world_size * args.gradient_accumulation * 2048
        total_steps = math.ceil(args.target_tokens / tokens_per_step)
        logger.info(f"TMLR Ablation — 150M × {len(RUN_CONFIGS)} variants, FSDP ({world_size} GPUs)")
        logger.info(f"  Tokens/step: {tokens_per_step:,} "
                    f"(batch={args.batch_size} × {world_size} GPUs × accum={args.gradient_accumulation} × seq=2048)")
        logger.info(f"  Total steps per run: {total_steps:,} ({args.target_tokens/1e9:.1f}B tokens)")

    # Parse runs
    run_arg = args.run if isinstance(args.run, list) else [args.run]
    if len(run_arg) == 1 and run_arg[0] == "all":
        run_ids = list(RUN_CONFIGS.keys())
    else:
        run_ids = [int(r) for r in run_arg]
        for r in run_ids:
            if r not in RUN_CONFIGS:
                raise ValueError(f"Invalid run ID: {r}. Choose from {list(RUN_CONFIGS.keys())}.")

    # Run
    results = {}
    for run_id in run_ids:
        try:
            results[run_id] = train_run(run_id, args, rank, world_size, is_main)
        except KeyboardInterrupt:
            if is_main:
                logger.info(f"Interrupted during Run {run_id} — stopping.")
            break

    if len(run_ids) > 1 and is_main:
        logger.info("=" * 70)
        for rid, summary in results.items():
            logger.info(f"  Run {rid} ({RUN_CONFIGS[rid][1]}): {summary}")
        logger.info(f"{len(results)}/{len(run_ids)} runs completed.")

    if distributed:
        cleanup()


if __name__ == "__main__":
    from complexity.gpu.distributed_cleanup import safe_main
    safe_main(main)
