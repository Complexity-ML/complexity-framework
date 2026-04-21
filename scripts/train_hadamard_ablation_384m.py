"""
384M dense A/B: Kaiming vs Hadamard initialization, Chinchilla-scale.

This script is a direct twin of the `abl-dense-adamw` run
(checkpoints/abl-dense-adamw/, 3815 steps, 8B tokens on 8 GPUs).
Everything is identical to that baseline — architecture, tokenizer,
data pipeline, batch size, learning rate, schedule, gradient clipping
— with a single variable: the weight initialization of the three
SwiGLU projections.

  --init-type default   → PyTorch's default nn.Linear init (Kaiming-
                          uniform with a=sqrt(5)). Consumes global
                          RNG state. This is the baseline, and it
                          should reproduce `abl-dense-adamw` when the
                          same seed is used.

  --init-type hadamard  → Deterministic Hadamard init (Sylvester +
                          per-layer Walsh sign pattern + Xavier-std
                          scaling). No PRNG consumed anywhere.

Hardware: built for an 8×B200 (or 8×H100 / 8×A100) FSDP setup.
Runs through the framework's Trainer API, so checkpoints, resume,
gradient accumulation, and mixed-precision all behave identically to
other ablations in this codebase.

Usage:
    # Kaiming baseline (reproduces abl-dense-adamw)
    torchrun --nproc_per_node 8 \\
        scripts/train_hadamard_ablation_384m.py \\
        --init-type default \\
        --checkpoint-dir ./checkpoints/abl-dense-kaiming

    # Hadamard treatment (same config, only init differs)
    torchrun --nproc_per_node 8 \\
        scripts/train_hadamard_ablation_384m.py \\
        --init-type hadamard \\
        --checkpoint-dir ./checkpoints/abl-dense-hadamard

Compare the final metrics.csv files head-to-head; if
|Δloss| < 0.01 at convergence, Hadamard is validated at
Chinchilla-scale on this codebase.

Complexity-ML — 2026
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from itertools import islice
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import PreTrainedTokenizerFast

from complexity.config import ModelConfig
from complexity.core.mlp import hadamard_init_
from complexity.models import ComplexityModel
from complexity.training import Trainer, TrainingConfig
from complexity.parallel import (
    init_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    cleanup,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("hadamard_ablation_384m")


# --------------------------------------------------------------------------
# Model config — 384M dense SwiGLU, matches abl-dense-adamw exactly
# --------------------------------------------------------------------------

def make_config() -> ModelConfig:
    """~384M dense SwiGLU baseline — same shape as `abl-dense-adamw`.

    Identical backbone to the 383M MoE ablation set in this repository
    (hidden=1024, 20 layers, GQA 16/4, RMSNorm + QK-norm), but with a
    dense SwiGLU FFN (`mlp_type="swiglu"`) instead of the Token-Routed
    MoE. Intermediate size is 3200, matching the sum of routed +
    shared expert widths in the MoE recipe, so the parameter budget is
    comparable across init schemes.
    """
    return ModelConfig(
        hidden_size=1024,
        num_hidden_layers=20,
        num_attention_heads=16,
        num_key_value_heads=4,
        intermediate_size=3200,
        vocab_size=32000,
        max_position_embeddings=2048,
        attention_type="gqa",
        mlp_type="swiglu",        # dense, no MoE
        num_experts=1,
        shared_expert=False,
        norm_type="rmsnorm",
        use_qk_norm=True,
        use_mu_guidance=False,
        rope_fraction=0.5,        # Partial RoPE, matches the MoE recipe
    )


# --------------------------------------------------------------------------
# Hadamard re-init pass
# --------------------------------------------------------------------------

def reinit_with_hadamard(model: nn.Module) -> int:
    """Walk the model, re-initialise every SwiGLU projection with Hadamard.

    We target exactly the three linear matrices that differ between
    Kaiming and Hadamard in this ablation: ``gate_proj``, ``up_proj``,
    ``down_proj`` inside each transformer block's MLP. Attention
    projections, embeddings, and LayerNorms retain their PyTorch
    defaults so that the only variable is the FFN init — matching the
    setup described in the paper.

    Returns the number of matrices re-initialised, for logging.
    """
    count = 0
    for block_idx, block in enumerate(model.layers):
        mlp = block.mlp
        # Three linears inside a SwiGLU block. Distinct layer_idx
        # offsets (×4 + k) so the three projections receive distinct
        # Walsh sign patterns.
        base = block_idx * 4
        for name, offset in (("gate_proj", 1), ("up_proj", 2), ("down_proj", 3)):
            linear = getattr(mlp, name, None)
            if linear is None:
                continue
            hadamard_init_(linear.weight, layer_idx=base + offset)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
            count += 1
    return count


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

class FineWebStreamingDataset(IterableDataset):
    """FineWeb-Edu streaming, rank-sharded for multi-GPU."""

    def __init__(self, tokenizer, max_length: int = 2048,
                 rank: int = 0, world_size: int = 1):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rank = rank
        self.world_size = world_size
        logger.info(f"Connecting to FineWeb-Edu (streaming) [rank {rank}/{world_size}]")
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
        for example in self.dataset:
            text = example.get("text", "")
            if not text:
                continue
            buffer.extend(self.tokenizer.encode(text))
            while len(buffer) >= self.max_length + 1:
                chunk = buffer[: self.max_length + 1]
                buffer = buffer[self.max_length :]
                yield {
                    "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                    "labels":    torch.tensor(chunk[1:], dtype=torch.long),
                }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def compute_steps_for_tokens(target_tokens: int, batch_size: int,
                             grad_accum: int, seq_len: int,
                             world_size: int) -> int:
    tokens_per_step = batch_size * grad_accum * seq_len * world_size
    return math.ceil(target_tokens / tokens_per_step)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="384M dense Kaiming↔Hadamard A/B ablation"
    )
    parser.add_argument("--init-type", type=str, default="hadamard",
                        choices=["default", "hadamard"],
                        help="default = PyTorch nn.Linear default "
                             "(Kaiming-uniform); hadamard = deterministic "
                             "Hadamard init (ours).")
    parser.add_argument("--tokenizer", type=str, default="./tokenizer")
    parser.add_argument("--target-tokens", type=int, default=8_000_000_000,
                        help="Target token count (default: 8B, matching "
                             "abl-dense-adamw).")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size per GPU.")
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="Warmup steps (0 = auto 5%% of total steps).")
    parser.add_argument("--lr-scheduler", type=str, default="cosine",
                        choices=["cosine", "wsd", "constant"])
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the PyTorch RNG. Relevant only "
                             "for --init-type default (and for data "
                             "ordering under both init types).")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Defaults to ./checkpoints/abl-dense-<init>.")
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    # --- Distributed setup ---------------------------------------------------
    init_distributed()
    rank = get_rank()
    world_size = get_world_size()

    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"./checkpoints/abl-dense-{args.init_type}"
    if is_main_process():
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    # --- Tokenizer + dataset -------------------------------------------------
    if not os.path.exists(args.tokenizer):
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)

    dataset = FineWebStreamingDataset(
        tokenizer=tokenizer,
        max_length=args.seq_len,
        rank=rank,
        world_size=world_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --- Model ---------------------------------------------------------------
    config = make_config()
    config.vocab_size = len(tokenizer)
    model = ComplexityModel(config)

    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model: 384M dense SwiGLU — {n_params/1e6:.1f}M params")
        logger.info(
            f"Config: hidden={config.hidden_size}, "
            f"layers={config.num_hidden_layers}, "
            f"heads={config.num_attention_heads}/{config.num_key_value_heads}, "
            f"inter={config.intermediate_size}, mlp={config.mlp_type}"
        )

    # --- Initialization: this is the single experimental variable ------------
    if args.init_type == "hadamard":
        count = reinit_with_hadamard(model)
        if is_main_process():
            logger.info(
                f"Init: Hadamard — re-initialised {count} FFN matrices "
                f"({config.num_hidden_layers} layers × 3 projections). "
                f"Attention/embedding/norm params keep PyTorch defaults."
            )
    else:
        if is_main_process():
            logger.info(
                f"Init: PyTorch default (Kaiming-uniform). "
                f"Seeded with torch.manual_seed({args.seed})."
            )

    # --- Steps budget to match abl-dense-adamw -------------------------------
    max_steps = compute_steps_for_tokens(
        target_tokens=args.target_tokens,
        batch_size=args.batch_size,
        grad_accum=args.gradient_accumulation,
        seq_len=args.seq_len,
        world_size=world_size,
    )
    warmup = args.warmup_steps if args.warmup_steps > 0 else max(1, int(max_steps * 0.05))

    if is_main_process():
        tokens_per_step = args.batch_size * args.gradient_accumulation * args.seq_len * world_size
        logger.info(
            f"Budget: target {args.target_tokens/1e9:.1f}B tokens, "
            f"{tokens_per_step/1e6:.2f}M tokens/step → {max_steps} steps"
        )
        logger.info(f"Warmup: {warmup} steps ({warmup / max_steps * 100:.1f}%)")

    # --- Trainer (FSDP, bf16, cosine, grad-clip, checkpointing) --------------
    train_config = TrainingConfig(
        max_steps=max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        warmup_steps=warmup,
        lr_scheduler=args.lr_scheduler,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        label_smoothing=args.label_smoothing,
        precision="bf16",
        save_steps=args.save_steps,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume,
    )

    trainer = Trainer(model=model, config=train_config, train_dataloader=dataloader)
    summary = trainer.train()

    if is_main_process():
        logger.info(f"Training complete: {summary}")
        logger.info(
            f"Compare: diff {args.checkpoint_dir}/metrics.csv vs "
            f"./checkpoints/abl-dense-adamw/abl-dense-adamw.csv"
        )

    cleanup()


if __name__ == "__main__":
    main()
