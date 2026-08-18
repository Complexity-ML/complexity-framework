#!/usr/bin/env python3
"""Train the TR-Hash ~118M bidirectional text-embedding model.

Two stages, following the E5/GTE/BGE recipe:
    --stage allnli   Stage 1: broad, weakly-supervised entailment pairs.
    --stage msmarco  Stage 2: curated query/passage pairs, full-parameter
                      refinement via --init-checkpoint pointed at stage 1's
                      final/ export (fresh optimizer/scheduler, same idea as
                      scripts/vast_finetune_tr_hash_200m_70b_unique_phase2.sh).

This does NOT subclass TrainRunner: TrainRunner.run() hardcodes
`model = ComplexityModel(config)` (no pluggable model class) and builds its
DataLoader with no collate_fn hook -- both fine for the causal pretrain
scripts but incompatible with BidirectionalEmbeddingModel and the
anchor/positive batch shape here. Trainer/TrainingConfig/CheckpointManager
are loss- and model-agnostic (compute_loss is a plain callable), so this
script drives them directly instead.

Usage:
    python -m scripts.train_tr_hash_embedding_100m \
        --stage allnli --checkpoint-dir artifacts/tr_hash_embedding_100m_allnli \
        --max-steps 20000

    python -m scripts.train_tr_hash_embedding_100m \
        --stage msmarco --checkpoint-dir artifacts/tr_hash_embedding_100m_msmarco \
        --init-checkpoint artifacts/tr_hash_embedding_100m_allnli/final \
        --max-steps 20000
"""

from __future__ import annotations

import argparse
import logging
import os
import signal

import torch
from torch.utils.data import DataLoader

from complexity.config import ModelConfig
from complexity.models.embedding import BidirectionalEmbeddingModel
from complexity.parallel import cleanup, get_rank, get_world_size, init_distributed, is_main_process
from complexity.training.callbacks import TqdmCallback
from complexity.training.config import TrainingConfig
from complexity.training.embedding_pairs import AllNLIPairDataset, MSMarcoPairDataset
from complexity.training.info_nce import info_nce_loss
from complexity.training.trainer import Trainer
from complexity.utils.device import configure_torch_acceleration

logger = logging.getLogger("train_tr_hash_embedding_100m")


def make_config(max_seq_len: int, vocab_size: int) -> ModelConfig:
    """~118M bidirectional TR-Hash encoder (see the implementation plan for
    the parameter-count derivation: BERT-base-shaped attention dims, TR-Hash
    hash-routed MoE MLP kept rather than collapsed to dense)."""
    return ModelConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=2,
        intermediate_size=224,
        shared_intermediate_size=2560,
        vocab_size=vocab_size,
        max_position_embeddings=max_seq_len,
        attention_type="gqa",
        mlp_type="tr_hash_engine",
        num_experts=4,
        top_k=2,
        top_k_primary_weight=0.5,
        routing_strategy="token_id_multi_hash",
        route_hash_count=2,
        shared_expert=True,
        use_qk_norm=True,
        tie_word_embeddings=True,
        is_causal=False,
    )


def compute_pair_loss(model, batch: dict, *, temperature: float) -> torch.Tensor:
    device = next(model.parameters()).device
    anchor_emb = model(
        batch["anchor_input_ids"].to(device), batch["anchor_attention_mask"].to(device),
    )
    positive_emb = model(
        batch["positive_input_ids"].to(device), batch["positive_attention_mask"].to(device),
    )
    return info_nce_loss(anchor_emb, positive_emb, temperature=temperature)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["allnli", "msmarco"], required=True)
    parser.add_argument("--tokenizer", default="tokenizer")
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64, help="Per-GPU batch size.")
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lr-scheduler", default="cosine")
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--precision", default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--resume", default=None, help="'auto' to resume from this stage's own checkpoint-dir.")
    parser.add_argument(
        "--init-checkpoint", default=None,
        help="HF-style export (e.g. stage 1's final/) to load weights from with a fresh optimizer/scheduler.",
    )
    parser.add_argument("--distributed-mode", choices=["ddp", "fsdp"], default="ddp")
    parser.add_argument("--use-custom-kernels", default="auto", choices=["auto", "true", "false"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S", level=logging.INFO,
    )

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

    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)

    config = make_config(args.max_seq_len, len(tokenizer))
    model = BidirectionalEmbeddingModel(config)
    if is_main:
        logger.info(f"Model: {model.num_parameters():,} params ({model.num_parameters() / 1e6:.1f}M)")

    if args.stage == "allnli":
        dataset = AllNLIPairDataset(
            tokenizer, max_seq_len=args.max_seq_len, rank=rank, world_size=world_size,
        )
    else:
        dataset = MSMarcoPairDataset(
            tokenizer, max_seq_len=args.max_seq_len, rank=rank, world_size=world_size,
        )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
    )

    train_config = TrainingConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        optimizer_type=args.optimizer,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        lr_scheduler=args.lr_scheduler,
        precision=args.precision,
        save_steps=args.save_steps,
        log_steps=args.log_steps,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume,
        init_checkpoint=args.init_checkpoint,
        use_fsdp=args.distributed_mode == "fsdp",
        sharding_mode="full_shard",
        num_workers=args.num_workers,
    )

    trainer = Trainer(
        model=model,
        config=train_config,
        train_dataloader=dataloader,
        compute_loss=lambda m, batch: compute_pair_loss(m, batch, temperature=args.temperature),
        callbacks=[TqdmCallback(total_steps=args.max_steps, desc=f"embedding-{args.stage}")],
    )

    signal.signal(signal.SIGTERM, lambda s, f: (_ for _ in ()).throw(KeyboardInterrupt()))

    summary = None
    try:
        summary = trainer.train()
    except (KeyboardInterrupt, SystemExit):
        pass

    if is_main and summary is not None:
        logger.info(f"Training complete: {summary}")

    if summary is not None:
        base = trainer.model
        while not hasattr(base, "save_pretrained"):
            nxt = getattr(base, "module", None)
            if nxt is None or nxt is base:
                break
            base = nxt
        if hasattr(base, "save_pretrained"):
            final_dir = os.path.join(args.checkpoint_dir, "final")
            base.save_pretrained(final_dir)
            if is_main:
                logger.info(f"Model saved to {final_dir}")

    if distributed:
        import torch.distributed as dist
        dist.barrier()
        cleanup()


if __name__ == "__main__":
    main()
