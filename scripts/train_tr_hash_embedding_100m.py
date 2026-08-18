#!/usr/bin/env python3
"""Train the TR-Hash ~118M bidirectional text-embedding model.

Three sequential stages, each chained via --init-checkpoint (fresh
optimizer/scheduler, same idea as
scripts/vast_finetune_tr_hash_200m_70b_unique_phase2.sh):

    --stage nomic_unsupervised  238,998,494 pairs across 28 weakly-supervised
                                 sources (nomic-ai/nomic-embed-unsupervised-data),
                                 sqrt-weighted. unique_tokens = 24.9B (one full
                                 pass); --target-tokens above that replays
                                 whichever shards keep coming up in the
                                 weighted-fair-queuing order, not a second
                                 full epoch (see NOMIC_UNSUPERVISED_UNIQUE_TOKENS).
    --stage nomic_supervised    10 curated sources with mined hard negatives
                                 (nomic-ai/nomic-embed-supervised-data),
                                 ~1.7M pairs, 1 epoch, matches the Nomic Embed
                                 paper's own recipe exactly.
    --stage allnli               314,315 entailment pairs, 1 epoch, final
                                 stage -- kept separate from nomic_supervised,
                                 not merged (dropped this project's plain
                                 MSMarco pairs from the mix: redundant with
                                 nomic_supervised's own reranked+hard-negative
                                 MSMarco distillation split).

This does NOT subclass TrainRunner: TrainRunner.run() hardcodes
`model = ComplexityModel(config)` (no pluggable model class) and builds its
DataLoader with no collate_fn hook -- both fine for the causal pretrain
scripts but incompatible with BidirectionalEmbeddingModel and the
anchor/positive(/negative) batch shape here. Trainer/TrainingConfig/
CheckpointManager are loss- and model-agnostic (compute_loss is a plain
callable), so this script drives them directly instead.

Usage:
    python -m scripts.train_tr_hash_embedding_100m \
        --stage nomic_unsupervised --checkpoint-dir artifacts/tr_hash_embedding_100m_stage0 \
        --target-tokens 30000000000

    python -m scripts.train_tr_hash_embedding_100m \
        --stage nomic_supervised --checkpoint-dir artifacts/tr_hash_embedding_100m_stage1 \
        --init-checkpoint artifacts/tr_hash_embedding_100m_stage0/final

    python -m scripts.train_tr_hash_embedding_100m \
        --stage allnli --checkpoint-dir artifacts/tr_hash_embedding_100m_stage2 \
        --init-checkpoint artifacts/tr_hash_embedding_100m_stage1/final
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
from complexity.training.embedding_pairs import (
    AllNLIPairDataset,
    MSMarcoPairDataset,
    WeightedPairMixtureDataset,
    sqrt_normalized_weights,
)
from complexity.training.info_nce import info_nce_loss
from complexity.training.trainer import Trainer
from complexity.utils.device import configure_torch_acceleration

logger = logging.getLogger("train_tr_hash_embedding_100m")

# Verified via datasets.load_dataset_builder against each split (see the
# implementation plan) -- hardcoded rather than queried at every launch,
# mirroring scripts/build_tr_hash_70b_replay_plan.py's DEFAULT_UNIQUE_BUDGETS.
NOMIC_UNSUPERVISED_SPLIT_COUNTS = {
    "reddit_title_body": 66_204_599, "paq": 53_874_545, "amazon_reviews": 39_357_860,
    "s2orc_title_abstract": 36_051_582, "wikianswers": 10_087_503, "s2orc_citation_titles": 7_722_225,
    "s2orc_abstract_citation": 7_639_890, "s2orc_abstract_body": 6_550_431, "wikipedia": 6_198_049,
    "gooaq": 1_281_138, "codesearch": 864_023, "agnews": 420_288, "npr": 365_075,
    "ccnews": 353_670, "cnn": 293_521, "yahoo_title_answer": 276_726, "amazonqa": 226_137,
    "yahoo_title_question": 213_320, "sentence_compression": 173_604, "yahoo_qa": 143_477,
    "altlex": 110_708, "eli5": 106_781, "simplewiki": 97_717, "wikihow": 96_029,
    "stackexchange_title_body": 80_695, "stackexchange_duplicate_questions": 73_210,
    "stackexchange_body_body": 65_689, "quora": 44_885, "squad": 25_117,
}
NOMIC_SUPERVISED_SPLIT_COUNTS = {
    "msmarco_distillation_simlm_rescored_reranked_min15": 485_721,
    "nli_simcse_50negs_fixed": 275_595, "reddit_triples": 200_000,
    "medi_supernli_sampled": 177_639, "hotpotqa_hn_mine_shuffled": 170_000,
    "fever_hn_mine": 140_085, "medi_sts_stackexchange_dupe": 100_591,
    "nq_cocondensor_hn_mine_reranked_min15": 70_002, "medi_sts_flickr_sampled": 51_186,
    "medi_sts_wiki_rephrasal": 25_000,
}
# Empirically measured (real 3,200-text sample, this project's tokenizer):
# mean 52 tok/text for the unsupervised mix -> 239M pairs * 2 * 52 ~= 24.9B.
NOMIC_UNSUPERVISED_UNIQUE_TOKENS = 24_900_000_000
DEFAULT_NUM_HARD_NEGATIVES = 4


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

    hard_negative_emb = None
    if "negative_input_ids" in batch:
        batch_size, num_hard_negatives, seq_len = batch["negative_input_ids"].shape
        flat_ids = batch["negative_input_ids"].reshape(batch_size * num_hard_negatives, seq_len)
        flat_mask = batch["negative_attention_mask"].reshape(batch_size * num_hard_negatives, seq_len)
        flat_emb = model(flat_ids.to(device), flat_mask.to(device))
        hard_negative_emb = flat_emb.reshape(batch_size, num_hard_negatives, -1)

    return info_nce_loss(
        anchor_emb, positive_emb, temperature=temperature, hard_negative_embeddings=hard_negative_emb,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["nomic_unsupervised", "nomic_supervised", "allnli", "msmarco"],
        required=True,
        help="msmarco/allnli are this project's original plain pairs (Part 1); "
        "nomic_unsupervised/nomic_supervised are the SOTA-scale weighted mixtures (Part 2).",
    )
    parser.add_argument("--tokenizer", default="tokenizer")
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64, help="Per-GPU batch size.")
    parser.add_argument(
        "--gradient-accumulation", type=int, default=1,
        help="Always 1 for InfoNCE unless you have a specific reason to change it: each "
        "micro-batch computes its own loss independently (no cross-micro-batch negatives), "
        "so accumulation never grows the in-batch negative pool the way it would for an "
        "LM loss -- batch_size, not gradient_accumulation, is the lever for more negatives.",
    )
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument(
        "--num-hard-negatives", type=int, default=DEFAULT_NUM_HARD_NEGATIVES,
        help="Only used by --stage nomic_supervised (the only stage with mined hard negatives).",
    )
    parser.add_argument(
        "--target-tokens", type=int, default=None,
        help=f"Only used by --stage nomic_unsupervised. None (default) = stop after one full pass "
        f"(unique_tokens ~= {NOMIC_UNSUPERVISED_UNIQUE_TOKENS:,}). Above that, exhausted sources "
        f"restart (partial replay of whichever shards recur in the weighted-fair-queuing order) "
        f"instead of a second full epoch.",
    )
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
        help="HF-style export (e.g. the previous stage's final/) to load weights from with a "
        "fresh optimizer/scheduler.",
    )
    parser.add_argument("--distributed-mode", choices=["ddp", "fsdp"], default="ddp")
    parser.add_argument("--use-custom-kernels", default="auto", choices=["auto", "true", "false"])
    return parser


def build_dataset(args, tokenizer, *, rank: int, world_size: int):
    if args.stage == "allnli":
        return AllNLIPairDataset(tokenizer, max_seq_len=args.max_seq_len, rank=rank, world_size=world_size)
    if args.stage == "msmarco":
        return MSMarcoPairDataset(tokenizer, max_seq_len=args.max_seq_len, rank=rank, world_size=world_size)
    if args.stage == "nomic_unsupervised":
        weights = sqrt_normalized_weights(NOMIC_UNSUPERVISED_SPLIT_COUNTS)
        target_tokens = args.target_tokens
        if is_main_process():
            if target_tokens is None:
                logger.info(
                    f"nomic_unsupervised: one full pass, unique_tokens ~= "
                    f"{NOMIC_UNSUPERVISED_UNIQUE_TOKENS:,}"
                )
            else:
                extra = target_tokens - NOMIC_UNSUPERVISED_UNIQUE_TOKENS
                logger.info(
                    f"nomic_unsupervised: target_tokens={target_tokens:,} "
                    f"(unique={NOMIC_UNSUPERVISED_UNIQUE_TOKENS:,} + partial replay~={max(extra, 0):,})"
                )
        return WeightedPairMixtureDataset(
            tokenizer, dataset_id="nomic-ai/nomic-embed-unsupervised-data",
            split_weights=weights, max_seq_len=args.max_seq_len,
            target_tokens=target_tokens, rank=rank, world_size=world_size,
        )
    if args.stage == "nomic_supervised":
        weights = sqrt_normalized_weights(NOMIC_SUPERVISED_SPLIT_COUNTS)
        return WeightedPairMixtureDataset(
            tokenizer, dataset_id="nomic-ai/nomic-embed-supervised-data",
            split_weights=weights, max_seq_len=args.max_seq_len,
            num_hard_negatives=args.num_hard_negatives, rank=rank, world_size=world_size,
        )
    raise ValueError(f"unknown stage: {args.stage!r}")


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

    dataset = build_dataset(args, tokenizer, rank=rank, world_size=world_size)
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
