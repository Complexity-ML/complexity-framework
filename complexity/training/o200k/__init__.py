"""Utilities for the o200k Token-Routed pretraining runner."""

from .data import (
    FineWebDataset,
    LocalTextDataset,
    RandomTokenDataset,
    batch_expert_counts,
    build_loaders,
    infer_vocab_size,
    text_token_frequencies,
    text_bigram_top_n,
    text_context_sig_top_n,
    build_bigram_expert_mapping,
    token_shard_frequencies,
    tokenizer_token_classes,
)
from .cli import build_parser
from .checkpointing import load_checkpoint, save_checkpoint
from .optimizer import build_optimizer
from .profiles import PROFILES, make_config
from .runtime import (
    apply_topk_primary_weight,
    evaluate,
    init_distributed,
    reduce_average,
    reduce_average_tensor,
    scheduled_topk_primary_weight,
)

__all__ = [
    "PROFILES",
    "make_config",
    "build_parser",
    "build_optimizer",
    "save_checkpoint",
    "load_checkpoint",
    "evaluate",
    "init_distributed",
    "reduce_average",
    "reduce_average_tensor",
    "scheduled_topk_primary_weight",
    "apply_topk_primary_weight",
    "RandomTokenDataset",
    "LocalTextDataset",
    "FineWebDataset",
    "batch_expert_counts",
    "build_loaders",
    "infer_vocab_size",
    "text_token_frequencies",
    "text_bigram_top_n",
    "text_context_sig_top_n",
    "build_bigram_expert_mapping",
    "token_shard_frequencies",
    "tokenizer_token_classes",
]
