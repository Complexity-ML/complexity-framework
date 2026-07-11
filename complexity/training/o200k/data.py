"""Data utilities for o200k Token-Routed pretraining."""

from __future__ import annotations

import logging
import string

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset

from complexity.data.token_shards import TokenShardDataset, token_shard_frequencies
from complexity.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class RandomTokenDataset(IterableDataset):
    def __init__(self, vocab_size: int, seq_len: int, seed: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.seed = seed

    def __iter__(self):
        gen = torch.Generator().manual_seed(self.seed)
        while True:
            ids = torch.randint(0, self.vocab_size, (self.seq_len + 1,), generator=gen)
            yield {"input_ids": ids[:-1], "labels": ids[1:]}


class LocalTextDataset(IterableDataset):
    def __init__(self, tokens: list[int], seq_len: int, seed: int):
        if len(tokens) < seq_len + 2:
            raise ValueError(f"Need at least {seq_len + 2} tokens, got {len(tokens)}")
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.seq_len = seq_len
        self.seed = seed

    def __iter__(self):
        gen = torch.Generator().manual_seed(self.seed)
        high = self.tokens.numel() - self.seq_len - 1
        while True:
            start = torch.randint(0, high + 1, (1,), generator=gen).item()
            chunk = self.tokens[start : start + self.seq_len + 1]
            yield {"input_ids": chunk[:-1], "labels": chunk[1:]}


class FineWebDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        seq_len: int,
        rank: int,
        world_size: int,
        split: str = "train",
        eval_stride: int = 1000,
    ):
        from datasets import load_dataset

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.split = split
        self.eval_stride = eval_stride
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )

    def _uses_document(self, index: int) -> bool:
        is_eval = index % self.eval_stride == 0
        if self.split == "train":
            return not is_eval
        if self.split == "eval":
            return is_eval
        raise ValueError(f"Unknown FineWeb split: {self.split}")

    def __iter__(self):
        buffer: list[int] = []
        for idx, example in enumerate(self.dataset):
            if not self._uses_document(idx):
                continue
            if idx % self.world_size != self.rank:
                continue
            text = example.get("text", "")
            if not text:
                continue
            buffer.extend(self.tokenizer.encode(text, add_special_tokens=False))
            if self.tokenizer.eos_token_id is not None:
                buffer.append(self.tokenizer.eos_token_id)
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len :]
                yield {
                    "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                    "labels": torch.tensor(chunk[1:], dtype=torch.long),
                }


def load_text_tokens(path: str, tokenizer_path: str) -> list[int]:
    tokenizer = Tokenizer.load(tokenizer_path)
    from pathlib import Path

    text = Path(path).read_text(encoding="utf-8")
    tokens = tokenizer.encode(text)
    logger.info(f"Text dataset: {path} ({len(tokens):,} tokens)")
    return tokens


def infer_vocab_size(args) -> int:
    if args.vocab_size is not None:
        return args.vocab_size
    vocab_size = Tokenizer.load(args.tokenizer).vocab_size
    logger.info(f"Tokenizer vocab size: {vocab_size:,} ({args.tokenizer})")
    return vocab_size


def text_token_frequencies(path: str, tokenizer_path: str, vocab_size: int) -> torch.Tensor:
    tokens = load_text_tokens(path, tokenizer_path)
    ids = torch.tensor(tokens, dtype=torch.long)
    ids = ids[(ids >= 0) & (ids < vocab_size)]
    freqs = torch.zeros(vocab_size, dtype=torch.float32)
    if ids.numel() > 0:
        freqs.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))
    logger.info(
        f"Zipf routing frequencies: {int(freqs.sum().item()):,} tokens, "
        f"{int((freqs > 0).sum().item()):,} vocab entries"
    )
    return freqs


def tokenizer_token_classes(tokenizer_path: str, vocab_size: int) -> torch.Tensor:
    """Classify each token into coarse lexical buckets for static routing."""

    tokenizer = Tokenizer.load(tokenizer_path)
    classes = torch.zeros(vocab_size, dtype=torch.long)
    encoding = getattr(getattr(tokenizer, "_tokenizer", None), "encoding", None)
    for token_id in range(vocab_size):
        text = _decode_token_for_class(tokenizer, encoding, token_id)
        classes[token_id] = _classify_token_text(text)
    counts = torch.bincount(classes, minlength=8)
    logger.info(
        "Token classes: "
        + ", ".join(f"{idx}={int(count)}" for idx, count in enumerate(counts.tolist()) if count)
    )
    return classes


def _decode_token_for_class(tokenizer: Tokenizer, encoding, token_id: int) -> str:
    try:
        if encoding is not None and hasattr(encoding, "decode_single_token_bytes"):
            return encoding.decode_single_token_bytes(token_id).decode("utf-8", errors="replace")
        return tokenizer.decode([token_id], skip_special_tokens=False)
    except Exception:
        return ""


def _classify_token_text(text: str) -> int:
    if not text:
        return 0
    if text.isspace():
        return 1
    stripped = text.strip()
    if not stripped:
        return 1
    if stripped.isdigit():
        return 2
    if stripped.isalpha() and stripped.isascii():
        return 3
    if stripped.isalnum() and stripped.isascii():
        return 4
    if any(ord(ch) > 127 for ch in stripped):
        return 6
    if all(ch in string.punctuation for ch in stripped):
        return 5
    return 7


def split_tokens(tokens: list[int], eval_ratio: float) -> tuple[list[int], list[int]]:
    n_eval = max(2048, int(len(tokens) * eval_ratio))
    n_eval = min(n_eval, max(1, len(tokens) // 5))
    return tokens[:-n_eval], tokens[-n_eval:]


def build_loaders(args, config, rank: int, world_size: int):
    if args.dataset == "fineweb":
        tokenizer = Tokenizer.load(args.tokenizer)
        if rank == 0:
            logger.info("Dataset: FineWeb-Edu sample-10BT streaming")
        train_ds = FineWebDataset(
            tokenizer, args.seq_len, rank, world_size, split="train"
        )
        eval_ds = (
            FineWebDataset(tokenizer, args.seq_len, rank, world_size, split="eval")
            if args.eval_steps > 0
            else None
        )
    elif args.dataset == "tokens":
        if not args.tokens_path:
            raise ValueError("--tokens-path is required when --dataset tokens")
        if rank == 0:
            logger.info(f"Dataset: token shard mmap ({args.tokens_path})")
        train_ds = TokenShardDataset(
            args.tokens_path,
            args.seq_len,
            rank=rank,
            world_size=world_size,
            seed=args.seed,
            split="train",
            eval_ratio=args.eval_ratio,
        )
        eval_ds = (
            TokenShardDataset(
                args.tokens_path,
                args.seq_len,
                rank=rank,
                world_size=world_size,
                seed=args.seed + 10_000,
                split="eval",
                eval_ratio=args.eval_ratio,
            )
            if args.eval_steps > 0 else None
        )
    elif args.dataset == "text":
        if not args.text_file:
            raise ValueError("--text-file is required when --dataset text")
        tokens = load_text_tokens(args.text_file, args.tokenizer)
        train_tokens, eval_tokens = split_tokens(tokens, args.eval_ratio)
        train_ds = LocalTextDataset(train_tokens, args.seq_len, args.seed + rank)
        eval_ds = LocalTextDataset(eval_tokens, args.seq_len, args.seed + 10_000 + rank)
    else:
        train_ds = RandomTokenDataset(config.vocab_size, args.seq_len, args.seed + rank)
        eval_ds = RandomTokenDataset(config.vocab_size, args.seq_len, args.seed + 10_000 + rank)

    loader_kwargs = {"batch_size": args.batch_size, "pin_memory": False}
    if args.num_workers > 0:
        loader_kwargs.update(num_workers=args.num_workers, persistent_workers=True)
    eval_loader = DataLoader(eval_ds, **loader_kwargs) if eval_ds is not None else None
    return DataLoader(train_ds, **loader_kwargs), eval_loader


def batch_expert_counts(raw_model, input_ids: torch.Tensor, num_experts: int, distributed: bool) -> torch.Tensor:
    """Return per-expert token counts for the current batch."""

    for module in raw_model.modules():
        if hasattr(module, "token_to_expert"):
            token_to_expert = getattr(module, "topk_token_to_expert", module.token_to_expert)
            if token_to_expert.ndim == 2:
                token_ids = input_ids.clamp(0, token_to_expert.shape[1] - 1)
                expert_ids = token_to_expert[:, token_ids].reshape(-1)
            else:
                token_ids = input_ids.clamp(0, token_to_expert.numel() - 1)
                expert_ids = token_to_expert[token_ids].reshape(-1)
            counts = torch.bincount(expert_ids, minlength=num_experts).to(
                device=input_ids.device,
                dtype=torch.float32,
            )
            if distributed:
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            return counts
    counts = torch.ones(num_experts, device=input_ids.device, dtype=torch.float32)
    if distributed:
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    return counts


__all__ = [
    "RandomTokenDataset",
    "LocalTextDataset",
    "FineWebDataset",
    "TokenShardDataset",
    "batch_expert_counts",
    "build_loaders",
    "infer_vocab_size",
    "text_token_frequencies",
    "token_shard_frequencies",
    "tokenizer_token_classes",
]
