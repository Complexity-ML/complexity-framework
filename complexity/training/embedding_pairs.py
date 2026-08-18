"""IterableDataset pair sources for contrastive embedding training.

Two stages, matching the E5/GTE/BGE-style recipe: AllNLI (broad,
weakly-supervised entailment pairs) for stage 1, MS MARCO (curated
query/relevant-passage pairs) for stage 2 via --init-checkpoint. Follows this
repo's established convention (corpus_mixture.py's load_dataset(...,
streaming=...) + manual per-example tokenize in __iter__, see
WeightedStreamingTextDataset) rather than dataset.map(), including the same
rank/worker sharding via HF's .shard().
"""

from __future__ import annotations

from typing import Iterator, Optional

import torch
from torch.utils.data import IterableDataset, get_worker_info


def _tokenize_pair(
    tokenizer, anchor_text: str, positive_text: str, max_seq_len: int
) -> dict[str, torch.Tensor]:
    anchor = tokenizer(
        anchor_text, padding="max_length", truncation=True,
        max_length=max_seq_len, return_tensors="pt",
    )
    positive = tokenizer(
        positive_text, padding="max_length", truncation=True,
        max_length=max_seq_len, return_tensors="pt",
    )
    return {
        "anchor_input_ids": anchor["input_ids"][0],
        "anchor_attention_mask": anchor["attention_mask"][0].float(),
        "positive_input_ids": positive["input_ids"][0],
        "positive_attention_mask": positive["attention_mask"][0].float(),
    }


def _shard_stream(dataset, rank: int, world_size: int):
    worker = get_worker_info()
    worker_count = worker.num_workers if worker is not None else 1
    worker_id = worker.id if worker is not None else 0
    shard_count = world_size * worker_count
    shard_index = rank * worker_count + worker_id
    if shard_count > 1:
        dataset = dataset.shard(num_shards=shard_count, index=shard_index)
    return dataset


class AllNLIPairDataset(IterableDataset):
    """Stage 1: entailment pairs (anchor premise / positive hypothesis)."""

    def __init__(
        self,
        tokenizer,
        *,
        max_seq_len: int = 128,
        split: str = "train",
        rank: int = 0,
        world_size: int = 1,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.split = split
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        from datasets import load_dataset

        stream = load_dataset(
            "sentence-transformers/all-nli", "pair", split=self.split, streaming=True,
        )
        stream = _shard_stream(stream, self.rank, self.world_size)
        for example in stream:
            anchor = example.get("anchor")
            positive = example.get("positive")
            if not anchor or not positive:
                continue
            yield _tokenize_pair(self.tokenizer, anchor, positive, self.max_seq_len)


class MSMarcoPairDataset(IterableDataset):
    """Stage 2: query / relevant-passage pairs from MS MARCO.

    sentence-transformers/msmarco splits text across separate "queries"
    (query_id -> text) and "corpus" (passage_id -> text, where passage_id
    equals the row index -- verified empirically, not an assumption) configs,
    joined by "triplets" (query_id, positive_id, negative_id). Only the
    positive pair is used; in-batch negatives supply the rest (see
    complexity/training/info_nce.py).

    "queries" (~500K-800K short strings) is materialized into an in-memory
    id->text dict once per worker -- small enough to be cheap. "corpus"
    (~8.8M passages) is kept as an HF Arrow-backed dataset (memory-mapped,
    not loaded into RAM) and indexed by row via the passage_id==row-index
    identity, avoiding a multi-GB in-memory lookup.
    """

    def __init__(
        self,
        tokenizer,
        *,
        max_seq_len: int = 128,
        split: str = "train",
        rank: int = 0,
        world_size: int = 1,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.split = split
        self.rank = rank
        self.world_size = world_size
        self._query_text_by_id: Optional[dict[str, str]] = None
        self._corpus = None

    def _ensure_lookups_loaded(self) -> None:
        if self._query_text_by_id is not None:
            return
        from datasets import load_dataset

        queries = load_dataset("sentence-transformers/msmarco", "queries", split=self.split)
        self._query_text_by_id = {row["query_id"]: row["query"] for row in queries}
        self._corpus = load_dataset("sentence-transformers/msmarco", "corpus", split=self.split)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        from datasets import load_dataset

        self._ensure_lookups_loaded()
        triplets = load_dataset(
            "sentence-transformers/msmarco", "triplets", split=self.split, streaming=True,
        )
        triplets = _shard_stream(triplets, self.rank, self.world_size)
        for example in triplets:
            query_text = self._query_text_by_id.get(example["query_id"])
            if query_text is None:
                continue
            passage_text = self._corpus[int(example["positive_id"])]["passage"]
            if not query_text or not passage_text:
                continue
            yield _tokenize_pair(self.tokenizer, query_text, passage_text, self.max_seq_len)
