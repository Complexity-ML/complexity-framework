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

import random
from typing import Iterator, Mapping, Optional, Sequence

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


def sqrt_normalized_weights(counts: Mapping[str, int]) -> dict[str, float]:
    """weight_i = sqrt(count_i) / sum(sqrt(count_j)).

    Same reweighting formula used by configs/sft_curriculum_200m_atlas_posttrain.yaml
    for that corpus's skewed task families -- boosts small sources' share of
    the interleaving order without the extreme multiplier a uniform target
    would force onto the smallest ones.
    """
    if not counts:
        raise ValueError("counts must be non-empty")
    sqrt_counts = {name: float(count) ** 0.5 for name, count in counts.items()}
    total = sum(sqrt_counts.values())
    return {name: value / total for name, value in sqrt_counts.items()}


class WeightedPairMixtureDataset(IterableDataset):
    """Interleaves N streaming HF pair sources (uniform {query, document[,
    negative]} schema) at target weights, via the same weighted-fair-queuing
    algorithm WeightedStreamingTextDataset._next_source uses for the causal
    pretrain's corpus mixture (complexity/training/corpus_mixture.py:478-483)
    -- reimplemented locally since that one is tied to TextCorpusSource.

    Runs each source to exhaustion (one full pass through the underlying
    data); the weights control interleaving order -- which sources appear
    more often earlier vs. only late in training -- not total per-source
    exposure, since every source's full row count gets consumed exactly once
    by the time every stream is exhausted. Optional hard negatives (used by
    the supervised stage) are sampled per example from the row's "negative"
    list, with replacement if that list is shorter than num_hard_negatives.
    """

    def __init__(
        self,
        tokenizer,
        *,
        dataset_id: str,
        split_weights: Mapping[str, float],
        max_seq_len: int = 128,
        num_hard_negatives: int = 0,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        target_tokens: Optional[int] = None,
        streams: Optional[Mapping[str, "Sequence[dict]"]] = None,
    ):
        if not split_weights:
            raise ValueError("split_weights must be non-empty")
        self.tokenizer = tokenizer
        self.dataset_id = dataset_id
        self.split_weights = dict(split_weights)
        self.max_seq_len = max_seq_len
        self.num_hard_negatives = num_hard_negatives
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        # None (default): stop once every source is exhausted -- exactly one
        # full pass over the mixture (its "unique_tokens"). If set above the
        # mixture's natural unique-token total, exhausted sources are
        # restarted (not dropped) so the run can continue past one epoch --
        # a *partial* replay of whichever shards keep coming up in the
        # weighted-fair-queuing order, not a second full epoch. Mirrors
        # unique_tokens vs. trained_tokens in
        # scripts/build_tr_hash_70b_replay_plan.py for the causal pretrain.
        self.target_tokens = target_tokens
        # Test-only override: skip load_dataset/network entirely and iterate
        # these in-memory sequences instead, mirroring
        # WeightedStreamingTextDataset's _provided_streams (corpus_mixture.py).
        self._provided_streams = streams

    def _open_stream(self, name: str) -> Iterator:
        if self._provided_streams is not None:
            return iter(self._provided_streams[name])
        from datasets import load_dataset

        stream = load_dataset(self.dataset_id, split=name, streaming=True)
        return iter(_shard_stream(stream, self.rank, self.world_size))

    def _next_source(self, active: Sequence[str], counts: Mapping[str, int]) -> str:
        return min(active, key=lambda name: counts[name] / self.split_weights[name])

    def _sample_hard_negatives(self, negatives: list, rng: random.Random) -> list[str]:
        if len(negatives) >= self.num_hard_negatives:
            return rng.sample(negatives, self.num_hard_negatives)
        return [rng.choice(negatives) for _ in range(self.num_hard_negatives)]

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rng = random.Random(self.seed + self.rank * 1000 + worker_id)

        streams: dict[str, Iterator] = {name: self._open_stream(name) for name in self.split_weights}

        counts = {name: 0 for name in self.split_weights}
        exhausted: set[str] = set()
        tokens_emitted = 0
        replaying = self.target_tokens is not None

        while len(exhausted) < len(streams):
            if replaying and tokens_emitted >= self.target_tokens:
                break
            active = [name for name in self.split_weights if name not in exhausted]
            source_name = self._next_source(active, counts)
            try:
                row = next(streams[source_name])
            except StopIteration:
                if replaying and tokens_emitted < self.target_tokens:
                    # Partial replay: this source is out of fresh rows but
                    # we haven't hit the token target yet -- restart it
                    # rather than dropping it from the active pool.
                    streams[source_name] = self._open_stream(source_name)
                    continue
                exhausted.add(source_name)
                continue
            counts[source_name] += 1

            query, document = row.get("query"), row.get("document")
            if not query or not document:
                continue
            example = _tokenize_pair(self.tokenizer, query, document, self.max_seq_len)
            tokens_emitted += int(example["anchor_attention_mask"].sum())
            tokens_emitted += int(example["positive_attention_mask"].sum())

            if self.num_hard_negatives > 0:
                negatives = row.get("negative") or []
                if not negatives:
                    continue
                chosen = self._sample_hard_negatives(negatives, rng)
                neg_ids, neg_masks = [], []
                for neg_text in chosen:
                    encoded = self.tokenizer(
                        neg_text, padding="max_length", truncation=True,
                        max_length=self.max_seq_len, return_tensors="pt",
                    )
                    neg_ids.append(encoded["input_ids"][0])
                    neg_masks.append(encoded["attention_mask"][0].float())
                example["negative_input_ids"] = torch.stack(neg_ids)
                example["negative_attention_mask"] = torch.stack(neg_masks)
                tokens_emitted += sum(int(mask.sum()) for mask in neg_masks)

            yield example
