#!/usr/bin/env python3
"""Evaluate a BidirectionalEmbeddingModel checkpoint with an in-batch
retrieval accuracy/MRR metric (not perplexity -- see
scripts/evaluate_associative_recall.py for the CSV-report pattern this
follows).

Builds a pool of N (anchor, positive) pairs from an AllNLI/MSMarco
held-out split, embeds every anchor and every positive, and for each anchor
ranks the whole positive pool by cosine similarity: accuracy@1 is "the
anchor's own positive scored highest among all N candidates", MRR is the
mean reciprocal rank of the true positive. No external qrels file needed --
the pair structure alone is the label.

Usage:
    python -m scripts.evaluate_embedding_retrieval \
        artifacts/tr_hash_embedding_100m_allnli/final \
        --source allnli --output artifacts/embedding_retrieval_eval.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from complexity.models.embedding import BidirectionalEmbeddingModel


def load_eval_pairs(source: str, split: str, num_examples: int) -> list[tuple[str, str]]:
    from datasets import load_dataset

    if source == "allnli":
        stream = load_dataset("sentence-transformers/all-nli", "pair", split=split, streaming=True)
        pairs = []
        for example in stream:
            anchor, positive = example.get("anchor"), example.get("positive")
            if not anchor or not positive:
                continue
            pairs.append((anchor, positive))
            if len(pairs) >= num_examples:
                break
        return pairs

    if source == "msmarco":
        queries = load_dataset("sentence-transformers/msmarco", "queries", split=split)
        query_text_by_id = {row["query_id"]: row["query"] for row in queries}
        corpus = load_dataset("sentence-transformers/msmarco", "corpus", split=split)
        triplets = load_dataset("sentence-transformers/msmarco", "triplets", split=split, streaming=True)
        pairs = []
        for example in triplets:
            query_text = query_text_by_id.get(example["query_id"])
            if query_text is None:
                continue
            passage_text = corpus[int(example["positive_id"])]["passage"]
            if not query_text or not passage_text:
                continue
            pairs.append((query_text, passage_text))
            if len(pairs) >= num_examples:
                break
        return pairs

    raise ValueError(f"unknown source: {source!r}")


@torch.inference_mode()
def embed_texts(
    model: BidirectionalEmbeddingModel, tokenizer, texts: list[str], *, max_seq_len: int, batch_size: int, device: torch.device,
) -> torch.Tensor:
    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts, padding="max_length", truncation=True, max_length=max_seq_len, return_tensors="pt",
        )
        emb = model(encoded["input_ids"].to(device), encoded["attention_mask"].float().to(device))
        embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--tokenizer", default="tokenizer")
    parser.add_argument("--source", choices=["allnli", "msmarco"], default="allnli")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--num-examples", type=int, default=1000)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)

    model = BidirectionalEmbeddingModel.from_pretrained(args.checkpoint)
    model.to(device).eval()

    pairs = load_eval_pairs(args.source, args.split, args.num_examples)
    if len(pairs) < 2:
        raise ValueError(f"need at least 2 eval pairs, got {len(pairs)}")
    anchors = [a for a, _ in pairs]
    positives = [p for _, p in pairs]

    anchor_embeddings = embed_texts(
        model, tokenizer, anchors, max_seq_len=args.max_seq_len, batch_size=args.batch_size, device=device,
    )
    positive_embeddings = embed_texts(
        model, tokenizer, positives, max_seq_len=args.max_seq_len, batch_size=args.batch_size, device=device,
    )

    similarity = anchor_embeddings @ positive_embeddings.T  # (N, N)
    ranks = similarity.argsort(dim=-1, descending=True)  # (N, N), indices sorted by similarity
    true_index = torch.arange(len(pairs)).unsqueeze(1)
    rank_of_true = (ranks == true_index).float().argmax(dim=-1)  # 0-indexed rank of the true positive

    accuracy_at_1 = (rank_of_true == 0).float().mean().item()
    mrr = (1.0 / (rank_of_true.float() + 1.0)).mean().item()

    print(f"source={args.source} split={args.split} n={len(pairs)}")
    print(f"accuracy@1: {accuracy_at_1:.4f}")
    print(f"MRR: {mrr:.4f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["checkpoint", "source", "split", "n", "accuracy_at_1", "mrr"])
        writer.writeheader()
        writer.writerow(
            {
                "checkpoint": args.checkpoint,
                "source": args.source,
                "split": args.split,
                "n": len(pairs),
                "accuracy_at_1": f"{accuracy_at_1:.6f}",
                "mrr": f"{mrr:.6f}",
            }
        )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
