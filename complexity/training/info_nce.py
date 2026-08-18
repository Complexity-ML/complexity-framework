"""In-batch-negative InfoNCE loss for contrastive embedding training.

The standard objective behind E5/GTE/BGE-style sentence embeddings: every
other example's positive in the batch serves as a negative for this
example's anchor, so batch size directly controls negative-sample count.
No labels beyond "index i's positive is index i" are needed.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def info_nce_loss(
    anchor_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    *,
    temperature: float = 0.05,
    symmetric: bool = False,
    hard_negative_embeddings: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """anchor_embeddings, positive_embeddings: (batch, dim), L2-normalized.

    symmetric=True also scores positive->anchor (positives ranking their own
    anchor among the batch's anchors) and averages both directions --
    slightly more signal per batch, standard in e.g. SimCSE; the default
    (False) is the plain anchor->positive direction used by
    sentence-transformers' MultipleNegativesRankingLoss, sufficient for the
    in-batch-negative regime this project uses.

    hard_negative_embeddings: optional (batch, num_hard_negatives, dim),
    L2-normalized -- mined negatives specific to each anchor (see
    complexity/training/embedding_pairs.py's WeightedPairMixtureDataset),
    appended to that anchor's negative pool alongside the in-batch
    positives (standard technique, E5/BGE). Only affects the forward
    (anchor->positive) direction: hard negatives are anchor-specific, not
    shared candidates for other anchors, so they have no natural role in
    the symmetric (positive->anchor) direction and are excluded from it.
    None (default) reproduces the exact prior behavior.
    """
    if anchor_embeddings.shape != positive_embeddings.shape:
        raise ValueError(
            f"anchor/positive shape mismatch: {anchor_embeddings.shape} vs "
            f"{positive_embeddings.shape}"
        )
    batch_size = anchor_embeddings.shape[0]
    if batch_size < 2:
        raise ValueError(
            f"info_nce_loss needs batch_size >= 2 for in-batch negatives, got {batch_size}"
        )

    in_batch_similarity = (anchor_embeddings @ positive_embeddings.T) / temperature
    labels = torch.arange(batch_size, device=in_batch_similarity.device)

    similarity = in_batch_similarity
    if hard_negative_embeddings is not None:
        if hard_negative_embeddings.dim() != 3 or hard_negative_embeddings.shape[0] != batch_size:
            raise ValueError(
                f"hard_negative_embeddings must be (batch={batch_size}, "
                f"num_hard_negatives, dim), got {tuple(hard_negative_embeddings.shape)}"
            )
        hard_negative_similarity = (
            torch.einsum("bd,bkd->bk", anchor_embeddings, hard_negative_embeddings) / temperature
        )
        similarity = torch.cat([in_batch_similarity, hard_negative_similarity], dim=1)

    loss = F.cross_entropy(similarity, labels)

    if symmetric:
        loss = loss + F.cross_entropy(in_batch_similarity.T, labels)
        loss = loss / 2

    return loss
