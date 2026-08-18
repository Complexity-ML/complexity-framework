"""In-batch-negative InfoNCE loss for contrastive embedding training.

The standard objective behind E5/GTE/BGE-style sentence embeddings: every
other example's positive in the batch serves as a negative for this
example's anchor, so batch size directly controls negative-sample count.
No labels beyond "index i's positive is index i" are needed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce_loss(
    anchor_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    *,
    temperature: float = 0.05,
    symmetric: bool = False,
) -> torch.Tensor:
    """anchor_embeddings, positive_embeddings: (batch, dim), L2-normalized.

    symmetric=True also scores positive->anchor (positives ranking their own
    anchor among the batch's anchors) and averages both directions --
    slightly more signal per batch, standard in e.g. SimCSE; the default
    (False) is the plain anchor->positive direction used by
    sentence-transformers' MultipleNegativesRankingLoss, sufficient for the
    in-batch-negative regime this project uses.
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

    similarity = (anchor_embeddings @ positive_embeddings.T) / temperature
    labels = torch.arange(batch_size, device=similarity.device)
    loss = F.cross_entropy(similarity, labels)

    if symmetric:
        loss = loss + F.cross_entropy(similarity.T, labels)
        loss = loss / 2

    return loss
