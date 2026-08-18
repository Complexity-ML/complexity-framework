"""Tests for the in-batch-negative InfoNCE loss (complexity/training/info_nce.py)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from complexity.training.info_nce import info_nce_loss


def test_perfectly_aligned_pairs_give_low_loss():
    torch.manual_seed(0)
    anchors = F.normalize(torch.randn(8, 16), dim=-1)
    positives = anchors.clone()  # each positive is exactly its own anchor

    loss = info_nce_loss(anchors, positives, temperature=0.05)

    assert loss.item() < 0.05


def test_shuffled_pairs_give_higher_loss_than_aligned():
    torch.manual_seed(0)
    anchors = F.normalize(torch.randn(8, 16), dim=-1)
    positives = anchors.clone()
    shuffled_positives = positives[torch.randperm(8)]

    aligned_loss = info_nce_loss(anchors, positives, temperature=0.05)
    shuffled_loss = info_nce_loss(anchors, shuffled_positives, temperature=0.05)

    assert shuffled_loss.item() > aligned_loss.item()


def test_rejects_batch_size_of_one():
    anchors = F.normalize(torch.randn(1, 16), dim=-1)
    positives = anchors.clone()

    with pytest.raises(ValueError, match="batch_size"):
        info_nce_loss(anchors, positives)


def test_rejects_mismatched_shapes():
    anchors = F.normalize(torch.randn(4, 16), dim=-1)
    positives = F.normalize(torch.randn(4, 8), dim=-1)

    with pytest.raises(ValueError, match="shape mismatch"):
        info_nce_loss(anchors, positives)


def test_symmetric_matches_average_of_both_directions():
    torch.manual_seed(0)
    anchors = F.normalize(torch.randn(6, 16), dim=-1)
    positives = F.normalize(torch.randn(6, 16), dim=-1)

    asymmetric = info_nce_loss(anchors, positives, temperature=0.1, symmetric=False)
    symmetric = info_nce_loss(anchors, positives, temperature=0.1, symmetric=True)

    similarity = (anchors @ positives.T) / 0.1
    labels = torch.arange(6)
    expected_symmetric = (
        F.cross_entropy(similarity, labels) + F.cross_entropy(similarity.T, labels)
    ) / 2

    assert torch.isclose(symmetric, expected_symmetric)
    assert not torch.isclose(asymmetric, symmetric)


def test_backward_compatible_when_hard_negatives_is_none():
    torch.manual_seed(0)
    anchors = F.normalize(torch.randn(6, 16), dim=-1)
    positives = F.normalize(torch.randn(6, 16), dim=-1)

    without_arg = info_nce_loss(anchors, positives, temperature=0.1)
    with_none = info_nce_loss(anchors, positives, temperature=0.1, hard_negative_embeddings=None)

    assert torch.equal(without_arg, with_none)


def test_hard_negatives_that_mimic_the_positive_increase_loss():
    """A hard negative nearly identical to the anchor's own true positive
    makes the retrieval task strictly harder than in-batch negatives alone."""
    torch.manual_seed(0)
    anchors = F.normalize(torch.randn(6, 16), dim=-1)
    positives = F.normalize(torch.randn(6, 16), dim=-1)

    easy_negatives = F.normalize(torch.randn(6, 3, 16), dim=-1)
    hard_negatives = positives.unsqueeze(1).expand(6, 3, 16) + 0.01 * torch.randn(6, 3, 16)
    hard_negatives = F.normalize(hard_negatives, dim=-1)

    baseline = info_nce_loss(anchors, positives, temperature=0.1)
    with_easy = info_nce_loss(anchors, positives, temperature=0.1, hard_negative_embeddings=easy_negatives)
    with_hard = info_nce_loss(anchors, positives, temperature=0.1, hard_negative_embeddings=hard_negatives)

    assert with_hard.item() > with_easy.item() > baseline.item()


def test_hard_negatives_rejects_wrong_batch_dimension():
    anchors = F.normalize(torch.randn(4, 16), dim=-1)
    positives = F.normalize(torch.randn(4, 16), dim=-1)
    mismatched_negatives = F.normalize(torch.randn(3, 2, 16), dim=-1)  # batch=3, not 4

    with pytest.raises(ValueError, match="hard_negative_embeddings must be"):
        info_nce_loss(anchors, positives, hard_negative_embeddings=mismatched_negatives)


def test_gradients_flow_through_hard_negatives():
    raw_anchors = torch.randn(4, 16, requires_grad=True)
    raw_positives = torch.randn(4, 16, requires_grad=True)
    raw_hard_negatives = torch.randn(4, 2, 16, requires_grad=True)
    anchors = F.normalize(raw_anchors, dim=-1)
    positives = F.normalize(raw_positives, dim=-1)
    hard_negatives = F.normalize(raw_hard_negatives, dim=-1)

    loss = info_nce_loss(anchors, positives, hard_negative_embeddings=hard_negatives)
    loss.backward()

    assert raw_hard_negatives.grad is not None and torch.isfinite(raw_hard_negatives.grad).all()


def test_gradients_flow_to_both_inputs():
    raw_anchors = torch.randn(4, 16, requires_grad=True)
    raw_positives = torch.randn(4, 16, requires_grad=True)
    anchors = F.normalize(raw_anchors, dim=-1)
    positives = F.normalize(raw_positives, dim=-1)

    loss = info_nce_loss(anchors, positives)
    loss.backward()

    assert raw_anchors.grad is not None and torch.isfinite(raw_anchors.grad).all()
    assert raw_positives.grad is not None and torch.isfinite(raw_positives.grad).all()
