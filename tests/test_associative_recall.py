import torch

from complexity.evaluation.associative_recall import (
    build_associative_recall_batch,
    build_induction_batch,
    score_target_logits,
)


def test_associative_batch_repeats_key_then_asks_for_its_value():
    batch = build_associative_recall_batch(
        batch_size=3,
        distance=8,
        vocab_size=100,
        seed=7,
    )

    assert batch.input_ids.shape == (3, 12)
    assert torch.equal(batch.input_ids[:, 1], batch.target_ids)
    assert torch.equal(batch.input_ids[:, -1], batch.input_ids[:, 0])
    assert not torch.any(batch.input_ids[:, 0] == batch.target_ids)


def test_induction_batch_repeats_a_prefix_with_same_continuation():
    batch = build_induction_batch(
        batch_size=2,
        distance=6,
        vocab_size=100,
        seed=11,
    )

    assert batch.input_ids.shape == (2, 10)
    assert torch.equal(batch.input_ids[:, 2], batch.target_ids)
    assert torch.equal(batch.input_ids[:, -2:], batch.input_ids[:, :2])


def test_target_scoring_reports_accuracy_rank_margin_and_nll():
    logits = torch.tensor(
        [
            [0.0, 4.0, 1.0, -1.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    )
    targets = torch.tensor([1, 2])

    metrics = score_target_logits(logits, targets)

    assert metrics.accuracy == 0.5
    assert metrics.mean_rank == 2.0
    assert metrics.mean_margin == 0.5
    expected_nll = torch.nn.functional.cross_entropy(logits, targets).item()
    assert abs(metrics.nll - expected_nll) < 1e-6
