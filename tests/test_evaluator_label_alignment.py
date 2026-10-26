"""Keep training and evaluation aligned on pre-shifted, masked targets."""

import math

import pytest
import torch
from torch import nn

from complexity.evaluation import EvalConfig, Evaluator, compute_perplexity


class FixedLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(
            torch.tensor([[0.0, 5.0, 0.0], [0.0, 0.0, 5.0], [5.0, 0.0, 0.0]])
        )

    def forward(self, input_ids, attention_mask=None):
        return {
            "logits": self.logits[: input_ids.size(1)]
            .unsqueeze(0)
            .expand(input_ids.size(0), -1, -1)
        }


def test_last_supervised_eos_is_counted_by_all_lm_metrics():
    model = FixedLanguageModel()
    evaluator = Evaluator(model, config=EvalConfig(verbose=False))
    batch = {"input_ids": torch.tensor([[0, 1, 2]]), "labels": torch.tensor([[-100, -100, 0]])}
    probability = torch.softmax(model.logits[-1], dim=-1)[0].item()
    assert evaluator.accuracy([batch]) == 1.0
    assert evaluator.perplexity([batch]) == pytest.approx(1 / probability)
    assert evaluator.calibration_error([batch]) == pytest.approx(1 - probability)


def test_explicit_unshifted_labels_and_implicit_next_tokens_agree():
    model = FixedLanguageModel()
    evaluator = Evaluator(model, config=EvalConfig(verbose=False, labels_are_shifted=False))
    ids = torch.tensor([[0, 1, 2]])
    batch = {"input_ids": ids, "labels": ids.clone()}
    expected_loss = nn.functional.cross_entropy(model.logits[:2], ids[0, 1:]).item()
    assert evaluator.perplexity([batch]) == pytest.approx(math.exp(expected_loss))
    assert evaluator.perplexity([{"input_ids": ids}]) == pytest.approx(math.exp(expected_loss))
    assert evaluator.accuracy([batch]) == 1.0


def test_all_masked_batch_has_no_calibration_result():
    evaluator = Evaluator(FixedLanguageModel(), config=EvalConfig(verbose=False))
    batch = {"input_ids": torch.tensor([[0, 1, 2]]), "labels": torch.full((1, 3), -100)}
    with pytest.raises(ValueError, match="supervised target"):
        evaluator.calibration_error([batch])


class TransitionLanguageModel(nn.Module):
    """Predictions depend on the preceding token, independently of window size."""

    def __init__(self):
        super().__init__()
        self.config = {"max_position_embeddings": 4}
        self.transitions = nn.Parameter(
            torch.tensor([[0.0, 4.0, 1.0], [1.0, 0.0, 3.0], [2.0, 1.0, 0.0]])
        )
        self.windows = []

    def forward(self, input_ids, attention_mask=None):
        self.windows.append((input_ids.shape[1], attention_mask is not None))
        return {"logits": self.transitions[input_ids]}


@pytest.mark.parametrize("stride", [1, 2, 3, 5])
def test_sliding_perplexity_scores_every_unpadded_batch_target_once(stride):
    model = TransitionLanguageModel()
    ids = torch.tensor([[0, 1, 2, 0, 1, 2, 0], [0, 0, 1, 2, 1, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0, 0]])
    targets = ids[:, 1:].clone()
    targets.masked_fill_(~(mask[:, :-1].bool() & mask[:, 1:].bool()), -100)
    expected_loss = nn.functional.cross_entropy(
        model.transitions[ids[:, :-1]].reshape(-1, 3), targets.reshape(-1)
    ).item()
    assert compute_perplexity(model, ids, attention_mask=mask, stride=stride) == pytest.approx(
        math.exp(expected_loss)
    )
    assert all(length <= 4 and mask_given for length, mask_given in model.windows)


def test_sliding_perplexity_has_no_score_for_an_empty_target_set():
    ids = torch.tensor([0, 1, 2])
    assert math.isinf(compute_perplexity(TransitionLanguageModel(), ids, torch.zeros_like(ids)))
    with pytest.raises(ValueError, match="positive integer"):
        compute_perplexity(TransitionLanguageModel(), ids, stride=0)


def test_all_evaluator_metrics_exclude_padding_and_pass_the_attention_mask():
    model = TransitionLanguageModel()
    ids = torch.tensor([[0, 1, 2, 0], [0, 0, 1, 2]])
    mask = torch.tensor([[1, 1, 1, 0], [0, 1, 1, 1]])
    targets = ids[:, 1:].clone()
    targets.masked_fill_(~(mask[:, :-1].bool() & mask[:, 1:].bool()), -100)
    logits = model.transitions[ids[:, :-1]]
    valid = targets != -100
    expected_loss = nn.functional.cross_entropy(logits.reshape(-1, 3), targets.reshape(-1))
    probability, prediction = logits.softmax(-1).max(-1)
    accuracy = (prediction[valid] == targets[valid]).float().mean().item()
    expected_ece = abs(probability[valid].mean().item() - accuracy)
    evaluator = Evaluator(model, config=EvalConfig(verbose=False))
    batch = {"input_ids": ids, "attention_mask": mask}
    assert evaluator.perplexity([batch]) == pytest.approx(math.exp(expected_loss.item()))
    assert evaluator.accuracy([batch]) == pytest.approx(accuracy)
    assert evaluator.calibration_error([batch], num_bins=1) == pytest.approx(expected_ece)
    assert all(mask_given for _, mask_given in model.windows)
