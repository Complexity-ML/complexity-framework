from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from complexity.core.losses import sft


def test_sft_liger_backend_dispatches_to_fused_linear_ce(monkeypatch) -> None:
    calls = []

    def fake_liger(hidden, weight, labels, **kwargs):
        calls.append((hidden, weight, labels, kwargs))
        return hidden.sum() * 0 + 1.25, SimpleNamespace(ce=1.25)

    monkeypatch.setattr(sft, "fused_linear_causal_lm_loss", fake_liger)
    hidden = torch.randn(2, 3, 8, requires_grad=True)
    weight = torch.randn(32, 8, requires_grad=True)
    labels = torch.randint(0, 32, (2, 3))

    loss, ce = sft.compute_sft_loss(
        hidden,
        weight,
        labels,
        fp32_loss=False,
        liger_loss=True,
        chunk_tokens=1,
        sync_metrics=True,
    )

    assert loss.item() == 1.25
    assert ce == 1.25
    assert len(calls) == 1
    assert calls[0][3] == {"use_liger": True, "sync_metrics": True}


def test_sft_liger_backend_rejects_weighted_loss() -> None:
    hidden = torch.randn(2, 3, 8)
    weight = torch.randn(32, 8)
    labels = torch.randint(0, 32, (2, 3))

    try:
        sft.compute_sft_loss(
            hidden,
            weight,
            labels,
            fp32_loss=False,
            liger_loss=True,
            chunk_tokens=1,
            example_weights=torch.ones(2),
        )
    except ValueError as error:
        assert "does not support per-example loss weights" in str(error)
    else:
        raise AssertionError("weighted Liger loss must fail closed")


@pytest.mark.parametrize("packed", [False, True])
def test_native_weighted_sft_loss_matches_dense_ce_and_gradients(packed):
    hidden = torch.randn(2, 3, 8, requires_grad=True)
    weight = torch.randn(32, 8, requires_grad=True)
    labels = torch.tensor([[2, -100, 0], [-100, 3, 0]])
    coefficients = torch.tensor([[0.5, 0.0, 0.5], [0.0, 2.0, 2.0]]) if packed else torch.tensor([0.5, 2.0])
    expanded = coefficients if packed else coefficients[:, None].expand_as(labels)
    losses = torch.nn.functional.cross_entropy(
        torch.nn.functional.linear(hidden, weight).reshape(-1, 32),
        labels.reshape(-1), reduction="none",
    ).reshape_as(labels)
    expected = (losses * expanded).sum() / expanded[labels != -100].sum()
    expected_gradients = torch.autograd.grad(expected, (hidden, weight), retain_graph=True)
    loss = sft.sft_loss_from_hidden(
        hidden, weight, labels, chunk_tokens=1, example_weights=coefficients
    )
    torch.testing.assert_close(loss, expected)
    gradients = torch.autograd.grad(loss, (hidden, weight))
    for actual, reference in zip(gradients, expected_gradients):
        torch.testing.assert_close(actual, reference)
        assert torch.count_nonzero(actual) > 0
    assert torch.count_nonzero(gradients[0][labels == -100]) == 0


def test_native_fp32_sft_loss_disables_surrounding_autocast(monkeypatch):
    hidden, weight = torch.randn(2, 3, 8), torch.randn(32, 8)
    labels = torch.tensor([[2, -100, 0], [3, -100, 0]])
    expected = sft.sft_loss_from_hidden(hidden, weight, labels, chunk_tokens=2)
    original = torch.nn.functional.linear
    calls = []

    def checked_linear(*args, **kwargs):
        calls.append(torch.is_autocast_enabled("cpu"))
        return original(*args, **kwargs)

    monkeypatch.setattr(sft.F, "linear", checked_linear)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = sft.sft_loss_from_hidden(hidden, weight, labels, chunk_tokens=1)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)
    assert calls and not any(calls)


@pytest.mark.parametrize("weights", [torch.tensor([-1.0, 1.0]), torch.zeros(2), torch.tensor([float("nan"), 1.0])])
def test_native_weighted_sft_loss_rejects_invalid_visible_mass(weights):
    with pytest.raises(ValueError, match="weight"):
        sft.sft_loss_from_hidden(
            torch.randn(2, 1, 8), torch.randn(32, 8), torch.tensor([[0], [0]]),
            chunk_tokens=1, example_weights=weights,
        )


def test_native_sft_loss_never_discards_requested_weights():
    with pytest.raises(ValueError, match="Weighted SFT loss requires fp32_loss"):
        sft.compute_sft_loss(
            torch.randn(2, 1, 8), torch.randn(32, 8), torch.tensor([[0], [0]]),
            fp32_loss=False, liger_loss=False, chunk_tokens=1, example_weights=torch.ones(2),
        )


def test_liger_runtime_validation_requires_explicit_cuda_backend(monkeypatch):
    with pytest.raises(ValueError, match="fp32_loss=False"):
        sft.validate_sft_loss_backend("cuda", fp32_loss=True, liger_loss=True)
    with pytest.raises(ValueError, match="CUDA device"):
        sft.validate_sft_loss_backend("cpu", fp32_loss=False, liger_loss=True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="available CUDA"):
        sft.validate_sft_loss_backend("cuda", fp32_loss=False, liger_loss=True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sft, "has_liger_fused_linear_ce", lambda: False)
    with pytest.raises(RuntimeError, match="fused linear cross-entropy"):
        sft.validate_sft_loss_backend("cuda", fp32_loss=False, liger_loss=True)
