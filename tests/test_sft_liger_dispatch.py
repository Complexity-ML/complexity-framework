from __future__ import annotations

from types import SimpleNamespace

import torch

from scripts import sft_500m_32k_tr as sft


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
