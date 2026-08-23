from __future__ import annotations

import logging

import pytest
import torch

from complexity.core.losses import fused_ce


def test_cuda_fallback_emits_a_visible_regression_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A missing Liger install must never silently select full-logits CE."""

    monkeypatch.delenv("COMPLEXITY_REQUIRE_LIGER", raising=False)
    monkeypatch.setattr(fused_ce, "_liger_available", lambda: False)
    monkeypatch.setattr(fused_ce, "_PYTORCH_FALLBACK_WARNED", False)

    with caplog.at_level(logging.WARNING, logger=fused_ce.__name__):
        assert fused_ce.log_liger_fused_linear_ce_status("cuda") is False
        fused_ce.log_liger_fused_linear_ce_status("cuda")

    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1
    assert "using the slower PyTorch F.linear + cross-entropy fallback" in messages[0]
    assert "liger-kernel>=0.5.0" in messages[0]


def test_cuda_production_mode_requires_liger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production CUDA runs must fail fast instead of accepting the fallback."""

    monkeypatch.setenv("COMPLEXITY_REQUIRE_LIGER", "1")
    monkeypatch.setattr(fused_ce, "_liger_available", lambda: False)

    with pytest.raises(RuntimeError, match="COMPLEXITY_REQUIRE_LIGER=1"):
        fused_ce.log_liger_fused_linear_ce_status("cuda")


def test_cuda_production_mode_accepts_liger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COMPLEXITY_REQUIRE_LIGER", "1")
    monkeypatch.setattr(fused_ce, "_liger_available", lambda: True)

    assert fused_ce.log_liger_fused_linear_ce_status("cuda") is True


def test_cuda_startup_reports_when_liger_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(fused_ce, "_liger_available", lambda: True)

    with caplog.at_level(logging.INFO, logger=fused_ce.__name__):
        assert fused_ce.log_liger_fused_linear_ce_status("cuda") is True

    assert "Liger fused linear CE: enabled" in caplog.messages


def test_cuda_status_can_validate_without_repeating_rank_logs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(fused_ce, "_liger_available", lambda: True)

    with caplog.at_level(logging.INFO, logger=fused_ce.__name__):
        assert fused_ce.log_liger_fused_linear_ce_status("cuda", log=False) is True

    assert "Liger fused linear CE: enabled" not in caplog.messages


@pytest.mark.skipif(
    not torch.cuda.is_available() or not fused_ce.has_liger_fused_linear_ce(),
    reason="requires CUDA and liger-kernel",
)
def test_cuda_loss_dispatches_to_liger_not_pytorch_linear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GPU regression: fail if dispatch ever reaches the PyTorch fallback."""

    hidden = torch.randn(2, 4, 16, device="cuda", requires_grad=True)
    weight = torch.randn(32, 16, device="cuda", requires_grad=True)
    labels = torch.randint(0, 32, (2, 4), device="cuda")

    def fail_linear(*_args, **_kwargs):
        raise AssertionError("PyTorch F.linear fallback was called")

    monkeypatch.setattr(fused_ce.F, "linear", fail_linear)
    loss, _ = fused_ce.fused_linear_causal_lm_loss(
        hidden,
        weight,
        labels,
        sync_metrics=False,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert hidden.grad is not None
    assert weight.grad is not None
