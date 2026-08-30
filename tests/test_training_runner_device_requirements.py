from __future__ import annotations

import pytest

from complexity.training import runner


def test_require_cuda_available_allows_optional_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: False)

    runner.require_cuda_available(False)


def test_require_cuda_available_rejects_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="refusing to fall back to CPU"):
        runner.require_cuda_available(True)


def test_require_cuda_available_accepts_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)

    runner.require_cuda_available(True)
