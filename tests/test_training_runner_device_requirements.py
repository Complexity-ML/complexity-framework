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


def test_pre_shifted_labels_keep_the_final_target_and_use_the_output_head(tmp_path):
    """An untied output head and a supervised final target catch both regressions."""
    from types import SimpleNamespace

    import torch
    from torch import nn

    from complexity.training.trainer import Trainer

    class LanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(5, 3)
            self.lm_head = nn.Linear(3, 5, bias=False)

        def forward(self, input_ids, return_logits=True):
            hidden = self.embed_tokens(input_ids)
            return {
                "logits": self.lm_head(hidden) if return_logits else None,
                "last_hidden_state": hidden,
            }

    torch.manual_seed(7)
    model = LanguageModel()
    batch = {"input_ids": torch.tensor([[0, 1, 2]]), "labels": torch.tensor([[1, 2, 4]])}
    expected = nn.functional.cross_entropy(
        model(batch["input_ids"])["logits"].reshape(-1, 5), batch["labels"].reshape(-1)
    )
    trainer_stub = SimpleNamespace(device=torch.device("cpu"))
    assert torch.allclose(Trainer._default_loss(trainer_stub, model, batch), expected)
    pipeline = runner.TrainRunner(lambda: None, "test", str(tmp_path), label_smoothing=0)
    assert torch.allclose(pipeline.build_compute_loss(trainer_stub, model)(model, batch), expected)
    without_labels = Trainer._default_loss(trainer_stub, model, {"input_ids": batch["input_ids"]})
    expected_without_labels = nn.functional.cross_entropy(
        model(batch["input_ids"])["logits"][:, :-1].reshape(-1, 5),
        batch["input_ids"][:, 1:].reshape(-1),
    )
    assert torch.allclose(without_labels, expected_without_labels)
