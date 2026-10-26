from __future__ import annotations

from pathlib import Path

import pytest

from complexity.training.tensorboard import finite_float, mirror_once


class FakeWriter:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []
        self.flushes = 0

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars.append((tag, value, step))

    def flush(self) -> None:
        self.flushes += 1


def test_finite_float_rejects_empty_and_non_finite_values():
    assert finite_float("") is None
    assert finite_float("nan") is None
    assert finite_float("inf") is None
    assert finite_float("2.5") == 2.5


def test_mirror_once_writes_new_scalar_points_only(tmp_path: Path):
    metrics = tmp_path / "metrics.csv"
    metrics.write_text(
        "step,train_loss,train_ppl,matched_eval_loss,matched_eval_ppl,lr,"
        "expert_lr,tok_s,supervised_tokens,min_label,max_label,bad_labels,"
        "matched_eval_tokens,natural_eval_loss,natural_eval_ppl,natural_eval_tokens\n"
        "0,,,3.0,20.0,1e-5,,,,,,,1000,,,\n"
        "10,2.0,7.4,,,9e-6,2e-6,56000,2000,1,31999,0,,,,\n",
        encoding="utf-8",
    )
    writer = FakeWriter()
    seen: set[tuple[int, str]] = set()

    first = mirror_once(metrics, writer, seen)
    second = mirror_once(metrics, writer, seen)

    assert first == 10
    assert second == 0
    assert ("Loss/eval", 3.0, 0) in writer.scalars
    assert ("Loss/train", 2.0, 10) in writer.scalars
    assert ("Throughput/tokens_per_second", 56000.0, 10) in writer.scalars
    assert writer.flushes == 1


def test_mirror_retries_partial_cells_without_duplicating_valid_scalars(tmp_path: Path):
    metrics = tmp_path / "metrics.csv"
    metrics.write_text("step,train_loss,lr\n2,1.25,1e-\n", encoding="utf-8")
    writer = FakeWriter()
    seen = set()
    assert mirror_once(metrics, writer, seen) == 1
    metrics.write_text("step,train_loss,lr\n2,1.25,1e-5\n", encoding="utf-8")
    assert mirror_once(metrics, writer, seen) == 1
    assert writer.scalars == [("Loss/train", 1.25, 2), ("LR", 1e-5, 2)]


def test_sft_jsonl_metrics_count_step_zero_and_retry_uncommitted_line(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    complete = '{"step":0,"loss":2.5,"gradient_norm":1.5,"seconds":0.25}\n'
    metrics.write_text(complete + '{"step":1,"loss":', encoding="utf-8")
    writer, seen = FakeWriter(), set()
    assert mirror_once(metrics, writer, seen) == 3
    metrics.write_text(complete + '{"step":1,"loss":2.0}\n', encoding="utf-8")
    assert mirror_once(metrics, writer, seen) == 1
    assert writer.scalars[-1] == ("Loss/train", 2.0, 1)


def test_metrics_reject_malformed_committed_json(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step":\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        mirror_once(path, FakeWriter(), set())
