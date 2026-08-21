from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

SCRIPT = Path("scripts/select_sft_v2_checkpoint.py")


def load_module():
    spec = importlib.util.spec_from_file_location("select_sft_v2_checkpoint", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_fixture(
    root: Path,
    metrics: Path,
    checkpoints: Path,
    rows: list[tuple[int, float, float, float, bool]],
) -> None:
    metrics.parent.mkdir(parents=True, exist_ok=True)
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("step", "matched_eval_loss", "matched_eval_ppl"),
        )
        writer.writeheader()
        for step, _acc, _acc_norm, loss, _passed in rows:
            writer.writerow(
                {
                    "step": step,
                    "matched_eval_loss": loss,
                    "matched_eval_ppl": 3.0,
                }
            )
    for epoch, (step, acc, acc_norm, _loss, passed) in enumerate(rows, start=1):
        report = root / f"epoch_{epoch:02d}_step_{step:06d}"
        report.mkdir(parents=True)
        (report / "piqa.json").write_text(
            json.dumps(
                {
                    "checkpoint_step": step,
                    "benchmarks": {
                        "piqa": {
                            "acc": acc,
                            "acc_norm": acc_norm,
                            "correct": round(acc * 1838),
                            "correct_norm": round(acc_norm * 1838),
                        }
                    },
                }
            )
        )
        (report / "promotion.json").write_text(
            json.dumps({"passed": passed, "failures": {} if passed else {"code": ["failed"]}})
        )
        checkpoint = checkpoints / f"step_{step:06d}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "checkpoint.pt").write_bytes(b"checkpoint")


def test_selection_excludes_failed_gate_and_prioritizes_piqa_norm(tmp_path) -> None:
    module = load_module()
    evaluation_root = tmp_path / "eval"
    metrics = tmp_path / "metrics.csv"
    checkpoints = tmp_path / "checkpoints"
    write_fixture(
        evaluation_root,
        metrics,
        checkpoints,
        [
            (100, 0.70, 0.70, 1.3, True),
            (200, 0.75, 0.75, 1.2, False),
            (300, 0.71, 0.72, 1.4, True),
        ],
    )

    summary = module.build_summary(evaluation_root, metrics, checkpoints)

    assert summary["release_ready"] is True
    assert summary["selected"]["step"] == 300
    assert summary["candidates"][1]["promotion_passed"] is False


def test_selection_uses_eval_loss_as_final_tie_breaker(tmp_path) -> None:
    module = load_module()
    evaluation_root = tmp_path / "eval"
    metrics = tmp_path / "metrics.csv"
    checkpoints = tmp_path / "checkpoints"
    write_fixture(
        evaluation_root,
        metrics,
        checkpoints,
        [
            (100, 0.70, 0.71, 1.3, True),
            (200, 0.70, 0.71, 1.1, True),
            (300, 0.69, 0.70, 1.0, True),
        ],
    )

    summary = module.build_summary(evaluation_root, metrics, checkpoints)

    assert summary["selected"]["step"] == 200


def test_selection_blocks_release_when_every_gate_fails(tmp_path) -> None:
    module = load_module()
    evaluation_root = tmp_path / "eval"
    metrics = tmp_path / "metrics.csv"
    checkpoints = tmp_path / "checkpoints"
    write_fixture(
        evaluation_root,
        metrics,
        checkpoints,
        [
            (100, 0.70, 0.71, 1.3, False),
            (200, 0.71, 0.72, 1.2, False),
            (300, 0.72, 0.73, 1.1, False),
        ],
    )

    summary = module.build_summary(evaluation_root, metrics, checkpoints)

    assert summary["release_ready"] is False
    assert summary["selected"] is None
