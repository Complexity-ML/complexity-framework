from __future__ import annotations

from scripts.promote_reasoning_sft_checkpoint import promote


def _reasoning(path: str, accuracy: float) -> dict:
    return {
        "model": path,
        "combined": {
            "native_accuracy": accuracy,
            "flexible_accuracy": accuracy,
            "strict_format_rate": 0.8,
        },
    }


def _zero(path: str, accuracy: float) -> dict:
    return {"checkpoint": path, "combined": {"acc": accuracy, "acc_norm": accuracy}}


def test_promotes_final_when_reasoning_improves_and_retention_holds(tmp_path) -> None:
    first = str(tmp_path / "step_001000")
    final = str(tmp_path / "step_004724")
    candidates = [
        {
            "checkpoint": first,
            "step": 1000,
            "piqa_acc_norm": 0.694,
            "matched_eval_loss": 1.03,
        },
        {
            "checkpoint": final,
            "step": 4724,
            "piqa_acc_norm": 0.693,
            "matched_eval_loss": 0.97,
        },
    ]
    summary = {"selected": candidates[0], "candidates": candidates}
    promoted, _, _ = promote(
        summary,
        [_reasoning(first, 0.25), _reasoning(final, 0.35)],
        [_zero(first, 0.49), _zero(final, 0.49)],
        {"combined": {"acc": 0.50}},
    )
    assert promoted["selected"]["step"] == 4724
    assert promoted["initial_piqa_selected"]["step"] == 1000


def test_rejects_reasoning_gain_that_destroys_arc_retention(tmp_path) -> None:
    first = str(tmp_path / "step_001000")
    final = str(tmp_path / "step_004724")
    candidates = [
        {
            "checkpoint": first,
            "step": 1000,
            "piqa_acc_norm": 0.694,
            "matched_eval_loss": 1.03,
        },
        {
            "checkpoint": final,
            "step": 4724,
            "piqa_acc_norm": 0.693,
            "matched_eval_loss": 0.97,
        },
    ]
    summary = {"selected": candidates[0], "candidates": candidates}
    promoted, _, _ = promote(
        summary,
        [_reasoning(first, 0.25), _reasoning(final, 0.50)],
        [_zero(first, 0.49), _zero(final, 0.40)],
        {"combined": {"acc": 0.50}},
    )
    assert promoted["selected"]["step"] == 1000
