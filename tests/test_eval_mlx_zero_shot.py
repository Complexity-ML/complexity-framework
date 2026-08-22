from __future__ import annotations

import sys
from types import ModuleType


def _import_evaluator_without_mlx(monkeypatch):
    mlx = ModuleType("mlx")
    mlx_core = ModuleType("mlx.core")
    mlx.core = mlx_core
    mlx_lm = ModuleType("mlx_lm")
    mlx_lm_utils = ModuleType("mlx_lm.utils")
    mlx_lm_utils.load = object()
    mlx_lm_utils.load_model = object()
    mlx_lm_utils.load_tokenizer = object()
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", mlx_lm_utils)
    sys.modules.pop("scripts.eval_mlx_zero_shot", None)
    from scripts import eval_mlx_zero_shot

    return eval_mlx_zero_shot


def test_arc_easy_matches_lm_eval_zero_shot_prompt_contract(monkeypatch) -> None:
    evaluator = _import_evaluator_without_mlx(monkeypatch)
    row = {
        "id": "Mercury_SC_401169",
        "question": "Which object is attracted to a magnet?",
        "choices": {
            "label": ["A", "B", "C", "D"],
            "text": ["a paper clip", "a rubber band", "a leaf", "a glass"],
        },
        "answerKey": "A",
    }
    monkeypatch.setattr(
        evaluator,
        "load_dataset",
        lambda path, name, split: [row],
    )

    examples = evaluator.load_arc_easy(maximum=None)

    assert examples == [
        evaluator.Example(
            benchmark="arc_easy",
            example_id="Mercury_SC_401169",
            context="Question: Which object is attracted to a magnet?\nAnswer:",
            continuations=(
                " a paper clip",
                " a rubber band",
                " a leaf",
                " a glass",
            ),
            answer=0,
        )
    ]


def test_arc_easy_limit_is_applied_after_formatting(monkeypatch) -> None:
    evaluator = _import_evaluator_without_mlx(monkeypatch)
    rows = [
        {
            "id": str(index),
            "question": f"Question {index}?",
            "choices": {"label": ["1", "2"], "text": ["yes", "no"]},
            "answerKey": 2,
        }
        for index in range(3)
    ]
    monkeypatch.setattr(
        evaluator,
        "load_dataset",
        lambda path, name, split: rows,
    )

    examples = evaluator.load_arc_easy(maximum=2)

    assert [example.example_id for example in examples] == ["0", "1"]
    assert [example.answer for example in examples] == [1, 1]
