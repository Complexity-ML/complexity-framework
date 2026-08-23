from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.prepare_tr_hash_200m_reasoning_sft_500m import (
    _numina_math_15,
    _restore_completed_rows,
    _truncate_to_offset,
    _write_build_state,
    benchmark_overlap,
    load_recipe,
    normalize_for_dedup,
    normalized_conversation_digest,
)


def test_production_recipe_is_exactly_500m_and_revision_pinned() -> None:
    recipe = load_recipe(Path("configs/tr_hash_200m_reasoning_sft_500m.json"))
    assert sum(source["train_token_target"] for source in recipe["sources"]) == 500_000_000
    assert recipe["sequence_length"] == 2_048
    assert recipe["sources"][0]["local_jsonl"] == "${REPLAY_JSONL}"
    assert recipe["sources"][0]["trusted_tokenized_replay"] is True
    assert recipe["sources"][0]["expected_train_examples"] == 300_000
    assert recipe["sources"][-1]["fill_to_total"] is True
    for source in recipe["sources"][1:]:
        assert len(source["revision"]) == 40
    assert {item["name"] for item in recipe["protected_benchmarks"]} == {
        "arc_easy",
        "arc_challenge",
        "piqa",
        "gsm8k",
        "hellaswag",
    }
    piqa = next(item for item in recipe["protected_benchmarks"] if item["name"] == "piqa")
    assert len(piqa["archive_sha256"]) == 64


def test_recipe_rejects_any_non_500m_quota(tmp_path: Path) -> None:
    recipe = json.loads(
        Path("configs/tr_hash_200m_reasoning_sft_500m.json").read_text(encoding="utf-8")
    )
    recipe["sources"][0]["train_token_target"] -= 1
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(recipe), encoding="utf-8")
    with pytest.raises(ValueError, match="sum to 499999999"):
        load_recipe(path)


def test_numina_adapter_fails_closed_on_validation_flags() -> None:
    valid = {
        "problem": "What is 2 + 2?",
        "solution": "Adding the terms gives 4.",
        "problem_is_valid": "Yes",
        "solution_is_valid": "Yes",
    }
    assert _numina_math_15(valid) == [
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "Adding the terms gives 4."},
    ]
    assert _numina_math_15({**valid, "solution_is_valid": "No"}) is None
    assert _numina_math_15({**valid, "problem_is_valid": None}) is None


def test_normalized_dedup_collapses_case_spacing_and_unicode() -> None:
    left = [{"role": "user", "content": "Café:  2 + 2?"}]
    right = [{"role": "user", "content": "CAFE\u0301 — 2+2 ?"}]
    assert normalize_for_dedup(left[0]["content"]) == normalize_for_dedup(right[0]["content"])
    assert normalized_conversation_digest(left) == normalized_conversation_digest(right)


def test_benchmark_guard_catches_exact_and_wrapped_long_questions() -> None:
    protected = {
        normalize_for_dedup(
            "A long protected benchmark question whose wording must never enter the "
            "training mixture because it would invalidate the held-out evaluation."
        )
    }
    exact = [{"role": "user", "content": next(iter(protected))}]
    wrapped = [
        {
            "role": "user",
            "content": "Solve carefully: " + next(iter(protected)) + " Answer now.",
        }
    ]
    clean = [{"role": "user", "content": "Prove that there are infinitely many primes."}]
    assert benchmark_overlap(exact, protected)
    assert benchmark_overlap(wrapped, protected)
    assert not benchmark_overlap(clean, protected)


def test_source_checkpoint_is_atomic_and_restores_exact_rows(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl.partial"
    evaluation = tmp_path / "eval.jsonl.partial"
    first = {"messages": [{"role": "user", "content": "First"}]}
    second = {"messages": [{"role": "user", "content": "Second"}]}
    with (
        train.open("w", encoding="utf-8") as train_handle,
        evaluation.open("w", encoding="utf-8") as eval_handle,
    ):
        train_handle.write(json.dumps(first) + "\n")
        _write_build_state(
            path=tmp_path / ".build_state.json",
            recipe_sha256="a" * 64,
            train_handle=train_handle,
            eval_handle=eval_handle,
            source_report={"source-a": {"train_examples": 1}},
        )
        saved_offset = train_handle.tell()
        train_handle.write(json.dumps(second) + "\n")
    state = json.loads((tmp_path / ".build_state.json").read_text())
    assert state["train_byte_offset"] == saved_offset
    _truncate_to_offset(train, saved_offset)
    exact, normalized = _restore_completed_rows((train, evaluation))
    assert len(exact) == len(normalized) == 1
    assert "Second" not in train.read_text()


def test_atomic_state_keeps_explicit_source_order_when_json_keys_are_sorted(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.json"
    with (
        (tmp_path / "train").open("w", encoding="utf-8") as train_handle,
        (tmp_path / "eval").open("w", encoding="utf-8") as eval_handle,
    ):
        _write_build_state(
            path=state_path,
            recipe_sha256="b" * 64,
            train_handle=train_handle,
            eval_handle=eval_handle,
            source_report={"z-first": {}, "a-second": {}},
        )
    state = json.loads(state_path.read_text())
    assert state["completed_sources"] == ["z-first", "a-second"]
    assert list(state["source_report"]) == ["a-second", "z-first"]
