from __future__ import annotations

import argparse
import json

from scripts.eval_arc_generative import (
    ARCExample,
    apply_native_recipe,
    build_open_answer_prompt,
    build_prompt,
    evenly_spaced,
    load_lm_eval_samples,
    parse_flexible_answer,
    parse_open_answer,
    parse_strict_answer,
)


def test_load_lm_eval_samples_remaps_numeric_labels_to_letters(tmp_path) -> None:
    path = tmp_path / "samples.jsonl"
    path.write_text(
        json.dumps(
            {
                "doc_id": 9,
                "doc": {
                    "id": "example",
                    "question": "What is 2 + 2?",
                    "choices": {
                        "label": ["1", "2", "3", "4"],
                        "text": ["3", "4", "5", "6"],
                    },
                    "answerKey": "2",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    examples = load_lm_eval_samples(path, "arc_easy")

    assert examples[0].labels == ("A", "B", "C", "D")
    assert examples[0].answer == "B"


def test_prompt_requires_one_explicit_final_answer() -> None:
    example = ARCExample(
        task="arc_challenge",
        doc_id=1,
        example_id="one",
        question="Which force attracts objects to Earth?",
        choices=("gravity", "friction", "electricity", "magnetism"),
        answer_index=0,
    )

    prompt = build_prompt(example)

    assert "A. gravity" in prompt
    assert "Final answer: X" in prompt
    assert "A, B, C, D" in prompt
    assert "After that line" not in prompt


def test_tr_hash_reasoning_checkpoint_ranking_is_greedy() -> None:
    args = argparse.Namespace(
        backend="tr_hash_torch",
        max_new_tokens=None,
        max_thinking_tokens=None,
        temperature=None,
        top_p=None,
        top_k=None,
        repetition_penalty=None,
    )

    apply_native_recipe(args)

    assert args.temperature == 0.0
    assert args.top_p == 1.0
    assert args.top_k == 0


def test_direct_prompt_for_non_reasoning_model_is_concise() -> None:
    example = ARCExample(
        task="arc_easy",
        doc_id=2,
        example_id="two",
        question="What is frozen water?",
        choices=("steam", "ice", "rain", "fog"),
        answer_index=1,
    )

    prompt = build_prompt(example, reasoning=False)

    assert "Reply only with" in prompt
    assert "Think step by step" not in prompt


def test_bare_prompt_contains_only_question_and_labeled_choices() -> None:
    example = ARCExample(
        task="arc_easy",
        doc_id=3,
        example_id="three",
        question="What is frozen water?",
        choices=("steam", "ice", "rain", "fog"),
        answer_index=1,
    )

    prompt = build_prompt(example, prompt_style="bare")

    assert prompt.startswith("Question: What is frozen water?")
    assert "B. ice" in prompt
    assert "Think step by step" not in prompt
    assert "Reply only" not in prompt
    assert "Final answer" not in prompt


def test_open_answer_prompt_has_no_option_letters() -> None:
    example = ARCExample(
        task="arc_easy",
        doc_id=2,
        example_id="two",
        question="What is frozen water?",
        choices=("steam", "ice", "rain", "fog"),
        answer_index=1,
    )

    prompt = build_open_answer_prompt(example)

    assert "- ice" in prompt
    assert "A." not in prompt
    assert "Final answer" not in prompt


def test_open_answer_parser_requires_one_named_choice() -> None:
    example = ARCExample(
        task="arc_easy",
        doc_id=2,
        example_id="two",
        question="What is frozen water?",
        choices=("steam", "ice", "rain", "fog"),
        answer_index=1,
    )

    assert parse_open_answer("Ice", example) == "B"
    assert parse_open_answer("Ice is frozen water.", example) == "B"
    assert parse_open_answer("It could be ice or rain.", example) is None
    assert parse_open_answer("Cold water", example) is None


def test_strict_parser_rejects_missing_or_conflicting_declarations() -> None:
    allowed = ("A", "B", "C", "D")

    assert parse_strict_answer("Reasoning. Final answer: C", allowed) == "C"
    assert parse_strict_answer("I choose C.", allowed) is None
    assert parse_strict_answer("Answer: B. Final answer: C", allowed) is None


def test_flexible_parser_uses_post_thinking_segment_only() -> None:
    text = "A and B are weak. C is strongest.<|think_end|>\nC<|im_end|>"

    assert parse_flexible_answer(text, ("A", "B", "C", "D")) == "C"


def test_evenly_spaced_covers_both_ends() -> None:
    rows = [ARCExample("arc_easy", index, str(index), "q", ("x", "y"), 0) for index in range(10)]

    selected = evenly_spaced(rows, 4)

    assert [row.doc_id for row in selected] == [0, 3, 6, 9]


def test_native_recipes_differ_without_overriding_explicit_values() -> None:
    nautile = argparse.Namespace(
        backend="nautile_torch",
        max_new_tokens=None,
        temperature=None,
        top_p=None,
        top_k=None,
        repetition_penalty=None,
    )
    tr_hash = argparse.Namespace(
        backend="tr_hash_mlx",
        max_new_tokens=128,
        temperature=None,
        top_p=None,
        top_k=None,
        repetition_penalty=None,
    )
    constrained = argparse.Namespace(
        backend="tr_hash_mlx_constrained",
        max_new_tokens=None,
        temperature=None,
        top_p=None,
        top_k=None,
        repetition_penalty=None,
    )
    open_answer = argparse.Namespace(
        backend="tr_hash_mlx_open",
        max_new_tokens=None,
        temperature=None,
        top_p=None,
        top_k=None,
        repetition_penalty=None,
    )

    apply_native_recipe(nautile)
    apply_native_recipe(tr_hash)
    apply_native_recipe(constrained)
    apply_native_recipe(open_answer)

    assert nautile.temperature == 0.15
    assert nautile.top_k == 50
    assert tr_hash.temperature == 0.30
    assert tr_hash.top_k == 30
    assert tr_hash.max_new_tokens == 128
    assert constrained.temperature == 0.0
    assert constrained.max_new_tokens == 1
    assert open_answer.max_new_tokens == 64
    assert open_answer.temperature == 0.30
