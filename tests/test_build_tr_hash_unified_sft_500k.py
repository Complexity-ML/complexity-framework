import json
from pathlib import Path

from scripts.build_tr_hash_unified_sft_500k import build_train, combine_eval


def _row(source: str, index: int) -> str:
    return json.dumps(
        {
            "messages": [
                {"role": "user", "content": f"question {source} {index}"},
                {
                    "role": "assistant",
                    "content": (
                        "<|think_start|>reason<|think_end|>"
                        "<|final_start|>answer<|final_end|>"
                    ),
                },
            ],
            "source": source,
            "capability": "test",
            "reasoning_format": "tr-hash-think-final-32004-v1",
        },
        sort_keys=True,
    )


def _write(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_build_train_keeps_general_and_non_math_then_samples_math(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "train.jsonl"
    rows = [
        _row("smoltalk_magpie_ultra", 0),
        _row("luciole_stem_new", 0),
        *[_row("numina_math_15_validated", index) for index in range(5)],
    ]
    _write(source, rows)

    result = build_train(source, output, target_examples=4, seed=7)

    built = [json.loads(line) for line in output.read_text().splitlines()]
    assert result["examples"] == 4
    assert result["general_examples"] == 1
    assert result["reasoning_non_math_examples"] == 1
    assert result["reasoning_math_examples"] == 2
    assert {row["source"] for row in built} == {
        "smoltalk_magpie_ultra",
        "luciole_stem_new",
        "numina_math_15_validated",
    }


def test_build_train_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    rows = [
        _row("smoltalk_magpie_ultra", 0),
        _row("luciole_stem_new", 0),
        *[_row("openr1_math_verified_new", index) for index in range(8)],
    ]
    _write(source, rows)
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    build_train(source, first, target_examples=5, seed=42)
    build_train(source, second, target_examples=5, seed=42)

    assert first.read_bytes() == second.read_bytes()


def test_combine_eval_preserves_both_inputs(tmp_path: Path) -> None:
    general = tmp_path / "general.jsonl"
    reasoning = tmp_path / "reasoning.jsonl"
    output = tmp_path / "eval.jsonl"
    _write(general, [_row("smoltalk_magpie_ultra", 1)])
    _write(reasoning, [_row("luciole_stem_new", 2)])

    result = combine_eval(general, reasoning, output, seed=3)

    assert result["examples"] == 2
    assert len(output.read_text().splitlines()) == 2
