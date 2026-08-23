from __future__ import annotations

import json

import pytest

from scripts.merge_arc_generative_shards import merge


def test_merge_arc_shards_is_complete_and_recomputes_metrics(tmp_path) -> None:
    shards = []
    for index, (task, correct) in enumerate((("arc_easy", True), ("arc_challenge", False))):
        report = tmp_path / f"shard{index}.json"
        report.write_text(
            json.dumps(
                {
                    "backend": "tr_hash_torch",
                    "combined": {},
                    "benchmarks": {},
                    "elapsed_seconds": index + 1,
                    "shard": {"index": index, "count": 2},
                }
            )
        )
        row = {
            "task": task,
            "doc_id": index,
            "strict_prediction": "A" if correct else "B",
            "flexible_prediction": "A" if correct else "B",
            "native_prediction": "A" if correct else "B",
            "strict_correct": correct,
            "flexible_correct": correct,
            "native_correct": correct,
        }
        report.with_suffix(".jsonl").write_text(json.dumps(row) + "\n")
        shards.append(report)
    output = tmp_path / "merged.json"
    result = merge(shards, output, 2)
    assert result["combined"]["strict_accuracy"] == 0.5
    assert result["benchmarks"]["arc_easy"]["strict_accuracy"] == 1.0
    assert result["shards"] == 2
    assert len(output.with_suffix(".jsonl").read_text().splitlines()) == 2


def test_merge_arc_shards_rejects_incomplete_probe(tmp_path) -> None:
    report = tmp_path / "shard.json"
    report.write_text(json.dumps({"elapsed_seconds": 1}))
    report.with_suffix(".jsonl").write_text(
        json.dumps(
            {
                "task": "arc_easy",
                "doc_id": 1,
                "strict_prediction": None,
                "flexible_prediction": None,
                "native_prediction": None,
                "strict_correct": False,
                "flexible_correct": False,
                "native_correct": False,
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="expected 2 examples"):
        merge([report], tmp_path / "merged.json", 2)
