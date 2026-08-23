from __future__ import annotations

import hashlib
import json

from scripts import prepare_arc_eval_samples as module


def test_materialize_writes_pinned_complete_arc_samples(tmp_path, monkeypatch) -> None:
    seen = []

    def fake_load_dataset(dataset, config, *, split, revision):
        seen.append((dataset, config, split, revision))
        return [
            {
                "id": f"{config}-{index}",
                "question": f"Question {index}?",
                "choices": {"label": ["A", "B"], "text": ["one", "two"]},
                "answerKey": "B",
            }
            for index in range(3)
        ]

    monkeypatch.setattr(module, "load_dataset", fake_load_dataset)
    manifest = module.materialize(tmp_path)

    assert manifest["revision"] == module.ARC_REVISION
    assert [entry[1] for entry in seen] == ["ARC-Easy", "ARC-Challenge"]
    for task in ("arc_easy", "arc_challenge"):
        path = tmp_path / f"samples_{task}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 3
        assert rows[0]["doc_id"] == 0
        assert rows[0]["doc"]["answerKey"] == "B"
        assert manifest["files"][task]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
