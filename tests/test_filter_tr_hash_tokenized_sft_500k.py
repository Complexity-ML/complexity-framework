from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.filter_tr_hash_tokenized_sft_500k import (
    SPECIAL_TOKEN_IDS,
    materialize_partition,
    resolve_selected_records,
)


def _write_source(root: Path, rows: list[dict]) -> tuple[Path, Path]:
    raw = root / "train.jsonl"
    partition = root / "tokenized/train"
    partition.mkdir(parents=True)
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    inputs: list[int] = []
    labels: list[int] = []
    index: list[dict] = []
    for number, row in enumerate(rows):
        segment = [10 + number, *SPECIAL_TOKEN_IDS.values()]
        offset = len(inputs)
        inputs.extend(segment)
        labels.extend([-100, *SPECIAL_TOKEN_IDS.values()])
        index.append(
            {
                "example_id": f"train-{number:06d}",
                "task": row["capability"],
                "source": row["source"],
                "capability": row["capability"],
                "offset": offset,
                "num_tokens": len(segment),
                "prompt_tokens": 1,
                "supervised_tokens": 4,
            }
        )
    np.asarray(inputs, dtype="<u4").tofile(partition / "input_ids.bin")
    np.asarray(labels, dtype="<i4").tofile(partition / "labels.bin")
    (partition / "examples.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in index), encoding="utf-8"
    )
    (partition / "sft.idx.json").write_text(json.dumps({"examples": len(rows)}), encoding="utf-8")
    return raw, partition


def test_resolve_selected_records_preserves_selected_order(tmp_path: Path) -> None:
    rows = [
        {"messages": [1], "source": "a", "capability": "instruction"},
        {"messages": [2], "source": "b", "capability": "reasoning"},
    ]
    raw, partition = _write_source(tmp_path / "source", rows)
    selected = tmp_path / "selected.jsonl"
    selected.write_text(json.dumps(rows[1]) + "\n" + json.dumps(rows[0]) + "\n")

    resolved = resolve_selected_records(selected, [(raw, partition)])

    assert [record[1]["source"] for record in resolved] == ["b", "a"]


def test_materialize_partition_copies_tokens_and_rebuilds_offsets(tmp_path: Path) -> None:
    rows = [
        {"messages": [1], "source": "a", "capability": "instruction"},
        {"messages": [2], "source": "b", "capability": "reasoning"},
    ]
    raw, partition = _write_source(tmp_path / "source", rows)
    selected = tmp_path / "selected.jsonl"
    selected.write_text(json.dumps(rows[1]) + "\n" + json.dumps(rows[0]) + "\n")

    metadata = materialize_partition(selected, [(raw, partition)], tmp_path / "output")

    assert metadata["examples"] == 2
    assert metadata["special_token_label_counts"] == {token: 2 for token in SPECIAL_TOKEN_IDS}
    output_index = [
        json.loads(line) for line in (tmp_path / "output/examples.jsonl").read_text().splitlines()
    ]
    assert [item["offset"] for item in output_index] == [0, 5]
    output_inputs = np.fromfile(tmp_path / "output/input_ids.bin", dtype="<u4")
    assert output_inputs.tolist()[:6] == [11, 32000, 32001, 32002, 32003, 10]
