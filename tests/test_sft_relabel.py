from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from complexity.inference.chat_template import default_chat_template
from complexity.training.sft_relabel import (
    ALL_ASSISTANT_PROJECTION,
    SHARD_FORMAT_V2,
    relabel_dataset,
    relabel_example,
)


class TinyTokenizer:
    eos_token_id = 0
    marker = [7, 8]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert text == "Assistant:\n"
        assert add_special_tokens is False
        return list(self.marker)


def test_relabel_example_supervises_previous_and_final_assistant_turns() -> None:
    # user, marker, answer-1, EOS, user, marker, final-answer
    inputs = np.asarray([1, 7, 8, 20, 21, 0, 2, 7, 8, 30], dtype=np.int64)
    labels = np.asarray(
        [-100, -100, -100, -100, -100, -100, -100, -100, 30, 0],
        dtype=np.int64,
    )

    updated, supervised = relabel_example(
        inputs,
        labels,
        assistant_marker=[7, 8],
        eos_token_id=0,
    )

    assert updated.tolist() == [
        -100,
        -100,
        20,
        21,
        0,
        -100,
        -100,
        -100,
        30,
        0,
    ]
    assert supervised == 5


def test_relabel_dataset_reuses_inputs_and_writes_v2_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source-shard"
    train = source / "train"
    train.mkdir(parents=True)
    template = default_chat_template()
    template["training_projection"] = "naturalize_card_hand_preserve_assistant_turns"
    (source / "chat_template.json").write_text(json.dumps(template), encoding="utf-8")
    inputs = np.asarray([1, 7, 8, 20, 0, 2, 7, 8, 30], dtype="<u4")
    labels = np.asarray([-100, -100, -100, -100, -100, -100, -100, 30, 0], dtype="<i4")
    inputs.tofile(train / "input_ids.bin")
    labels.tofile(train / "labels.bin")
    example = {"example_id": "one", "offset": 0, "num_tokens": len(inputs)}
    (train / "examples.jsonl").write_text(json.dumps(example) + "\n", encoding="utf-8")
    (train / "sft.idx.json").write_text(
        json.dumps(
            {
                "format": "complexity-sft-token-shard-v1",
                "chat_template_id": template["id"],
                "examples": 1,
                "num_tokens": len(inputs),
                "supervised_tokens": 2,
                "eos_token_id": 0,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "32k-v17"

    result = relabel_dataset(
        source,
        output,
        tokenizer=TinyTokenizer(),
        skip_content_verification=True,
    )

    metadata = json.loads((output / "train" / "sft.idx.json").read_text())
    output_template = json.loads((output / "chat_template.json").read_text())
    output_labels = np.fromfile(output / "train" / "labels.bin", dtype="<i4")
    assert result["train"]["changed_labels"] == 2
    assert output_labels.tolist() == [-100, -100, 20, 0, -100, -100, -100, 30, 0]
    assert metadata["format"] == SHARD_FORMAT_V2
    assert metadata["assistant_supervision"] == "all_assistant_turns"
    assert metadata["content_verification"] == "skipped_unchanged"
    assert output_template["training_projection"] == ALL_ASSISTANT_PROJECTION
    assert (source / "train" / "labels.bin").read_bytes() != (
        output / "train" / "labels.bin"
    ).read_bytes()
