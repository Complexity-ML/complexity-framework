from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from complexity.training.sft_shard import (
    FINAL_ASSISTANT_SUPERVISION,
    MASKED_ASSISTANT_HISTORY,
    SHARD_FORMAT_V2,
)
from scripts.tokenize_luciole_16way_sft import materialize_partition


class _Tokenizer:
    eos_token_id = 2
    pad_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [3 + (ord(character) % 17) for character in text]


def test_materialized_luciole_shard_masks_prompt_and_preserves_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "train.jsonl"
    source.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ],
                "source": "test_source",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    metadata = materialize_partition(
        source,
        tmp_path / "train",
        tokenizer=_Tokenizer(),
        seq_len=64,
        min_completion_tokens=8,
    )

    assert metadata["format"] == SHARD_FORMAT_V2
    assert metadata["assistant_supervision"] == FINAL_ASSISTANT_SUPERVISION
    assert metadata["history_assistant_turns"] == MASKED_ASSISTANT_HISTORY
    assert metadata["examples"] == 1
    labels = np.fromfile(tmp_path / "train" / "labels.bin", dtype="<i4")
    assert np.any(labels == -100)
    assert np.any(labels != -100)
    example = json.loads(
        (tmp_path / "train" / "examples.jsonl").read_text(encoding="utf-8")
    )
    assert example["task"] == "test_source"
    assert example["source"] == "test_source"
