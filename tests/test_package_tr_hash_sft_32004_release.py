from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from scripts.package_tr_hash_sft_32004_release import (
    audit_raw_partition,
    audit_refinement_checkpoint,
)
from scripts.recompile_tr_hash_sft_32004 import (
    FINAL_END,
    FINAL_START,
    FORMAT_ID,
    THINK_END,
    THINK_START,
)


def test_raw_audit_requires_one_closed_envelope_per_assistant_turn(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    row = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": THINK_START + THINK_END + FINAL_START + "Hi" + FINAL_END,
            },
        ],
        "reasoning_format": FORMAT_ID,
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    report = audit_raw_partition(path)
    assert report["examples"] == 1
    assert report["assistant_turns"] == 1
    assert set(report["special_token_text_counts"].values()) == {1}


def test_refinement_audit_requires_32004_tied_embedding_rows(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"vocab_size": 32_004, "tie_word_embeddings": True}),
        encoding="utf-8",
    )
    save_file(
        {"embed_tokens.weight": torch.zeros(32_004, 2)},
        tmp_path / "model.safetensors",
    )
    report = audit_refinement_checkpoint(tmp_path)
    assert report["embedding_shape"] == [32_004, 2]
    assert report["tie_word_embeddings"] is True
