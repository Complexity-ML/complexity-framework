from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from complexity.inference.chat_template import (
    CHAT_TEMPLATE_ID,
    default_chat_template,
    render_inference_prompt,
)
from scripts.sft_100m_o200k_tr_local import (
    SFTBinDataset,
    build_parser,
    format_record,
)


def _write_shard(root: Path) -> None:
    train = root / "train"
    train.mkdir(parents=True)
    (root / "chat_template.json").write_text(
        json.dumps(default_chat_template()) + "\n",
        encoding="utf-8",
    )
    input_ids = np.asarray([10, 11, 12, 13, 20, 21], dtype="<u4")
    labels = np.asarray([-100, -100, 12, 13, -100, 21], dtype="<i4")
    input_ids.tofile(train / "input_ids.bin")
    labels.tofile(train / "labels.bin")
    examples = [
        {
            "example_id": "first",
            "task": "test",
            "offset": 0,
            "num_tokens": 4,
            "supervised_tokens": 2,
        },
        {
            "example_id": "second",
            "task": "test",
            "offset": 4,
            "num_tokens": 2,
            "supervised_tokens": 1,
        },
    ]
    with (train / "examples.jsonl").open("w") as handle:
        for example in examples:
            handle.write(json.dumps(example) + "\n")
    (train / "sft.idx.json").write_text(
        json.dumps(
            {
                "format": "complexity-sft-token-shard-v1",
                "chat_template_id": CHAT_TEMPLATE_ID,
                "examples": 2,
                "num_tokens": 6,
                "supervised_tokens": 3,
                "eos_token_id": 199999,
            }
        )
    )


def test_sft_bin_dataset_reads_and_pads_indexed_examples(tmp_path: Path) -> None:
    _write_shard(tmp_path)
    dataset = SFTBinDataset(tmp_path, seq_len=5, seed=42, rank=0, world_size=1)
    first = dataset._tensor_example(dataset.examples[0])
    assert first["input_ids"].tolist() == [10, 11, 12, 13, 199999]
    assert first["labels"].tolist() == [-100, -100, 12, 13, -100]
    assert dataset.chat_template["id"] == CHAT_TEMPLATE_ID


def test_messages_use_the_canonical_chat_template() -> None:
    template = default_chat_template()
    prompt, completion = format_record(
        {
            "messages": [
                {"role": "user", "content": "What is a hash route?"},
                {"role": "assistant", "content": "A fixed mapping."},
            ]
        },
        template,
    )
    assert prompt == render_inference_prompt("What is a hash route?", template)
    assert completion == "A fixed mapping."


def test_sft_bin_dataset_truncates_from_left_to_keep_response(tmp_path: Path) -> None:
    _write_shard(tmp_path)
    dataset = SFTBinDataset(tmp_path, seq_len=2, seed=42, rank=0, world_size=1)
    first = dataset._tensor_example(dataset.examples[0])
    assert first["input_ids"].tolist() == [12, 13]
    assert first["labels"].tolist() == [12, 13]


def test_sft_bin_eval_iterator_is_finite(tmp_path: Path) -> None:
    _write_shard(tmp_path)
    dataset = SFTBinDataset(
        tmp_path,
        seq_len=5,
        seed=42,
        rank=0,
        world_size=1,
        repeat=False,
    )
    assert len(list(dataset)) == 2


def test_sft_parser_rejects_two_dataset_sources() -> None:
    parser = build_parser()
    try:
        parser.parse_args(
            [
                "--checkpoint",
                "checkpoint",
                "--jsonl",
                "data.jsonl",
                "--sft-bin",
                "data-bin",
            ]
        )
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("JSONL and SFT bin inputs must be mutually exclusive")
