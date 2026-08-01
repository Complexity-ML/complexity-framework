from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from complexity.inference.chat_template import (
    CHAT_TEMPLATE_ID,
    default_chat_template,
    render_inference_prompt,
)
from scripts.sft_100m_o200k_tr_local import (
    SFTBinDataset,
    build_parser,
    configure_trainable_parameters,
    format_record,
    update_early_stopping,
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


def test_sft_can_freeze_token_input_and_output_parameters() -> None:
    class ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(16, 8)
            self.layers = torch.nn.Linear(8, 8)
            self.lm_head = torch.nn.Linear(8, 16, bias=False)

    model = ToyModel()
    stats = configure_trainable_parameters(model, freeze_token_io=True)

    assert model.embed_tokens.weight.requires_grad is False
    assert model.lm_head.weight.requires_grad is False
    assert model.layers.weight.requires_grad is True
    assert stats["token_io_frozen"] is True
    assert stats["frozen"] == 16 * 8 * 2
    assert stats["trainable"] > 0


def test_sft_early_stopping_state_resets_only_on_real_improvement() -> None:
    improved, best, misses = update_early_stopping(
        4.0,
        2,
        3.98,
        min_delta=0.01,
    )
    assert improved is True
    assert best == 3.98
    assert misses == 0

    improved, best, misses = update_early_stopping(
        best,
        misses,
        3.975,
        min_delta=0.01,
    )
    assert improved is False
    assert best == 3.98
    assert misses == 1


def test_sft_parser_exposes_conservative_training_controls() -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "checkpoint",
            "--freeze-token-io",
            "--save-best",
            "--early-stopping-patience",
            "3",
        ]
    )
    assert args.freeze_token_io is True
    assert args.eval_at_start is True
    assert args.save_best is True
    assert args.early_stopping_patience == 3
