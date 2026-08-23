from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from complexity.tokenizer import Tokenizer
from complexity.training.sft_shard import SHARD_FORMAT_V2
from scripts.sft_500m_32k_tr import encode_sft_example
from scripts.tokenize_tr_hash_200m_clean_sft_v2 import (
    REASONING_SPECIAL_TOKEN_IDS,
    encode_complete_example,
    materialize_partition,
    materialize_release,
    tokenizer_aligned_chat_template,
    validate_reasoning_tokenizer,
)

TOKENIZER = Path("artifacts/hf/tr-hash-tokenizer-32k-canonical-vocab32004")


def _record() -> dict:
    return {
        "messages": [
            {"role": "user", "content": "What is 17 times 23?"},
            {"role": "assistant", "content": "17 × 23 = 391."},
        ],
        "source": "verified-math",
        "capability": "verified_math",
    }


def test_complete_encoder_uses_32004_ids_and_preserves_all_labels() -> None:
    tokenizer = Tokenizer.load(str(TOKENIZER))
    inputs, labels, _, completion_tokens = encode_complete_example(
        tokenizer,
        _record(),
        sequence_length=2048,
    )

    assert len(tokenizer) == 32_004
    assert int(inputs.max()) < 32_004
    assert int(np.count_nonzero(labels != -100)) == completion_tokens


def test_fast_complete_encoder_matches_reference_sft_encoding() -> None:
    tokenizer = Tokenizer.load(str(TOKENIZER))
    template = tokenizer_aligned_chat_template(tokenizer)
    inputs, labels, _, _ = encode_complete_example(
        tokenizer,
        _record(),
        sequence_length=2048,
        chat_template=template,
    )
    reference = encode_sft_example(
        tokenizer,
        _record(),
        2048,
        min_completion_tokens=1,
        chat_template=template,
    )

    np.testing.assert_array_equal(inputs, reference["input_ids"].numpy()[: len(inputs)])
    np.testing.assert_array_equal(labels, reference["labels"].numpy()[: len(labels)])


def test_training_template_uses_the_real_tokenizer_eos_id() -> None:
    tokenizer = Tokenizer.load(str(TOKENIZER))
    template = tokenizer_aligned_chat_template(tokenizer)

    assert template["eos_token"] == "</s>"
    assert tokenizer.encode(template["eos_token"], add_special_tokens=False) == [
        tokenizer.eos_token_id
    ]
    assert tokenizer.encode("<|endoftext|>", add_special_tokens=False) != [tokenizer.eos_token_id]


def test_complete_encoder_fails_instead_of_truncating() -> None:
    tokenizer = Tokenizer.load(str(TOKENIZER))
    record = _record()
    record["messages"][-1]["content"] = "long answer " * 500

    with pytest.raises(ValueError, match="would be truncated"):
        encode_complete_example(tokenizer, record, sequence_length=32)


def test_materialized_partition_is_directly_readable_sft_v2(tmp_path: Path) -> None:
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
    target = tmp_path / "train"

    metadata = materialize_partition(
        source,
        target,
        tokenizer=Tokenizer.load(str(TOKENIZER)),
        sequence_length=2048,
    )

    assert metadata["format"] == SHARD_FORMAT_V2
    assert metadata["examples"] == 1
    assert metadata["vocab_size"] == 32_004
    assert metadata["truncation_policy"] == "fail_closed_no_truncation"
    index = json.loads((target / "examples.jsonl").read_text(encoding="utf-8"))
    assert index["task"] == "verified_math"
    assert index["source"] == "verified-math"
    assert (target / "input_ids.bin").stat().st_size == metadata["num_tokens"] * 4
    assert (target / "labels.bin").stat().st_size == metadata["num_tokens"] * 4


def test_release_names_held_out_partition_eval_for_trainer(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    rendered = json.dumps(_record()) + "\n"
    (source / "train.jsonl").write_text(rendered, encoding="utf-8")
    (source / "eval.jsonl").write_text(rendered, encoding="utf-8")
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "sequence_length": 2048,
                "train_sha256": "train-hash",
                "eval_sha256": "eval-hash",
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "tokenized"

    manifest = materialize_release(source, TOKENIZER, output)

    assert (output / "eval" / "sft.idx.json").is_file()
    assert not (output / "validation").exists()
    assert "eval" in manifest["partitions"]


def test_reasoning_release_propagates_and_verifies_exact_token_count(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    rendered = json.dumps(_record()) + "\n"
    (source / "train.jsonl").write_text(rendered, encoding="utf-8")
    (source / "eval.jsonl").write_text(rendered, encoding="utf-8")
    tokenizer = Tokenizer.load(str(TOKENIZER))
    inputs, _, _, _ = encode_complete_example(tokenizer, _record(), sequence_length=2048)
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "sequence_length": 2048,
                "train_sha256": "train-hash",
                "eval_sha256": "eval-hash",
                "nominal_target_unique_formatted_tokens": 500_000_000,
                "actual_unique_formatted_tokens": len(inputs),
                "token_quota_overshoot": 7,
            }
        ),
        encoding="utf-8",
    )

    manifest = materialize_release(source, TOKENIZER, tmp_path / "tokenized")

    assert manifest["actual_unique_formatted_tokens"] == len(inputs)
    assert manifest["nominal_target_unique_formatted_tokens"] == 500_000_000
    assert manifest["token_quota_overshoot"] == 7


def test_reasoning_tokenizer_contract_is_exact() -> None:
    tokenizer = Tokenizer.load(str(TOKENIZER))
    validate_reasoning_tokenizer(tokenizer)
    assert {
        token: tokenizer.encode(token, add_special_tokens=False)[0]
        for token in REASONING_SPECIAL_TOKEN_IDS
    } == REASONING_SPECIAL_TOKEN_IDS


def test_materialized_partition_counts_all_four_supervised_markers(tmp_path: Path) -> None:
    tokenizer = Tokenizer.load(str(TOKENIZER))
    record = _record()
    record["messages"][-1]["content"] = (
        "<|think_start|><|think_end|><|final_start|>391<|final_end|>"
    )
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")
    metadata = materialize_partition(
        source,
        tmp_path / "train",
        tokenizer=tokenizer,
        sequence_length=2048,
    )
    assert metadata["special_token_label_counts"] == {
        token: 1 for token in REASONING_SPECIAL_TOKEN_IDS
    }
