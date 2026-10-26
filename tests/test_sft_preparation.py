"""Local preparation contracts, independent of historical production quotas."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer as RawTokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from complexity.tokenizer import Tokenizer
from complexity.tokenizer.config import TokenizerConfig, get_special_tokens
from complexity.training.sft import SFTTokenShardDataset
from complexity.training.sft_preparation import (
    compile_sft_dataset,
    direct_completion,
    encode_agentic_trajectory,
    load_sft_recipe,
    mini_projected_record,
    parse_when2call_response,
    prepare_sft_dataset,
)
from complexity.training.sft_shard import (
    FINAL_ASSISTANT_SUPERVISION,
    LEGACY_ALL_ASSISTANT_SUPERVISION,
)


@pytest.fixture()
def tiny_tokenizer(tmp_path: Path):
    markers = get_special_tokens("tr_hash_agentic_reasoning")
    vocab = {word: index for index, word in enumerate([*markers, "a", "b", "c", "answer"])}
    raw = RawTokenizer(WordLevel(vocab=vocab, unk_token="<|unk|>"))
    raw.pre_tokenizer = Whitespace()
    raw.add_special_tokens(markers)
    tokenizer = Tokenizer(
        raw, TokenizerConfig(format="tr_hash_agentic_reasoning", vocab_size=len(vocab))
    )
    path = tmp_path / "tokenizer"
    tokenizer.save(path)
    # Portable tokenizer export uses the HF filename rather than model config.
    (path / "tokenizer_config.json").write_text(
        json.dumps({"bos_token": "<|begin|>", "eos_token": "<|end|>", "pad_token": "<|pad|>"})
    )
    return tokenizer, path


def row(prompt: str, source_id: str):
    return {
        "prompt": f"<|user|>{prompt}<|end_of_turn|><|assistant|>",
        "completion": direct_completion("answer"),
        "source_id": source_id,
    }


def recipe(train=1, validation=1):
    return {
        "format": "complexity-sft-preparation-v1",
        "sequence_length": 64,
        "quotas": {"train": {"local": train}, "validation": {"local": validation}},
    }


def write_partitions(root, records):
    root.mkdir()
    for filename in ("train.jsonl", "eval.jsonl"):
        (root / filename).write_text("".join(json.dumps(record) + "\n" for record in records))


def test_native_preparation_and_shard_reader_roundtrip(tiny_tokenizer, tmp_path):
    tokenizer, token_root = tiny_tokenizer
    prepared, compiled = tmp_path / "prepared", tmp_path / "compiled"
    manifest = prepare_sft_dataset(
        {"train": {"local": [row("a", "train")]}, "validation": {"local": [row("b", "eval")]}},
        recipe(),
        tokenizer,
        prepared,
    )
    assert manifest["release_eligible"] is False
    compiled_manifest = compile_sft_dataset(
        prepared,
        token_root,
        compiled,
        sequence_length=64,
        assistant_supervision=FINAL_ASSISTANT_SUPERVISION,
    )
    dataset = SFTTokenShardDataset(compiled, sequence_length=64, vocab_size=tokenizer.vocab_size)
    example = dataset.examples[0]
    length = example["num_tokens"]
    assert dataset.labels[length - 1] == tokenizer.eos_token_id
    assert dataset.labels[0] == -100
    assert example["supervised_tokens"] == 5  # final_start, answer, final_end, EOT, EOS
    assert compiled_manifest["partitions"]["train"]["history_assistant_turns"] == "masked_context"
    with pytest.raises(FileExistsError):
        compile_sft_dataset(
            prepared, token_root, compiled, assistant_supervision=FINAL_ASSISTANT_SUPERVISION
        )


def test_validation_selected_before_train_and_duplicates_rejected(tiny_tokenizer, tmp_path):
    tokenizer, _ = tiny_tokenizer
    manifest = prepare_sft_dataset(
        {
            "train": {
                "local": [row("b", "dup-prompt"), row("c", "same-source"), row("a", "train")]
            },
            "validation": {"local": [row("b", "same-source")]},
        },
        recipe(),
        tokenizer,
        tmp_path / "data",
    )
    assert manifest["rejected"]["duplicate"] == 2
    assert json.loads((tmp_path / "data/train.jsonl").read_text())["source_id"] == "train"


def test_exhausted_quota_has_no_output(tiny_tokenizer, tmp_path):
    tokenizer, _ = tiny_tokenizer
    output = tmp_path / "data"
    with pytest.raises(ValueError, match="exhausted"):
        prepare_sft_dataset(
            {"train": {"local": [row("a", "train")]}, "validation": {"local": [row("b", "eval")]}},
            recipe(train=2),
            tokenizer,
            output,
        )
    assert not output.exists()


def test_reasoning_legacy_and_broken_final_targets_are_rejected(tiny_tokenizer, tmp_path):
    tokenizer, _ = tiny_tokenizer
    invalid = [
        {
            **row("a", "reasoning"),
            "completion": "<|think_start|>trace<|think_end|>" + direct_completion("answer"),
        },
        {**row("a", "legacy"), "completion": "<TOOLCALL>[]</TOOLCALL><|end_of_turn|>"},
        {**row("a", "malformed"), "completion": "answer<|end_of_turn|>"},
        {
            **row("a", "null-call"),
            "completion": "<|tool_call_start|>null<|tool_call_end|><|end_of_turn|>",
        },
    ]
    manifest = prepare_sft_dataset(
        {
            "train": {"local": [*invalid, row("a", "train")]},
            "validation": {"local": [row("b", "eval")]},
        },
        recipe(),
        tokenizer,
        tmp_path / "data",
    )
    assert manifest["rejected"] == {"unsupported_target_markers": 2, "invalid_or_overlength": 2}


def test_preparation_never_truncates_to_fill_quota(tiny_tokenizer, tmp_path):
    tokenizer, _ = tiny_tokenizer
    with pytest.raises(ValueError, match="exhausted"):
        prepare_sft_dataset(
            {
                "train": {"local": [row("a " * 100, "train")]},
                "validation": {"local": [row("b", "eval")]},
            },
            recipe(),
            tokenizer,
            tmp_path / "data",
        )
    assert not (tmp_path / "data").exists()


def test_trajectory_final_policy_masks_prior_assistant(tiny_tokenizer):
    tokenizer, _ = tiny_tokenizer
    record = {
        "trajectory": [
            {"role": "user", "content": "a"},
            {
                "role": "assistant",
                "content": "<|tool_call_start|>{}<|tool_call_end|><|end_of_turn|>",
            },
            {"role": "tool", "content": "b"},
            {"role": "assistant", "content": direct_completion("answer")},
        ]
    }
    every = encode_agentic_trajectory(tokenizer, record, 64)
    final = encode_agentic_trajectory(
        tokenizer, record, 64, assistant_supervision=FINAL_ASSISTANT_SUPERVISION
    )
    assert every["input_ids"].equal(final["input_ids"])
    assert 8 in every["labels"].tolist()
    assert 8 not in final["labels"].tolist()
    assert 10 not in every["labels"].tolist()
    assert 10 not in final["labels"].tolist()
    assert final["labels"][final["labels"] != -100][-1] == tokenizer.eos_token_id


def test_encoding_owns_eos_even_with_existing_tokenizer_postprocessor(tiny_tokenizer):
    from tokenizers.processors import TemplateProcessing

    from complexity.training.sft_preparation import encode_native_sft_record

    tokenizer, _ = tiny_tokenizer
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single="<|begin|> $A <|end|>",
        special_tokens=[("<|begin|>", 0), ("<|end|>", 1)],
    )
    encoded = encode_native_sft_record(tokenizer, row("a", "id"), 64)
    active = encoded["labels"][encoded["labels"] != -100].tolist()
    assert active == [18, 23, 19, 7, 1]
    assert encoded["input_ids"][0] == 5


def test_compile_trajectory_policy_is_explicit_and_historical_reader_rejects_all(
    tiny_tokenizer, tmp_path
):
    tokenizer, token_root = tiny_tokenizer
    source = tmp_path / "source"
    write_partitions(
        source,
        [
            {
                "trajectory": [
                    {"role": "user", "content": "a"},
                    {"role": "assistant", "content": direct_completion("answer")},
                ]
            }
        ],
    )
    output = tmp_path / "all-assistant"
    compile_sft_dataset(
        source,
        token_root,
        output,
        sequence_length=64,
        assistant_supervision=LEGACY_ALL_ASSISTANT_SUPERVISION,
    )
    with pytest.raises(ValueError, match="final_assistant_only"):
        SFTTokenShardDataset(output, sequence_length=64, vocab_size=tokenizer.vocab_size)
    final_output = tmp_path / "final-only"
    compile_sft_dataset(
        source,
        token_root,
        final_output,
        sequence_length=64,
        assistant_supervision=FINAL_ASSISTANT_SUPERVISION,
    )
    assert (
        SFTTokenShardDataset(
            final_output, sequence_length=64, vocab_size=tokenizer.vocab_size
        ).examples[0]["supervised_tokens"]
        == 5
    )


def test_source_hash_mismatch_aborts_compilation(tiny_tokenizer, tmp_path):
    _, token_root = tiny_tokenizer
    source = tmp_path / "source"
    write_partitions(source, [row("a", "id")])
    (source / "dataset_info.json").write_text(
        json.dumps({"artifacts": {"train": {"sha256": "wrong"}}})
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        compile_sft_dataset(
            source,
            token_root,
            tmp_path / "compiled",
            assistant_supervision=FINAL_ASSISTANT_SUPERVISION,
        )
    assert not (tmp_path / "compiled").exists()


def test_compile_no_truncation_failure_leaves_no_output(tiny_tokenizer, tmp_path):
    _, token_root = tiny_tokenizer
    source = tmp_path / "source"
    write_partitions(source, [row("a " * 100, "id")])
    with pytest.raises(ValueError, match="exceeds context without truncation"):
        compile_sft_dataset(
            source,
            token_root,
            tmp_path / "compiled",
            sequence_length=8,
            assistant_supervision=FINAL_ASSISTANT_SUPERVISION,
        )
    assert not (tmp_path / "compiled").exists()


def test_wrong_marker_ids_are_rejected_before_output(tiny_tokenizer, tmp_path):
    _, token_root = tiny_tokenizer
    raw = json.loads((token_root / "tokenizer.json").read_text())
    raw["model"]["vocab"]["<|system|>"], raw["model"]["vocab"]["<|user|>"] = 5, 4
    (token_root / "tokenizer.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="marker IDs"):
        compile_sft_dataset(
            tmp_path / "absent",
            token_root,
            tmp_path / "compiled",
            assistant_supervision=FINAL_ASSISTANT_SUPERVISION,
        )
    assert not (tmp_path / "compiled").exists()


def test_common_data_cli_compiles_a_readable_sft_shard(tiny_tokenizer, tmp_path):
    from typer.testing import CliRunner

    from complexity.cli.app import app

    tokenizer, token_root = tiny_tokenizer
    source, output = tmp_path / "source", tmp_path / "compiled"
    write_partitions(source, [row("a", "id")])
    argv = [
        "data",
        "sft",
        str(source),
        str(output),
        "--tokenizer",
        str(token_root),
        "--assistant-supervision",
        FINAL_ASSISTANT_SUPERVISION,
        "--sequence-length",
        "64",
    ]
    runner = CliRunner()
    result = runner.invoke(app, argv)
    assert result.exit_code == 0, result.output
    manifest = json.loads(result.stdout)
    assert manifest["partitions"]["train"]["examples"] == 1
    assert manifest["release_eligible"] is False
    loaded = SFTTokenShardDataset(output, sequence_length=64, vocab_size=tokenizer.vocab_size)
    assert loaded.supervised_tokens > 0
    before = (output / "manifest.json").read_bytes()
    result = runner.invoke(app, argv)
    assert result.exit_code == 1
    assert (output / "manifest.json").read_bytes() == before


def test_recipe_quota_rejects_boolean_and_keeps_historical_100k(tmp_path):
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe(train=True)))
    with pytest.raises(ValueError, match="positive integer"):
        load_sft_recipe(path)
    historical = load_sft_recipe(
        Path(__file__).resolve().parents[1] / "configs/examples/agentic_sft_balanced_100k.json"
    )
    assert sum(historical["quotas"]["train"].values()) == 100_000
    assert sum(historical["quotas"]["validation"].values()) == 5_000
    assert historical["release_eligible"] is False


def test_legacy_parser_never_downgrades_broken_call_to_answer():
    assert parse_when2call_response("<TOOLCALL>[]") is None
    assert parse_when2call_response("prefix<TOOLCALL>[]</TOOLCALL>") is None


def test_mini_projection_malformed_function_is_rejected():
    assert (
        mini_projected_record(
            {
                "messages": [
                    {"role": "user", "content": "a"},
                    {"role": "assistant", "tool_calls": [{"name": "x", "arguments": {}}]},
                ],
                "tools": [{"function": None}],
            }
        )
        is None
    )


def test_trajectory_rejects_context_markers_in_assistant_body(tiny_tokenizer):
    tokenizer, _ = tiny_tokenizer
    with pytest.raises(ValueError, match="context-role markers"):
        encode_agentic_trajectory(
            tokenizer,
            {"trajectory": [{"role": "assistant", "content": "<|user|>a<|end_of_turn|>"}]},
            64,
        )
