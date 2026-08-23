from __future__ import annotations

import json
from pathlib import Path

import pytest

from complexity.tokenizer import Tokenizer
from scripts.recompile_tr_hash_sft_32004 import (
    FINAL_END,
    FINAL_START,
    FORMAT_ID,
    SPECIAL_TOKEN_IDS,
    THINK_END,
    THINK_START,
    _last_boxed,
    make_benchmark_guard,
    recompile_partition,
    render_envelope,
    split_assistant_content,
    validate_enveloped_messages,
    validate_tokenizer_32004,
    wrap_messages,
)

TOKENIZER = Path("artifacts/hf/tr-hash-tokenizer-32k-canonical-vocab32004")


def test_canonical_tokenizer_has_exact_reasoning_ids() -> None:
    tokenizer = Tokenizer.load(str(TOKENIZER))
    validate_tokenizer_32004(tokenizer)
    for token, token_id in SPECIAL_TOKEN_IDS.items():
        assert tokenizer.encode(token, add_special_tokens=False) == [token_id]


def test_balanced_policy_does_not_invent_reasoning_for_plain_answer() -> None:
    envelope = split_assistant_content("Paris is the capital of France.", "instruction")
    assert envelope.reasoning == ""
    assert envelope.final == "Paris is the capital of France."
    assert envelope.extraction == "direct_final"
    assert render_envelope(envelope) == (
        f"{THINK_START}{THINK_END}{FINAL_START}Paris is the capital of France.{FINAL_END}"
    )


def test_math_boxed_answer_keeps_trace_and_distills_final() -> None:
    content = "Compute 17 times 23 as 17(20+3)=340+51=391. Thus \\boxed{391}."
    envelope = split_assistant_content(content, "validated_math_reasoning")
    assert envelope.reasoning == content
    assert envelope.final == r"\boxed{391}"
    assert envelope.extraction == "boxed_math"


def test_balanced_box_parser_handles_nested_latex() -> None:
    assert _last_boxed(r"Therefore \\boxed{\frac{1}{2}}.") == r"\boxed{\frac{1}{2}}"


def test_explicit_legacy_envelope_is_converted_without_old_tags() -> None:
    envelope = split_assistant_content("<think>2 + 2 = 4.</think><final>4</final>", "verified_math")
    assert envelope.reasoning == "2 + 2 = 4."
    assert envelope.final == "4"
    assert envelope.extraction == "source_explicit"


def test_every_assistant_turn_gets_one_ordered_envelope() -> None:
    messages, extractions = wrap_messages(
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Capital?"},
            {"role": "assistant", "content": "Paris"},
        ],
        "instruction",
    )
    counts = validate_enveloped_messages(messages)
    assert extractions == {"direct_final": 2}
    assert counts == {token: 2 for token in SPECIAL_TOKEN_IDS}


def test_recompile_partition_removes_benchmark_overlap_and_tracks_format(
    tmp_path: Path,
) -> None:
    source = tmp_path / "train.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "Protected benchmark question"},
                            {"role": "assistant", "content": "A"},
                        ],
                        "source": "test",
                        "capability": "instruction",
                    }
                ),
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "What is two plus two?"},
                            {"role": "assistant", "content": "Four."},
                        ],
                        "source": "test",
                        "capability": "instruction",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    target = tmp_path / "compiled.jsonl"
    report = recompile_partition(
        source,
        target,
        tokenizer=Tokenizer.load(str(TOKENIZER)),
        sequence_length=2048,
        benchmark_guard=make_benchmark_guard({"protected benchmark question"}),
    )
    assert report["input_examples"] == 2
    assert report["output_examples"] == 1
    assert report["rejected"] == {"benchmark_overlap": 1}
    row = json.loads(target.read_text(encoding="utf-8"))
    assert row["reasoning_format"] == FORMAT_ID
    assert row["messages"][-1]["content"].startswith(THINK_START + THINK_END)


def test_tokenizer_validation_rejects_old_32000_tokenizer() -> None:
    with pytest.raises(ValueError, match="32,004"):
        validate_tokenizer_32004(Tokenizer.load("tokenizer"))
