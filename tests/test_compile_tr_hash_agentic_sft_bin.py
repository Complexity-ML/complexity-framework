import json
from pathlib import Path

import pytest

from complexity.tokenizer import Tokenizer
from scripts.compile_tr_hash_agentic_sft_bin import encode_agentic_trajectory
from scripts.train_tr_hash_agentic_tokenizer import train_tokenizer


@pytest.fixture()
def agentic_tokenizer(tmp_path: Path) -> Tokenizer:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    rows = [
        {
            "text": (
                f"Task {index}: calculate values, call a tool, inspect its result, "
                "and provide a concise verified answer."
            )
        }
        for index in range(300)
    ]
    (corpus / "agentic.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output = tmp_path / "tokenizer"
    train_tokenizer(corpus, output, vocab_size=384, min_frequency=1)
    return Tokenizer.load(str(output))


def test_full_trajectory_supervises_each_assistant_action_only(
    agentic_tokenizer: Tokenizer,
) -> None:
    tokenizer = agentic_tokenizer
    record = {
        "trajectory": [
            {
                "role": "system",
                "content": 'Available tools:\n[{"name":"calculator"}]',
            },
            {
                "role": "user",
                "content": "A workshop packs 17 boxes of 24 parts and removes 85.",
            },
            {
                "role": "assistant",
                "content": (
                    '<|tool_call_start|>{"name":"calculator","arguments":'
                    '{"expression":"17*24-85"}}<|tool_call_end|><|end_of_turn|>'
                ),
            },
            {"role": "tool", "content": "323"},
            {
                "role": "assistant",
                "content": "<|final_start|>323<|final_end|><|end_of_turn|>",
            },
        ]
    }

    encoded = encode_agentic_trajectory(tokenizer, record, seq_len=256)
    labels = encoded["labels"].tolist()
    supervised = {label for label in labels if label != -100}

    assert {7, 8, 9, 18, 19}.issubset(supervised)
    assert 10 not in supervised
    assert 11 not in supervised
    assert 4 not in supervised
    assert 5 not in supervised
    assert 6 not in supervised


def test_full_trajectory_rejects_context_truncation(agentic_tokenizer: Tokenizer) -> None:
    tokenizer = agentic_tokenizer
    record = {
        "trajectory": [
            {"role": "user", "content": "Explain this carefully. " * 100},
            {
                "role": "assistant",
                "content": "<|final_start|>Done.<|final_end|><|end_of_turn|>",
            },
        ]
    }

    try:
        encode_agentic_trajectory(tokenizer, record, seq_len=32)
    except ValueError as error:
        assert "exceeds context without truncation" in str(error)
    else:
        raise AssertionError("overlength trajectory was silently truncated")
