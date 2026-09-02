import json
from pathlib import Path

import pytest

from complexity.cli.commands.tokenize import _get_format_special_tokens
from complexity.tokenizer import get_special_tokens
from scripts.train_tr_hash_agentic_tokenizer import AGENTIC_SPECIAL_TOKENS, train_tokenizer


def _write_training_corpus(root: Path) -> None:
    rows = []
    for index in range(300):
        rows.append(
            {
                "text": (
                    f"Procedure {index}: POST /v1/jobs/{index} with JSON arguments. "
                    f"```python\nresult_{index} = run_tool({index})\nassert result_{index}.ok\n``` "
                    "Plan the operation, execute it, inspect the result, and verify the final answer."
                )
            }
        )
    (root / "00-agentic.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_cli_and_tokenizer_config_share_agentic_markers() -> None:
    expected = get_special_tokens("tr_hash_agentic_reasoning")
    assert _get_format_special_tokens("tr_hash_agentic_reasoning") == expected
    assert AGENTIC_SPECIAL_TOKENS == expected


def test_agentic_tokenizer_reserves_markers_inside_fixed_vocab(tmp_path: Path) -> None:
    pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_training_corpus(corpus)
    output = tmp_path / "tokenizer"

    manifest = train_tokenizer(corpus, output, vocab_size=384, min_frequency=1)
    tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(output)
    markers = get_special_tokens("tr_hash_agentic_reasoning")

    assert len(tokenizer) == 384
    assert manifest["special_token_ids"] == {marker: index for index, marker in enumerate(markers)}
    assert [tokenizer.convert_tokens_to_ids(marker) for marker in markers] == list(
        range(len(markers))
    )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Run the diagnostic."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert rendered.endswith("<|assistant|><|think_start|>")
    assert (output / "chat_template.jinja").is_file()
    assert (output / "agentic_tokenizer_manifest.json").is_file()


def test_agentic_chat_template_serializes_reasoning_tools_and_final(tmp_path: Path) -> None:
    pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_training_corpus(corpus)
    output = tmp_path / "tokenizer"
    train_tokenizer(corpus, output, vocab_size=384, min_frequency=1)
    tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(output)

    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Follow the user request."},
            {"role": "user", "content": "Add two and three."},
            {
                "role": "assistant",
                "reasoning": "I should use the calculator.",
                "tool_calls": [{"name": "calculator", "arguments": {"expression": "2+3"}}],
                "content": "",
            },
            {"role": "tool", "content": "5"},
            {
                "role": "assistant",
                "reasoning": "The tool returned five.",
                "content": "The answer is 5.",
            },
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Evaluate an arithmetic expression.",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                },
            }
        ],
        tokenize=False,
        add_generation_prompt=False,
    )

    assert rendered.startswith("<|system|>Available tools:\n")
    assert "<|system|>Follow the user request.<|end_of_turn|>" in rendered
    assert "<|user|>Add two and three.<|end_of_turn|>" in rendered
    assert "<|think_start|>I should use the calculator.<|think_end|>" in rendered
    assert "<|tool_call_start|>" in rendered
    assert "<|tool_result_start|>5<|tool_result_end|><|end_of_turn|>" in rendered
    assert (
        "<|think_start|>The tool returned five.<|think_end|>"
        "<|final_start|>The answer is 5.<|final_end|><|end_of_turn|>"
    ) in rendered
