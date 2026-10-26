import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models

from complexity.training.sft_examples import (
    agentic_category,
    load_historical_recipe,
    normalize_record,
)

RECIPE = load_historical_recipe(
    Path(__file__).parents[1] / "configs/examples/historical_sft_250k.json"
)
TRAIN_QUOTAS, EVAL_QUOTAS = RECIPE["train_quotas"], RECIPE["eval_quotas"]
TRAIN_THINK_QUOTAS, EVAL_THINK_QUOTAS = RECIPE["train_think_quotas"], RECIPE["eval_think_quotas"]


def test_250k_recipe_counts_and_conditional_thinking() -> None:
    assert sum(TRAIN_QUOTAS.values()) == 250_000
    assert sum(EVAL_QUOTAS.values()) == 12_500
    assert sum(TRAIN_THINK_QUOTAS.values()) == 25_000
    assert sum(EVAL_THINK_QUOTAS.values()) == 1_250
    assert TRAIN_QUOTAS == {
        "direct": 100_000,
        "tool_call": 90_000,
        "no_call": 35_000,
        "tool_final": 25_000,
    }


def test_agentic_direct_turns_become_no_call() -> None:
    assert agentic_category({"category": "direct"}) == "no_call"
    assert agentic_category({"category": "tool_call"}) == "tool_call"


def test_reasoning_can_be_kept_or_removed_without_forcing_it() -> None:
    vocab = {
        token: index
        for index, token in enumerate(
            [
                "<unk>",
                "<|think_start|>",
                "reason",
                "<|think_end|>",
                "<|final_start|>",
                "answer",
                "<|final_end|>",
                "<|end_of_turn|>",
            ]
        )
    }
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    source = {
        "prompt": "prompt",
        "completion": (
            "<|think_start|>reason<|think_end|><|final_start|>answer<|final_end|><|end_of_turn|>"
        ),
        "source_id": "example",
    }
    kept = normalize_record(source, category="no_call", thinking=True, tokenizer=tokenizer)
    stripped = normalize_record(source, category="no_call", thinking=False, tokenizer=tokenizer)
    assert "<|think_start|>" in kept["completion"]
    assert "<|think_start|>" not in stripped["completion"]
    assert stripped["completion"].startswith("<|final_start|>")
    assert json.dumps(stripped)


@pytest.mark.parametrize(
    "completion",
    [
        "<|think_start|>reason<|final_start|>answer<|final_end|><|end_of_turn|>",
        "<|final_start|>answer<|final_end|><|end_of_turn|><|end_of_turn|>",
        "<|tool_call_start|>null<|tool_call_end|><|end_of_turn|>",
    ],
)
def test_normalization_rejects_incomplete_or_repeated_control_markers(completion):
    tokenizer = Tokenizer(models.WordLevel({"<unk>": 0}, unk_token="<unk>"))
    with pytest.raises(ValueError):
        normalize_record(
            {"prompt": "prompt", "completion": completion, "source_id": "one"},
            category="no_call",
            thinking=False,
            tokenizer=tokenizer,
        )


def test_native_projection_owns_exactly_one_eos_even_with_tokenizer_postprocessing():
    from tokenizers import pre_tokenizers, processors

    from complexity.training.sft_examples import token_count
    from complexity.training.sft_preparation import encode_native_sft_record

    words = [
        "</s>",
        "<pad>",
        "<s>",
        "<unk>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|end_of_turn|>",
        "<|tool_call_start|>",
        "<|tool_call_end|>",
        "<|tool_result_start|>",
        "<|tool_result_end|>",
        "<|think_start|>",
        "<|think_end|>",
        "<|final_start|>",
        "<|final_end|>",
        "prompt",
        "reason",
        "answer",
    ]
    engine = Tokenizer(
        models.WordLevel({word: index for index, word in enumerate(words)}, unk_token="<unk>")
    )
    engine.add_special_tokens(words[:16])
    engine.pre_tokenizer = pre_tokenizers.Whitespace()
    engine.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 2), ("</s>", 0)]
    )

    class NativeTokenizer:
        _tokenizer = engine
        eos_token_id = 0
        pad_token_id = 1

    source = {
        "prompt": "<|user|>prompt<|end_of_turn|><|assistant|>",
        "completion": "<|think_start|>reason<|think_end|><|final_start|>answer<|final_end|><|end_of_turn|>",
        "source_id": "native-postprocessed",
    }
    record = normalize_record(
        source, category="no_call", thinking=False, tokenizer=NativeTokenizer()
    )
    expected = (
        sum(
            len(engine.encode(record[field], add_special_tokens=False).ids)
            for field in ("prompt", "completion")
        )
        + 1
    )
    assert record["token_count"] == expected
    batch = encode_native_sft_record(NativeTokenizer(), record, 32)
    supervised = batch["labels"][batch["labels"] != -100].tolist()
    assert supervised[-1] == 0
    assert supervised.count(0) == 1
    assert 0 not in batch["input_ids"].tolist()
    assert batch["input_ids"][0].item() == 5
    with pytest.raises(ValueError, match="already contain EOS"):
        token_count(
            NativeTokenizer(), record["prompt"], "<|final_start|></s><|final_end|><|end_of_turn|>"
        )
