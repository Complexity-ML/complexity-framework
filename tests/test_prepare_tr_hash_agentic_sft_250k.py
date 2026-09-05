import json

from tokenizers import Tokenizer, models

from scripts.prepare_tr_hash_agentic_sft_250k import (
    EVAL_QUOTAS,
    EVAL_THINK_QUOTAS,
    TRAIN_QUOTAS,
    TRAIN_THINK_QUOTAS,
    agentic_category,
    normalize_record,
)


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
