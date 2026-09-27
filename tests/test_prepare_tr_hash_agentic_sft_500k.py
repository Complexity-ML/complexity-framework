import json

from tokenizers import Tokenizer

from scripts.prepare_tr_hash_agentic_sft_500k import (
    EVAL_QUOTAS,
    TRAIN_QUOTAS,
    arithmetic_example,
    collect_calculator,
    thinking_completion,
)


def test_500k_recipe_counts_and_thinking_budget() -> None:
    assert sum(TRAIN_QUOTAS.values()) == 500_000
    assert sum(EVAL_QUOTAS.values()) == 25_000
    assert TRAIN_QUOTAS["base_clean"] == 228_184
    assert TRAIN_QUOTAS["verified_math_reasoning"] == 60_000
    assert TRAIN_QUOTAS["verified_arithmetic_reasoning"] == 75_000


def test_arithmetic_examples_are_internally_consistent() -> None:
    for index in range(100):
        question, reasoning, answer = arithmetic_example(index)
        assert question
        assert answer.lstrip("$") in reasoning
        completion = thinking_completion(reasoning, answer)
        assert completion.count("<|think_start|>") == 1
        assert completion.count("<|think_end|>") == 1
        assert completion.count("<|final_start|>") == 1
        assert completion.count("<|final_end|>") == 1


def test_calculator_records_use_native_tool_contract() -> None:
    tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
    records = collect_calculator(tokenizer)
    call = records["calculator_tool_call"][0].record
    final = records["calculator_tool_final"][0].record
    payload = call["completion"].split("<|tool_call_start|>", 1)[1].split(
        "<|tool_call_end|>", 1
    )[0]
    assert json.loads(payload)["name"] == "calculator"
    assert "<|tool_result_start|>" in final["prompt"]
    assert final["completion"].startswith("<|final_start|>")
