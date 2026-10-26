import json
import random
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from complexity.training.sft_examples import (
    arithmetic_example,
    collect_calculator,
    load_historical_recipe,
    thinking_completion,
)

ROOT = Path(__file__).parents[1]
RECIPE = load_historical_recipe(ROOT / "configs/examples/historical_sft_500k.json")
TRAIN_QUOTAS, EVAL_QUOTAS = RECIPE["train_quotas"], RECIPE["eval_quotas"]


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
    tokenizer = Tokenizer.from_file(str(ROOT / "tokenizer/tokenizer.json"))
    records = collect_calculator(tokenizer, count=2)
    call = records["calculator_tool_call"][0].record
    final = records["calculator_tool_final"][0].record
    payload = call["completion"].split("<|tool_call_start|>", 1)[1].split("<|tool_call_end|>", 1)[0]
    assert json.loads(payload)["name"] == "calculator"
    assert "<|tool_result_start|>" in final["prompt"]
    assert final["completion"].startswith("<|final_start|>")


def test_historical_recipe_preserves_exact_pins_and_explicit_budgets(tmp_path):
    assert RECIPE["historical_reference"] is True
    assert RECIPE["release_eligible"] is False
    assert RECIPE["tokenizer"]["revision"] == "2fcbc2c5359ded0244ca14531f1b3806eebac55e"
    assert RECIPE["sources"]["base"]["revision"] == "c6697cc1bc48a086e47fefabcd326ba371500f86"
    assert RECIPE["generator_seed_bases"] == {"arithmetic": 500000, "calculator": 900000}
    bad = json.loads(json.dumps(RECIPE))
    bad["sources"]["base"]["revision"] = "main"
    path = tmp_path / "unpinned.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="immutable"):
        load_historical_recipe(path)


def test_generators_use_local_rng_and_caller_sized_collections():
    state = random.getstate()
    assert arithmetic_example(42) == arithmetic_example(42)
    tokenizer = Tokenizer.from_file(str(ROOT / "tokenizer/tokenizer.json"))
    records = collect_calculator(tokenizer, count=2)
    assert all(len(group) == 2 for group in records.values())
    assert random.getstate() == state
    with pytest.raises(ValueError, match="Insufficient valid"):
        collect_calculator(tokenizer, count=1, max_tokens=2, max_attempts=2)


@pytest.mark.parametrize("count", [0, -1, True])
def test_calculator_collection_rejects_invalid_explicit_budget(count):
    with pytest.raises(ValueError, match="count"):
        collect_calculator(None, count=count)
