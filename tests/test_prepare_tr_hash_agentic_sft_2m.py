import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from complexity.training.sft_examples import (
    TOOLS,
    agentic_record,
    calculator_values,
    code_record,
    evaluate_arithmetic,
    final_completion,
    generated_code_record,
    generated_constraint_record,
    generated_reasoning_record,
    load_historical_recipe,
    tool_call_completion,
    tools_system,
    validate_agentic_trajectory,
)


def test_code_record_removes_only_leading_legacy_think_wrapper() -> None:
    row = {
        "id": "code-1",
        "messages": [
            {"role": "user", "content": "Write a Python add function."},
            {
                "role": "assistant",
                "content": "<think></think>\n```python\ndef add(a, b):\n    return a + b\n```",
            },
        ],
    }

    record = code_record(row, repository="example/code", subset="train")

    assert record is not None
    assert "<think>" not in record["completion"]
    assert record["category"] == "python_code"


def test_agentic_generator_uses_short_dynamic_tool_contract() -> None:
    for index, subtype in enumerate(
        (
            "single_tool_end_to_end",
            "multi_tool_end_to_end",
            "tool_failure_and_recovery",
            "reasoning_code_and_tools",
        )
    ):
        record = agentic_record(index, subtype)
        trajectory = record["trajectory"]
        system = trajectory[0]["content"]

        assert system.startswith("Available tools:\n[")
        assert "MUST" not in system
        assert "at least 3" not in system
        assert trajectory[-1]["role"] == "assistant"
        assert "<|final_start|>" in trajectory[-1]["content"]
        assert trajectory[-1]["content"].endswith("<|end_of_turn|>")


def test_every_generated_tool_call_is_valid_and_offered() -> None:
    for index in range(30):
        subtype = (
            "single_tool_end_to_end",
            "multi_tool_end_to_end",
            "tool_failure_and_recovery",
            "reasoning_code_and_tools",
        )[index % 4]
        trajectory = agentic_record(index, subtype)["trajectory"]
        offered = {name for name in TOOLS if f'"name":"{name}"' in trajectory[0]["content"]}
        calls = []
        for message in trajectory:
            content = message["content"]
            if "<|tool_call_start|>" not in content:
                continue
            payload = content.split("<|tool_call_start|>", 1)[1].split("<|tool_call_end|>", 1)[0]
            calls.append(json.loads(payload))

        assert calls
        assert all(call["name"] in offered for call in calls)
        assert all(isinstance(call["arguments"], dict) for call in calls)
        validate_agentic_trajectory(trajectory)


def test_trajectory_validator_rejects_wrong_calculator_result() -> None:
    trajectory = agentic_record(0, "single_tool_end_to_end")["trajectory"]
    trajectory[3]["content"] = "999999"

    try:
        validate_agentic_trajectory(trajectory)
    except ValueError as error:
        assert "calculator result" in str(error)
    else:
        raise AssertionError("invalid calculator trajectory was accepted")


def test_generated_reasoning_and_constraints_are_deterministic_and_unique() -> None:
    reasoning = [generated_reasoning_record(index) for index in range(100)]
    constraints = [generated_constraint_record(index) for index in range(100)]
    code = [generated_code_record(index) for index in range(100)]

    assert len({record["prompt"] for record in reasoning}) == 100
    assert len({record["prompt"] for record in constraints}) == 100
    assert len({record["prompt"] for record in code}) == 100
    assert generated_reasoning_record(42) == generated_reasoning_record(42)
    assert generated_constraint_record(42) == generated_constraint_record(42)
    assert generated_code_record(42) == generated_code_record(42)


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("wrong_tool", "offered schema"),
        ("wrong_argument", "offered type"),
        ("orphan_result", "preceding call"),
        ("wrong_result_name", "result name"),
        ("duplicate_end", "terminal end_of_turn"),
        ("malformed_offered", "valid JSON"),
    ],
)
def test_validator_rejects_broken_tool_contract_and_result_order(mutation, error):
    trajectory = deepcopy(agentic_record(0, "single_tool_end_to_end")["trajectory"])
    if mutation in {"wrong_tool", "wrong_argument"}:
        payload = (
            trajectory[2]["content"].split("<|tool_call_start|>")[1].split("<|tool_call_end|>")[0]
        )
        call = json.loads(payload)
        if mutation == "wrong_tool":
            call["name"] = "unoffered_tool"
        else:
            call["arguments"]["expression"] = 123
        trajectory[2]["content"] = (
            "<|tool_call_start|>" + json.dumps(call) + "<|tool_call_end|><|end_of_turn|>"
        )
    elif mutation == "orphan_result":
        trajectory.insert(4, {"role": "tool", "content": "unexpected extra result"})
    elif mutation == "wrong_result_name":
        trajectory[3]["name"] = "date_time"
    elif mutation == "duplicate_end":
        trajectory[-1]["content"] += "<|end_of_turn|>"
    else:
        trajectory[0]["content"] = "Available tools:\n[broken]"
    with pytest.raises(ValueError, match=error):
        validate_agentic_trajectory(trajectory)


def test_non_calculator_results_use_an_explicit_ground_truth_validator():
    trajectory = agentic_record(1, "single_tool_end_to_end")["trajectory"]
    expected = json.loads(trajectory[3]["content"])["owner"]
    check = {
        "search_knowledge_base": lambda arguments, result: (
            json.loads(result).get("owner") == expected
        )
    }
    validate_agentic_trajectory(trajectory, tool_result_validators=check)
    trajectory[3]["content"] = json.dumps({"owner": "forged-owner"})
    with pytest.raises(ValueError, match="tool result is invalid"):
        validate_agentic_trajectory(trajectory, tool_result_validators=check)


@pytest.mark.parametrize(
    "expression", ["1/0", "10**1000000000", "__import__('os').system('false')", "True+2"]
)
def test_calculator_rejects_unsafe_or_unbounded_expressions(expression):
    with pytest.raises(ValueError):
        evaluate_arithmetic(expression)


def test_historical_2m_recipe_retains_source_pins_and_separate_generator_seeds():
    recipe = load_historical_recipe(
        Path(__file__).parents[1] / "configs/examples/historical_sft_2m.json"
    )
    assert sum(recipe["train_quotas"].values()) == 2000000
    assert sum(recipe["eval_quotas"].values()) == 100000
    assert recipe["sources"]["numina"]["revision"] == "1b05109f9e5c1ad06c0663519502416c30b300f8"
    assert (
        recipe["protected_benchmarks"]["piqa"]["sha256"]
        == "54d32a04f59a7e354396f321723c8d7ec35cc6b08506563d8d1ffcc15ce98ddd"
    )
    assert recipe["generator_seed_bases"] == {
        "reasoning": 20000000,
        "constraints": 30000000,
        "code": 35000000,
        "agentic": 40000000,
    }
    assert recipe["release_eligible"] is False


def test_code_projection_leaves_its_source_record_unchanged():
    row = {
        "id": "original",
        "messages": [
            {"role": "user", "content": "Write an add function."},
            {"role": "assistant", "content": "<think></think>\ndef add(a,b):\n    return a+b"},
        ],
    }
    before = deepcopy(row)
    assert code_record(row, repository="example/code", subset="train") is not None
    assert row == before


def test_calculator_validates_archived_integer_division_results_and_rejects_nan():
    expression, answer = calculator_values(3)
    trajectory = [
        {"role": "system", "content": tools_system(["calculator"])},
        {"role": "user", "content": f"Compute {expression}."},
        {
            "role": "assistant",
            "content": tool_call_completion("calculator", {"expression": expression}),
        },
        {"role": "tool", "content": answer},
        {"role": "assistant", "content": final_completion(answer)},
    ]
    assert evaluate_arithmetic(expression) == int(answer)
    validate_agentic_trajectory(trajectory)
    trajectory[3]["content"] = "NaN"
    with pytest.raises(ValueError, match="calculator result"):
        validate_agentic_trajectory(trajectory)


def test_example_helpers_do_not_import_training_or_dataset_runtimes():
    program = """
import importlib.abc, sys
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','datasets','pyarrow','tokenizers','huggingface_hub'}:
            raise RuntimeError('unexpected runtime import: ' + fullname)
sys.meta_path.insert(0, BlockRuntime())
from complexity.training.sft_examples import agentic_record, validate_agentic_trajectory
validate_agentic_trajectory(agentic_record(42,'reasoning_code_and_tools')['trajectory'])
assert 'torch' not in sys.modules
print('pure-helpers-ok')
"""
    result = subprocess.run([sys.executable, "-c", program], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "pure-helpers-ok"
