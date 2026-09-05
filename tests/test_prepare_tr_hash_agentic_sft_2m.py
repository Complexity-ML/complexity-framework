import json

from scripts.prepare_tr_hash_agentic_sft_2m import (
    TOOLS,
    agentic_record,
    code_record,
    generated_code_record,
    generated_constraint_record,
    generated_reasoning_record,
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
