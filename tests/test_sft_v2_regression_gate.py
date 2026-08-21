from __future__ import annotations

import json
from pathlib import Path

from scripts.check_sft_v2_regression import audit_regression, check_response


def _panel() -> dict:
    return json.loads(Path("configs/tr_hash_200m_sft_v2_regression.json").read_text())


def test_panel_covers_code_math_memory_constraints_and_piqa() -> None:
    panel = _panel()
    capabilities = {item["capability"] for item in panel["prompts"]}

    assert {"code", "math", "multi_turn_memory", "instruction_constraints"} <= capabilities
    assert panel["piqa"]["examples"] == 1838
    assert panel["generation"]["temperature"] == 0.0


def test_code_check_executes_required_function_cases() -> None:
    item = next(item for item in _panel()["prompts"] if item["id"] == "code_add")

    assert check_response(item, "```python\ndef add(a, b):\n    return a + b\n```") == []
    assert any(
        "expected" in failure
        for failure in check_response(item, "def add(a, b):\n    return a - b")
    )


def test_code_check_rejects_unsafe_import_and_times_out() -> None:
    item = next(item for item in _panel()["prompts"] if item["id"] == "code_add")

    assert "unsafe_python_import" in check_response(
        item, "import os\ndef add(a, b):\n    return a + b"
    )
    assert any(
        "TimeoutError" in failure
        for failure in check_response(
            item, "def add(a, b):\n    while True:\n        pass"
        )
    )


def test_regression_gate_combines_behavior_and_piqa() -> None:
    panel = _panel()
    results = []
    valid = {
        "greeting": "Hello!",
        "math_multiplication": "17 × 23 = 391.",
        "math_word_problem": "35 balls remain.",
        "code_add": "def add(a, b):\n    return a + b",
        "code_is_prime": (
            "def is_prime(n):\n"
            "    if n < 2: return False\n"
            "    return all(n % i for i in range(2, int(n ** 0.5) + 1))"
        ),
        "memory_name": "Your name is Boris.",
        "memory_rules": "The secret piece is a blue triangle; a star gives 3 points.",
        "instruction_three_bullets": "- Turn off unused lights\n- Unplug idle chargers\n- Lower the thermostat",
    }
    for item in panel["prompts"]:
        results.append({"id": item["id"], "response": valid[item["id"]]})
    report = {"chat_template_applied": True, "results": results}
    piqa = {"benchmarks": {"piqa": {"acc_norm": 0.6839}}}

    audit = audit_regression(panel, report, piqa)

    assert audit["passed"] is True
    piqa["benchmarks"]["piqa"]["acc_norm"] = 0.67
    assert audit_regression(panel, report, piqa)["passed"] is False


def test_multiturn_panel_uses_complete_message_history() -> None:
    panel = _panel()
    memory = next(item for item in panel["prompts"] if item["id"] == "memory_rules")

    assert [message["role"] for message in memory["messages"]] == [
        "user",
        "assistant",
        "user",
    ]
    assert "blue triangle" in memory["messages"][0]["content"]
