import json
from pathlib import Path

RECIPE_PATH = Path("configs/tr_hash_agentic_100m_sft_2m.json")


def load_recipe() -> dict:
    return json.loads(RECIPE_PATH.read_text(encoding="utf-8"))


def components_by_name(recipe: dict) -> dict[str, dict]:
    return {item["name"]: item for item in recipe["components"]}


def test_2m_recipe_is_a_clean_build() -> None:
    recipe = load_recipe()

    assert recipe["train_examples"] == 2_000_000
    assert recipe["validation_examples"] == 100_000
    assert recipe["epochs"] == 3
    assert recipe["replay"]["enabled"] is False
    assert sum(item["train_examples"] for item in recipe["components"]) == 2_000_000
    assert sum(item["validation_examples"] for item in recipe["components"]) == 100_000


def test_signed_example_distribution_is_exact() -> None:
    components = components_by_name(load_recipe())
    expected = {
        "general_conversations": (800_000, 40_000),
        "verified_compact_reasoning": (450_000, 22_500),
        "python_code": (350_000, 17_500),
        "constraints_and_formats": (150_000, 7_500),
        "tool_available_direct_answer": (150_000, 7_500),
        "complete_agentic_trajectories": (100_000, 5_000),
    }

    assert set(components) == set(expected)
    for name, (train_examples, validation_examples) in expected.items():
        assert components[name]["train_examples"] == train_examples
        assert components[name]["validation_examples"] == validation_examples


def test_tools_are_integrated_into_generalist_examples() -> None:
    components = components_by_name(load_recipe())
    top_level_names = set(components)

    for forbidden_silo in ("calculator", "rag", "date_time", "tool_result"):
        assert all(not name.startswith(forbidden_silo) for name in top_level_names)

    no_call = components["tool_available_direct_answer"]
    trajectories = components["complete_agentic_trajectories"]
    assert "answers_directly" in no_call["contract"]
    assert sum(item["train_examples"] for item in trajectories["submix"].values()) == 100_000
    assert sum(item["validation_examples"] for item in trajectories["submix"].values()) == 5_000
    assert sum(trajectories["tool_mix_policy"].values()) == 1.0


def test_token_shares_are_measured_without_padding_trajectories() -> None:
    policy = load_recipe()["token_distribution_policy"]
    partitions = policy["partitions"]

    assert policy["measure_with_pinned_native_tokenizer"] is True
    assert policy["enforcement"] == "report_observed_shares_preserve_signed_example_mix"
    assert set(partitions) == {
        "general_reasoning_code_and_constraints",
        "complete_agentic_trajectories",
        "tools_available_without_call",
    }
    assert all("pilot_observed_share" in item for item in partitions.values())


def test_agentic_contract_and_audit_gates_are_explicit() -> None:
    recipe = load_recipe()
    template = recipe["chat_template"]
    policy = recipe["quality_policy"]

    assert template == {
        "id": "tr-hash-agentic-chat-v1",
        "training_projection": "native_agentic_full_trajectory",
        "assistant_only_loss": True,
        "thinking_policy": "optional_and_never_prefilled",
    }
    for gate in (
        "reject_any_token_truncation",
        "reject_legacy_markers",
        "require_native_marker_balance",
        "require_valid_tool_json",
        "require_tool_name_in_offered_schema",
        "require_calculator_expression_execution",
        "require_calculator_result_match",
        "require_python_parse",
        "require_cross_split_conversation_isolation",
        "require_exact_and_normalized_deduplication",
        "require_benchmark_contamination_audit",
        "require_compiled_label_audit",
        "require_token_distribution_audit",
    ):
        assert policy[gate] is True
    assert policy["record_upstream_python_execution_metadata"] is True
