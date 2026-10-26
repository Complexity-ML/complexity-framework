from pathlib import Path

from complexity.training.sft_preparation import (
    direct_completion,
    load_sft_recipe,
    mini_projected_record,
    normalize_tool,
    parse_when2call_response,
    tool_completion,
)

RECIPE = load_sft_recipe(
    Path(__file__).resolve().parents[1] / "configs/examples/agentic_sft_balanced_100k.json"
)
TRAIN_QUOTAS = RECIPE["quotas"]["train"]
EVAL_QUOTAS = RECIPE["quotas"]["validation"]


def test_balanced_recipe_preserves_verified_historical_allocations():
    # The archived source has 100K/5K. The former 200K assertion was added in
    # the commit deleting that source, without a corresponding 200K recipe.
    assert sum(TRAIN_QUOTAS.values()) == 100_000
    assert sum(EVAL_QUOTAS.values()) == 5_000
    assert TRAIN_QUOTAS["smol_smoltalk"] == 90_000
    assert EVAL_QUOTAS["smol_smoltalk"] == 4_500
    assert RECIPE["source_revision"] == "749c9a2:scripts/prepare_tr_hash_agentic_sft.py"
    assert RECIPE["release_eligible"] is False


def test_native_targets_never_add_reasoning_markers():
    direct = direct_completion("Hello")
    tool = tool_completion({"name": "weather", "arguments": {"city": "Paris"}})
    assert direct == "<|final_start|>Hello<|final_end|><|end_of_turn|>"
    assert "<|think_start|>" not in direct + tool
    assert "<TOOLCALL>" not in tool


def test_when2call_accepts_one_call_and_rejects_multiple_calls():
    parsed = parse_when2call_response(
        '<TOOLCALL>[{"name":"weather","arguments":{"city":"Paris"}}]</TOOLCALL>'
    )
    assert parsed == ("tool_call", {"name": "weather", "arguments": {"city": "Paris"}})
    assert (
        parse_when2call_response(
            '<TOOLCALL>[{"name":"a","arguments":{}},{"name":"b","arguments":{}}]</TOOLCALL>'
        )
        is None
    )


def test_normalize_tool_requires_structured_arguments():
    assert normalize_tool({"name": "x", "arguments": '{"n":1}'}) == {
        "name": "x",
        "arguments": {"n": 1},
    }
    assert normalize_tool({"name": "x", "arguments": "not-json"}) is None


def test_tool_mini_projection_uses_native_markers():
    row = {
        "messages": [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "weather", "arguments": {"city": "Paris"}},
                    }
                ],
            },
            {"role": "tool", "content": "Sunny"},
            {"role": "assistant", "content": "It is sunny."},
        ],
        "tools": [
            {"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}
        ],
    }
    record = mini_projected_record(row)
    assert record is not None
    assert record["prompt"].endswith("<|assistant|>")
    assert "<TOOLCALL>" not in record["completion"]
