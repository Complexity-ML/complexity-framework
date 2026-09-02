from scripts.prepare_tr_hash_agentic_sft import (
    direct_completion,
    mini_projected_record,
    normalize_tool,
    parse_when2call_response,
    tool_completion,
)


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
                    {"type": "function", "function": {"name": "weather", "arguments": {"city": "Paris"}}}
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
