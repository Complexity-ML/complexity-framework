from scripts.agentic_calculator_demo import (
    ToolCallParseError,
    extract_arithmetic_expression,
    parse_tool_call,
    reflect_calculator_call,
    resolve_direct_response,
    resolve_respond_tool,
    safe_calculator,
)


def test_safe_calculator_basic_arithmetic():
    assert safe_calculator("17 + 25") == "42"
    assert safe_calculator("19 + 23 * 2") == "65"
    assert safe_calculator("(80 - 15) / 5") == "13"


def test_parse_tool_call():
    call = parse_tool_call(
        '<tool_call>{"name":"calculator","arguments":{"expression":"123+456"}}</tool_call>'
    )
    assert call is not None
    assert call.name == "calculator"
    assert call.arguments["expression"] == "123+456"


def test_parse_tool_call_rejects_invalid_json():
    try:
        parse_tool_call('<tool_call>{"name":"calculator","arguments":</tool_call>')
    except ToolCallParseError:
        pass
    else:
        raise AssertionError("invalid tool JSON should raise ToolCallParseError")


def test_resolve_respond_tool():
    assert (
        resolve_respond_tool(
            '<tool_call>{"name":"respond","arguments":{"text":"Hello! How can I help?"}}</tool_call>'
        )
        == "Hello! How can I help?"
    )
    assert (
        resolve_respond_tool(
            '<tool_call>{"name":"final_answer","arguments":{"answer":"Done."}}</tool_call>'
        )
        == "Done."
    )
    assert resolve_respond_tool("Hello without tool") is None


def test_resolve_direct_response_requires_respond_for_tool_output():
    assert resolve_direct_response("Hello!") == ("Hello!", None)
    assert resolve_direct_response(
        '<tool_call>{"name":"respond","arguments":{"text":"Hello!"}}</tool_call>'
    ) == ("Hello!", None)
    assert resolve_direct_response(
        '<tool_call>{"name":"web_search","arguments":{"query":"hello"}}</tool_call>'
    ) == (None, "wrong_direct_tool:web_search")


def test_extract_arithmetic_expression():
    assert extract_arithmetic_expression("What is (80 - 15) / 5?") == "(80 - 15) / 5"
    assert extract_arithmetic_expression("What is 7 * 8 + 9?") == "7 * 8 + 9"
    assert extract_arithmetic_expression("Explain gravity") is None


def test_reflect_calculator_call_repairs_incomplete_expression():
    call = parse_tool_call(
        '<tool_call>{"name":"calculator","arguments":{"expression":"7 * 8"}}</tool_call>'
    )
    reflection = reflect_calculator_call(
        "What is 7 * 8 + 9?",
        "7 * 8 + 9",
        call,
        "7 * 8",
        "56",
    )
    assert reflection is not None
    assert reflection.corrected_expression == "7 * 8 + 9"


def test_reflect_calculator_call_repairs_missing_tool_call():
    reflection = reflect_calculator_call(
        "What is (80 - 15) / 5?",
        "(80 - 15) / 5",
        None,
        None,
        None,
    )
    assert reflection is not None
    assert reflection.corrected_expression == "(80 - 15) / 5"
