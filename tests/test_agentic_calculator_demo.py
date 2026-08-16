from scripts.agentic_calculator_demo import (
    DatetimeError,
    ToolCallParseError,
    extract_arithmetic_expression,
    extract_datetime_hint,
    parse_tool_call,
    safe_calculator,
    safe_datetime,
)


def test_safe_calculator_basic_arithmetic():
    assert safe_calculator("17 + 25") == "42"
    assert safe_calculator("19 + 23 * 2") == "65"
    assert safe_calculator("(80 - 15) / 5") == "13"


def test_safe_datetime_diff():
    assert safe_datetime("diff", a="2026-01-01", b="2026-12-31") == "364"


def test_safe_datetime_add_days():
    assert safe_datetime("add_days", date="2026-05-20", days=30) == "2026-06-19"
    assert safe_datetime("add_days", date="2026-05-20", days=-7) == "2026-05-13"


def test_safe_datetime_weekday():
    assert safe_datetime("weekday", date="2027-01-01") == "Friday"


def test_safe_datetime_rejects_unknown_op():
    try:
        safe_datetime("unknown", date="2026-05-20")
    except DatetimeError:
        pass
    else:
        raise AssertionError("unknown op should raise DatetimeError")


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


def test_extract_arithmetic_expression():
    assert extract_arithmetic_expression("What is (80 - 15) / 5?") == "(80-15)/5"
    assert extract_arithmetic_expression("What is 7 * 8 + 9?") == "7*8+9"
    assert extract_arithmetic_expression("Tell me 18 * 7.") == "18*7."
    assert extract_arithmetic_expression("Explain gravity") is None


def test_extract_datetime_hint_diff():
    hint = extract_datetime_hint("How many days between 2026-01-01 and 2026-12-31?")
    assert hint is not None
    assert hint.op == "diff"
    assert hint.args == {"a": "2026-01-01", "b": "2026-12-31"}


def test_extract_datetime_hint_weekday():
    hint = extract_datetime_hint("What day of the week is 2027-01-01?")
    assert hint is not None
    assert hint.op == "weekday"
    assert hint.args == {"date": "2027-01-01"}


def test_extract_datetime_hint_add_days():
    hint = extract_datetime_hint("What is 2026-05-20 plus 30 days?")
    assert hint is not None
    assert hint.op == "add_days"
    assert hint.args["date"] == "2026-05-20"
    assert hint.args["days"] == 30


def test_extract_datetime_hint_returns_none_for_non_date_question():
    assert extract_datetime_hint("Hello") is None
    assert extract_datetime_hint("What is 17 + 25?") is None
