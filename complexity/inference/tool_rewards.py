"""Verified tool-use rewards for online RL inference."""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, Optional


class CalculatorError(ValueError):
    pass


class DatetimeError(ValueError):
    pass


@dataclass
class VerifiedToolEpisode:
    prompt: str
    target_response: str
    reward: float
    tool_call: Dict[str, Any]
    tool_result: Dict[str, Any]
    metadata: Dict[str, Any]


def extract_arithmetic_expression(question: str) -> Optional[str]:
    if _DATE_RE.search(question):
        return None
    cleaned = question.strip().rstrip("?")
    match = re.search(
        r"(?:what is|calculate|compute|tell me|how much is|combien fait|calcule)\s+(.+)$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if match:
        cleaned = match.group(1)
    allowed = set("0123456789+-*/().% ")
    expr = "".join(ch for ch in cleaned if ch in allowed).strip()
    expr = re.sub(r"\s+", "", expr)
    if any(ch.isdigit() for ch in expr) and any(op in expr for op in "+-*/%"):
        return expr
    return None


def safe_calculator(expression: str) -> str:
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](visit(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            left = visit(node.left)
            right = visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise CalculatorError("power too large")
            return ops[type(node.op)](left, right)
        raise CalculatorError(f"unsupported expression: {expression!r}")

    try:
        value = visit(ast.parse(expression, mode="eval"))
    except ZeroDivisionError as exc:
        raise CalculatorError("division by zero") from exc
    except SyntaxError as exc:
        raise CalculatorError(f"invalid expression: {expression!r}") from exc

    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def build_calculator_episode(prompt: str, question: str) -> Optional[VerifiedToolEpisode]:
    expr = extract_arithmetic_expression(question)
    if expr is None:
        return None
    result = safe_calculator(expr)
    target_response = (
        '<tool_call>{"name":"calculator","arguments":{"expression":"'
        + expr
        + '"}}</tool_call>\n'
        + f"Tool result from calculator: {result}\n"
        + f"Assistant:\n{result}"
    )
    tool_call = {"name": "calculator", "arguments": {"expression": expr}}
    tool_result = {"name": "calculator", "result": result}
    return VerifiedToolEpisode(
        prompt=prompt,
        target_response=target_response,
        reward=1.0,
        tool_call=tool_call,
        tool_result=tool_result,
        metadata={"verified_tool": "calculator", "expression": expr, "expected": result},
    )


_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_SIGNED_INT_RE = re.compile(r"-?\d+")


@dataclass
class DatetimeHint:
    op: str
    args: Dict[str, Any]


def extract_datetime_hint(question: str) -> Optional[DatetimeHint]:
    dates = _DATE_RE.findall(question)
    if not dates:
        return None
    qlow = question.lower()
    if any(kw in qlow for kw in ("day of the week", "weekday", "jour de la semaine")):
        return DatetimeHint(op="weekday", args={"date": dates[0]})
    if len(dates) >= 2 and any(
        kw in qlow for kw in ("between", "difference", "from ", "entre", "différence")
    ):
        return DatetimeHint(op="diff", args={"a": dates[0], "b": dates[1]})
    if any(kw in qlow for kw in ("plus", "add", "after", "après")) and "day" in qlow:
        stripped = _DATE_RE.sub("", question)
        ints = [int(m.group(0)) for m in _SIGNED_INT_RE.finditer(stripped)]
        if ints:
            return DatetimeHint(op="add_days", args={"date": dates[0], "days": ints[0]})
    return None


def safe_datetime(op: str, **args: Any) -> str:
    try:
        if op == "diff":
            a = date.fromisoformat(str(args["a"]))
            b = date.fromisoformat(str(args["b"]))
            return str((b - a).days)
        if op == "add_days":
            d = date.fromisoformat(str(args["date"]))
            days = int(args["days"])
            return (d + timedelta(days=days)).isoformat()
        if op == "weekday":
            d = date.fromisoformat(str(args["date"]))
            return _WEEKDAYS[d.weekday()]
    except (KeyError, ValueError, TypeError) as exc:
        raise DatetimeError(f"invalid datetime args op={op} args={args}: {exc}") from exc
    raise DatetimeError(f"unknown datetime op: {op!r}")


def build_datetime_episode(prompt: str, question: str) -> Optional[VerifiedToolEpisode]:
    hint = extract_datetime_hint(question)
    if hint is None:
        return None
    result = safe_datetime(hint.op, **hint.args)
    args_json = ",".join(
        f'"{key}":"{value}"' if isinstance(value, str) else f'"{key}":{value}'
        for key, value in {"op": hint.op, **hint.args}.items()
    )
    target_response = (
        '<tool_call>{"name":"datetime","arguments":{'
        + args_json
        + "}}</tool_call>\n"
        + f"Tool result from datetime: {result}\n"
        + f"Assistant:\n{result}"
    )
    tool_call = {"name": "datetime", "arguments": {"op": hint.op, **hint.args}}
    tool_result = {"name": "datetime", "result": result}
    return VerifiedToolEpisode(
        prompt=prompt,
        target_response=target_response,
        reward=1.0,
        tool_call=tool_call,
        tool_result=tool_result,
        metadata={
            "verified_tool": "datetime",
            "op": hint.op,
            "arguments": hint.args,
            "expected": result,
        },
    )


def build_verified_tool_episode(prompt: str, question: str) -> Optional[VerifiedToolEpisode]:
    return build_datetime_episode(prompt, question) or build_calculator_episode(prompt, question)
    tool_call = {"name": "calculator", "arguments": {"expression": expr}}
    tool_result = {"name": "calculator", "result": result}
    return VerifiedToolEpisode(
        prompt=prompt,
        target_response=target_response,
        reward=1.0,
        tool_call=tool_call,
        tool_result=tool_result,
        metadata={"verified_tool": "calculator", "expression": expr, "expected": result},
    )
