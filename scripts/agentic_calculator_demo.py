#!/usr/bin/env python3
"""Run a tiny tool-use loop against a local Complexity checkpoint.

This is intentionally small: it tests whether an SFT checkpoint emits a
calculator tool call, executes that call, then asks the model for a final
answer using the tool result.
"""

from __future__ import annotations

import argparse
import ast
import json
import operator
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datetime import date, timedelta

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from complexity.models import ComplexityModel
from complexity.tokenizer import Tokenizer
from complexity.utils.device import configure_torch_acceleration
from scripts.sft_100m_o200k_tr_local import checkpoint_config, load_checkpoint_state


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


class CalculatorError(ValueError):
    pass


class DatetimeError(ValueError):
    pass


class ToolCallParseError(ValueError):
    pass


_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


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


def safe_calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression without eval()."""
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
        return str(int(value))
    return str(value)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    raw: str


@dataclass
class Reflection:
    issue: str
    corrected_expression: str | None

    def to_tool_result(self) -> str:
        payload = {
            "name": "reflect",
            "result": {
                "issue": self.issue,
                "corrected_expression": self.corrected_expression,
            },
        }
        return json.dumps(payload, separators=(",", ":"))


def parse_tool_call(text: str) -> ToolCall | None:
    match = TOOL_CALL_RE.search(text)
    if not match:
        if "<tool_call>" in text or "</tool_call>" in text:
            raise ToolCallParseError(f"malformed tool call block: {text}")
        return None
    raw = match.group(1)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ToolCallParseError(f"invalid tool JSON: {raw}") from exc
    return ToolCall(
        name=str(payload.get("name", "")),
        arguments=dict(payload.get("arguments", {})),
        raw=raw,
    )


def resolve_respond_tool(text: str) -> str | None:
    """Return final chat text when the model uses the respond pseudo-tool."""
    try:
        tool_call = parse_tool_call(text)
    except ToolCallParseError:
        return None
    if tool_call is None or tool_call.name not in {"respond", "final_answer", "speak"}:
        return None
    for key in ("text", "answer", "message"):
        value = tool_call.arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def resolve_direct_response(text: str) -> tuple[str | None, str | None]:
    """Resolve non-tool chat output, returning (final, error)."""
    try:
        tool_call = parse_tool_call(text)
    except ToolCallParseError:
        return None, "invalid_direct_tool_json"
    if tool_call is None:
        return text.strip(), None
    if tool_call.name not in {"respond", "final_answer", "speak"}:
        return None, f"wrong_direct_tool:{tool_call.name}"
    for key in ("text", "answer", "message"):
        value = tool_call.arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip(), None
    return "", None


@dataclass
class DatetimeHint:
    op: str
    args: dict[str, Any]

    def hint_line(self) -> str:
        if self.op == "diff":
            return f"Operation: diff. Date A: {self.args['a']}. Date B: {self.args['b']}."
        if self.op == "add_days":
            return f"Operation: add_days. Date: {self.args['date']}. Days: {self.args['days']}."
        if self.op == "weekday":
            return f"Operation: weekday. Date: {self.args['date']}."
        raise DatetimeError(f"unknown op: {self.op!r}")


_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_SIGNED_INT_RE = re.compile(r"-?\d+")


def extract_datetime_hint(question: str) -> DatetimeHint | None:
    """Detect whether a question is datetime-shaped and return its hint."""
    dates = _DATE_RE.findall(question)
    if not dates:
        return None
    qlow = question.lower()
    if any(kw in qlow for kw in ("day of the week", "weekday", "weekend")):
        return DatetimeHint(op="weekday", args={"date": dates[0]})
    if len(dates) >= 2 and any(kw in qlow for kw in ("between", "difference", "from ")):
        return DatetimeHint(op="diff", args={"a": dates[0], "b": dates[1]})
    if "day" in qlow:
        stripped = _DATE_RE.sub("", question)
        ints = [int(m) for m in _SIGNED_INT_RE.findall(stripped)]
        if ints:
            return DatetimeHint(op="add_days", args={"date": dates[0], "days": ints[0]})
    return None


def extract_arithmetic_expression(question: str) -> str | None:
    """Best-effort extraction of an arithmetic span from an English question."""
    cleaned = question.strip().rstrip("?")
    match = re.search(r"(?:what is|calculate|compute)\s+(.+)$", cleaned, flags=re.IGNORECASE)
    if match:
        cleaned = match.group(1)
    allowed = set("0123456789+-*/().% ")
    expr = "".join(ch for ch in cleaned if ch in allowed).strip()
    expr = re.sub(r"\s+", " ", expr)
    if any(ch.isdigit() for ch in expr) and any(op in expr for op in "+-*/%"):
        return expr
    return None


def normalize_expression(expression: str) -> str:
    return re.sub(r"\s+", "", expression)


def reflect_calculator_call(
    question: str,
    expression_hint: str | None,
    tool_call: ToolCall | None,
    expression: str | None,
    tool_result: str | None,
) -> Reflection | None:
    """Detect calculator mistakes that a reflect tool can repair."""
    if not expression_hint:
        return None

    if tool_call is None:
        return Reflection(
            issue="missing calculator tool call; use the arithmetic expression from the question",
            corrected_expression=expression_hint,
        )

    if tool_call.name != "calculator":
        return Reflection(
            issue=f"wrong tool {tool_call.name!r}; use calculator for arithmetic",
            corrected_expression=expression_hint,
        )

    if expression is None or not expression.strip():
        return Reflection(
            issue="empty calculator expression",
            corrected_expression=expression_hint,
        )

    if normalize_expression(expression) == normalize_expression(expression_hint):
        return None

    try:
        expected_result = safe_calculator(expression_hint)
    except CalculatorError:
        return None

    if tool_result != expected_result:
        return Reflection(
            issue=(
                f"calculator expression {expression!r} does not match the user question {question!r}; "
                f"use the full expression {expression_hint!r}"
            ),
            corrected_expression=expression_hint,
        )
    return None


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint: Path, tokenizer_path: Path, device: torch.device) -> tuple[ComplexityModel, Tokenizer]:
    configure_torch_acceleration(kernel_policy=False, log=False)
    _, state = load_checkpoint_state(checkpoint, map_location="cpu")
    config = checkpoint_config(state)
    config.use_custom_kernels = False
    model = ComplexityModel(config).to(device)
    missing, unexpected = model.load_state_dict(state["model"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    tokenizer = Tokenizer.load(str(tokenizer_path))
    return model, tokenizer


@torch.no_grad()
def generate(
    model: ComplexityModel,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=max(temperature, 1e-5),
        top_k=0,
        top_p=top_p,
        do_sample=temperature > 0,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
    )[0]
    new_ids = output_ids[input_ids.shape[1] :]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    for stop in ("<|endoftext|>", "\nUser:", "\nTool:"):
        text = text.split(stop, 1)[0]
    return text.strip()


def run_calculator_loop(
    model: ComplexityModel,
    tokenizer: Tokenizer,
    question: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    use_reflection: bool = True,
    model_final: bool = False,
) -> dict[str, Any]:
    expression_hint = extract_arithmetic_expression(question)
    if expression_hint is None:
        direct_prompt = (
            "User:\n"
            f"{question}\n\n"
            "Available tool for normal chat:\n"
            '<tool_call>{"name":"respond","arguments":{"text":"..."}}</tool_call>\n'
            "Use respond for greetings, casual chat, explanations, or any answer that does not need another tool.\n\n"
            "Assistant:\n"
        )
        direct = generate(model, tokenizer, direct_prompt, device, max_new_tokens, temperature, top_p)
        response_text, direct_error = resolve_direct_response(direct)
        return {
            "question": question,
            "mode": "direct",
            "tool_call": direct if "<tool_call>" in direct else None,
            "final": response_text,
            "error": direct_error,
        }

    hint = f"Arithmetic expression: {expression_hint}\n" if expression_hint else ""
    first_prompt = (
        "User:\n"
        f"Use the calculator tool to answer: {question}\n"
        f"{hint}"
        "Copy the full arithmetic expression into the calculator expression argument. "
        "Do not compute it mentally.\n\n"
        "Assistant:\n"
    )
    first = generate(model, tokenizer, first_prompt, device, max_new_tokens, temperature, top_p)
    try:
        tool_call = parse_tool_call(first)
    except ToolCallParseError as exc:
        return {"question": question, "first": first, "error": "invalid_tool_json", "detail": str(exc)}
    expression = None
    result = None
    error = None
    if tool_call is not None and tool_call.name == "calculator":
        expression = str(tool_call.arguments.get("expression", "")).strip()
        try:
            result = safe_calculator(expression)
        except CalculatorError as exc:
            error = str(exc)

    reflection = reflect_calculator_call(question, expression_hint, tool_call, expression, result)
    if use_reflection and reflection is not None and reflection.corrected_expression:
        expression = reflection.corrected_expression
        try:
            result = safe_calculator(expression)
            error = None
        except CalculatorError as exc:
            error = str(exc)

    if tool_call is None and (not use_reflection or reflection is None):
        return {"question": question, "first": first, "error": "missing_tool_call"}
    if tool_call is not None and tool_call.name != "calculator" and (not use_reflection or reflection is None):
        return {"question": question, "first": first, "tool_call": tool_call.raw, "error": "wrong_tool"}
    if error is not None:
        return {
            "question": question,
            "first": first,
            "tool_call": tool_call.raw if tool_call else None,
            "expression": expression,
            "reflection": reflection.to_tool_result() if reflection else None,
            "error": error,
        }

    raw_calculator_call = (
        tool_call.raw
        if tool_call is not None and reflection is None
        else json.dumps(
            {"name": "calculator", "arguments": {"expression": expression}},
            separators=(",", ":"),
        )
    )
    reflect_block = ""
    if reflection is not None:
        reflect_call = json.dumps(
            {
                "name": "reflect",
                "arguments": {
                    "question": question,
                    "draft_tool_call": tool_call.raw if tool_call else first,
                    "task": "check whether the calculator expression matches the user question",
                },
            },
            separators=(",", ":"),
        )
        reflect_block = (
            f"Assistant:\n<tool_call>{reflect_call}</tool_call>\n\n"
            f"Tool result from reflect: {reflection.to_tool_result()}\n\n"
        )

    initial_tool_block = (
        f"Assistant:\n{first}\n\n"
        if reflection is not None
        else f"Assistant:\n<tool_call>{raw_calculator_call}</tool_call>\n\n"
    )
    corrected_tool_block = (
        f"Assistant:\n<tool_call>{raw_calculator_call}</tool_call>\n\n"
        if reflection is not None
        else ""
    )
    if model_final:
        final_prompt = (
            "User:\n"
            f"Use the calculator tool to answer: {question}\n"
            f"{hint}"
            "Copy the full arithmetic expression into the calculator expression argument. "
            "Do not compute it mentally.\n\n"
            f"{initial_tool_block}"
            f"{reflect_block}"
            f"{corrected_tool_block}"
            f"Tool result from calculator: {result}\n\n"
            "Assistant:\n"
        )
        final = generate(model, tokenizer, final_prompt, device, max_new_tokens, temperature, top_p)
    else:
        final = f"The answer is {result}."
    return {
        "question": question,
        "first": first,
        "tool_call": raw_calculator_call,
        "expression": expression,
        "tool_result": result,
        "reflection": reflection.to_tool_result() if reflection else None,
        "final": final,
    }


def run_datetime_loop(
    model: ComplexityModel,
    tokenizer: Tokenizer,
    question: str,
    hint: DatetimeHint,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    model_final: bool = False,
) -> dict[str, Any]:
    expected_result = safe_datetime(hint.op, **hint.args)

    first_prompt = (
        "User:\n"
        f"Use the datetime tool to answer: {question}\n"
        f"{hint.hint_line()}\n"
        "Copy the dates and numbers exactly into the datetime arguments. "
        "Do not compute mentally.\n\n"
        "Assistant:\n"
    )
    first = generate(model, tokenizer, first_prompt, device, max_new_tokens, temperature, top_p)
    try:
        tool_call = parse_tool_call(first)
    except ToolCallParseError as exc:
        return {"question": question, "tool": "datetime", "first": first, "error": "invalid_tool_json", "detail": str(exc)}
    if tool_call is None:
        return {"question": question, "tool": "datetime", "first": first, "error": "missing_tool_call"}
    if tool_call.name != "datetime":
        return {
            "question": question,
            "tool": "datetime",
            "first": first,
            "tool_call": tool_call.raw,
            "error": f"wrong_tool:{tool_call.name}",
        }

    op = str(tool_call.arguments.get("op", ""))
    call_args = {k: v for k, v in tool_call.arguments.items() if k != "op"}
    try:
        result = safe_datetime(op, **call_args)
    except DatetimeError as exc:
        return {
            "question": question,
            "tool": "datetime",
            "first": first,
            "tool_call": tool_call.raw,
            "error": str(exc),
            "expected_result": expected_result,
        }

    if model_final:
        final_prompt = (
            first_prompt
            + f"<tool_call>{tool_call.raw}</tool_call>\n\n"
            + f"Tool result from datetime: {result}\n\nAssistant:\n"
        )
        final = generate(model, tokenizer, final_prompt, device, max_new_tokens, temperature, top_p)
    else:
        final = f"The answer is {result}."
    return {
        "question": question,
        "tool": "datetime",
        "first": first,
        "tool_call": tool_call.raw,
        "tool_result": result,
        "expected_result": expected_result,
        "final": final,
    }


def run_chat(
    model: ComplexityModel,
    tokenizer: Tokenizer,
    question: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    prompt = f"User:\n{question}\n\nAssistant:\n"
    out = generate(model, tokenizer, prompt, device, max_new_tokens, temperature, top_p)
    tool_marker = "<tool_call>" in out
    return {
        "question": question,
        "tool": "chat",
        "final": None if tool_marker else out.strip(),
        "error": "unexpected_tool_call" if tool_marker else None,
        "raw": out,
    }


def dispatch_question(
    model: ComplexityModel,
    tokenizer: Tokenizer,
    question: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    use_reflection: bool,
    model_final: bool,
) -> dict[str, Any]:
    dt_hint = extract_datetime_hint(question)
    if dt_hint is not None:
        return run_datetime_loop(
            model, tokenizer, question, dt_hint, device, max_new_tokens, temperature, top_p, model_final
        )
    if extract_arithmetic_expression(question) is not None:
        return run_calculator_loop(
            model, tokenizer, question, device, max_new_tokens, temperature, top_p, use_reflection, model_final
        )
    return run_chat(model, tokenizer, question, device, max_new_tokens, temperature, top_p)


def run_bootstrap_loop(
    model: ComplexityModel,
    tokenizer: Tokenizer,
    question: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    model_final: bool = False,
) -> dict[str, Any]:
    """Drive the reflect-bootstrap pipeline for a raw NL question.

    Pipeline: NL -> model emits reflect call (or chats) -> orchestrator
    extracts args via regex -> model emits real tool call -> execute ->
    optional model finalization.
    """
    prompt = f"User:\n{question}\n\nAssistant:\n"
    first = generate(model, tokenizer, prompt, device, max_new_tokens, temperature, top_p)
    try:
        tc = parse_tool_call(first)
    except ToolCallParseError as exc:
        return {"question": question, "mode": "bootstrap", "first": first, "error": f"invalid_tool_json: {exc}"}

    if tc is None:
        return {
            "question": question,
            "mode": "chat",
            "tool": "chat",
            "first": first,
            "final": first.strip(),
        }
    if tc.name != "reflect":
        return {
            "question": question,
            "mode": "bootstrap",
            "first": first,
            "tool_call": tc.raw,
            "error": f"expected_reflect_got_{tc.name}",
        }

    embedded = str(tc.arguments.get("question", question))
    dt_hint = extract_datetime_hint(embedded)
    if dt_hint is not None:
        rresult = {"tool": "datetime", "arguments": {"op": dt_hint.op, **dt_hint.args}}
    else:
        expr_hint = extract_arithmetic_expression(embedded)
        if expr_hint is None:
            return {
                "question": question,
                "mode": "bootstrap",
                "first": first,
                "error": "reflect_extraction_failed",
                "embedded_question": embedded,
            }
        rresult = {
            "tool": "calculator",
            "arguments": {"expression": normalize_expression(expr_hint)},
        }

    rresult_str = json.dumps(rresult, separators=(",", ":"))
    second_prompt = (
        prompt
        + f"{first}\n\n"
        + f"Tool result from reflect: {rresult_str}\n\n"
        + "Assistant:\n"
    )
    second = generate(model, tokenizer, second_prompt, device, max_new_tokens, temperature, top_p)
    try:
        tc2 = parse_tool_call(second)
    except ToolCallParseError as exc:
        return {
            "question": question,
            "mode": "bootstrap",
            "first": first,
            "second": second,
            "reflect_result": rresult,
            "error": f"invalid_2nd_tool_json: {exc}",
        }
    if tc2 is None or tc2.name != rresult["tool"]:
        return {
            "question": question,
            "mode": "bootstrap",
            "first": first,
            "second": second,
            "reflect_result": rresult,
            "error": f"wrong_2nd_tool:{None if tc2 is None else tc2.name}",
        }

    if tc2.name == "calculator":
        expr = str(tc2.arguments.get("expression", "")).strip()
        try:
            tool_result = safe_calculator(expr)
        except CalculatorError as exc:
            return {
                "question": question, "mode": "bootstrap", "tool": "calculator",
                "tool_call": tc2.raw, "error": str(exc),
            }
    else:
        op = str(tc2.arguments.get("op", ""))
        call_args = {k: v for k, v in tc2.arguments.items() if k != "op"}
        try:
            tool_result = safe_datetime(op, **call_args)
        except DatetimeError as exc:
            return {
                "question": question, "mode": "bootstrap", "tool": "datetime",
                "tool_call": tc2.raw, "error": str(exc),
            }

    if model_final:
        final_prompt = (
            second_prompt
            + f"<tool_call>{tc2.raw}</tool_call>\n\n"
            + f"Tool result from {tc2.name}: {tool_result}\n\nAssistant:\n"
        )
        final = generate(model, tokenizer, final_prompt, device, max_new_tokens, temperature, top_p)
    else:
        final = f"The answer is {tool_result}."

    return {
        "question": question,
        "mode": "bootstrap",
        "tool": tc2.name,
        "reflect_result": rresult,
        "tool_call": tc2.raw,
        "tool_result": tool_result,
        "final": final,
    }


def bootstrap_eval_cases(max_cases: int) -> list[tuple[str, str, str]]:
    # (raw NL question, expected_tool, expected_substring_in_final)
    cases = [
        ("What is 17 + 25?", "calculator", "42"),
        ("How many days between 2026-01-01 and 2026-12-31?", "datetime", "364"),
        ("What is 18 * 7?", "calculator", "126"),
        ("What day of the week is 2027-01-01?", "datetime", "Friday"),
        ("Hello", "chat", "Hello"),
        ("What is 91 - 34?", "calculator", "57"),
        ("I have 30 days of vacation this year.", "chat", ""),
        ("What is 2026-05-20 plus 30 days?", "datetime", "2026-06-19"),
    ]
    return cases[:max_cases]


def datetime_eval_cases(max_cases: int) -> list[str]:
    cases = [
        "How many days between 2026-01-01 and 2026-12-31?",
        "What is 2026-05-20 plus 30 days?",
        "What day of the week is 2027-01-01?",
        "From 2026-05-20 to 2026-05-21, how many days?",
        "Add -7 days to 2026-05-20.",
        "Which weekday falls on 2026-12-25?",
        "How many days between 2026-05-20 and 2026-05-20?",
        "What is 2026-06-15 plus 100 days?",
    ]
    return cases[:max_cases]


def routing_eval_cases(max_cases: int) -> list[tuple[str, str, str]]:
    # (question, expected_tool, expected_substring_in_final)
    cases = [
        ("What is 17 + 25?", "calculator", "42"),
        ("How many days between 2026-01-01 and 2026-12-31?", "datetime", "364"),
        ("Hello", "chat", "Hello"),
        ("What day of the week is 2027-01-01?", "datetime", "Friday"),
        ("What is 18 * 7?", "calculator", "126"),
        ("Add 30 days to 2026-05-20.", "datetime", "2026-06-19"),
        ("I have 30 days of vacation this year.", "chat", ""),
        ("From 2026-05-20 to 2026-05-21, how many days?", "datetime", "1"),
    ]
    return cases[:max_cases]


def eval_cases(max_cases: int) -> list[tuple[str, str]]:
    cases = [
        ("What is 17 + 25?", "42"),
        ("What is 123 + 456?", "579"),
        ("What is 18 * 7?", "126"),
        ("What is 91 - 34?", "57"),
        ("What is 144 / 12?", "12"),
        ("What is 19 + 23 * 2?", "65"),
        ("What is (80 - 15) / 5?", "13"),
        ("What is 7 * 8 + 9?", "65"),
    ]
    return cases[:max_cases]


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculator tool-call demo/eval for a local SFT checkpoint")
    parser.add_argument("question", nargs="?", default="What is 17 + 25?")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer-o200k"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval-datetime", action="store_true", help="Run the datetime tool eval suite")
    parser.add_argument("--eval-routing", action="store_true", help="Run a mixed calculator/datetime/chat routing eval")
    parser.add_argument("--eval-bootstrap", action="store_true", help="Run the reflect-bootstrap eval (raw NL, no scaffold)")
    parser.add_argument("--max-cases", type=int, default=8)
    parser.add_argument("--no-reflect", action="store_true", help="Disable the reflect repair tool")
    parser.add_argument("--model-final", action="store_true", help="Ask the model to write the final answer")
    args = parser.parse_args()

    device = pick_device(args.device)
    model, tokenizer = load_model(args.checkpoint, args.tokenizer, device)

    if args.eval_datetime:
        passed = 0
        questions = datetime_eval_cases(args.max_cases)
        for question in questions:
            hint = extract_datetime_hint(question)
            if hint is None:
                print(f"FAIL routing | {question} | could not detect datetime intent")
                continue
            row = run_datetime_loop(
                model, tokenizer, question, hint, device,
                args.max_new_tokens, args.temperature, args.top_p, args.model_final,
            )
            final = str(row.get("final", ""))
            expected = row.get("expected_result")
            ok = row.get("tool_result") == expected and expected is not None and str(expected) in final
            status = "PASS" if ok else "FAIL"
            passed += int(ok)
            print(f"{status} | {question} | tool={row.get('tool_result')} expected={expected} | final={final}")
        print(f"\nscore={passed}/{len(questions)}")
        return

    if args.eval_bootstrap:
        passed = 0
        cases = bootstrap_eval_cases(args.max_cases)
        for question, expected_tool, expected_sub in cases:
            row = run_bootstrap_loop(
                model, tokenizer, question, device,
                args.max_new_tokens, args.temperature, args.top_p, args.model_final,
            )
            chosen = row.get("tool") or "?"
            final = str(row.get("final", ""))
            tool_ok = chosen == expected_tool
            content_ok = (expected_sub == "") or (expected_sub in final)
            ok = tool_ok and content_ok and row.get("error") is None
            status = "PASS" if ok else "FAIL"
            passed += int(ok)
            err = row.get("error")
            print(f"{status} | {question} | tool={chosen} (expected={expected_tool}) | final={final!r}" + (f" | err={err}" if err else ""))
        print(f"\nscore={passed}/{len(cases)}")
        return

    if args.eval_routing:
        passed = 0
        cases = routing_eval_cases(args.max_cases)
        for question, expected_tool, expected_sub in cases:
            row = dispatch_question(
                model, tokenizer, question, device,
                args.max_new_tokens, args.temperature, args.top_p,
                not args.no_reflect, args.model_final,
            )
            chosen = row.get("tool") or ("calculator" if row.get("tool_call") else "chat")
            final = str(row.get("final", ""))
            tool_ok = chosen == expected_tool
            content_ok = (expected_sub == "") or (expected_sub in final)
            ok = tool_ok and content_ok and row.get("error") is None
            status = "PASS" if ok else "FAIL"
            passed += int(ok)
            print(f"{status} | {question} | tool={chosen} (expected={expected_tool}) | final={final!r}")
        print(f"\nscore={passed}/{len(cases)}")
        return

    if not args.eval:
        result = dispatch_question(
            model,
            tokenizer,
            args.question,
            device,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            not args.no_reflect,
            args.model_final,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    rows = []
    passed = 0
    for question, expected in eval_cases(args.max_cases):
        row = run_calculator_loop(
            model,
            tokenizer,
            question,
            device,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            not args.no_reflect,
            args.model_final,
        )
        final = str(row.get("final", ""))
        ok = row.get("tool_result") == expected and expected in final
        row["expected"] = expected
        row["ok"] = ok
        rows.append(row)
        passed += int(ok)
        status = "PASS" if ok else "FAIL"
        reflected = " reflect" if row.get("reflection") else ""
        print(f"{status}{reflected} | {question} | tool={row.get('tool_result')} | final={final}")

    print(f"\nscore={passed}/{len(rows)}")


if __name__ == "__main__":
    main()
