#!/usr/bin/env python3
"""Run the reflect-bootstrap tool-use loop against a local Complexity checkpoint.

The model accepts a raw natural-language question. If it's tool-shaped it
emits a reflect tool call; the orchestrator extracts the args via regex
and feeds them back; the model emits the real tool call; the orchestrator
executes it; the model (optionally) finalizes the result in natural
language.

Supported tools: calculator (arithmetic), datetime (diff / add_days /
weekday). Anything else routes to chat.
"""

from __future__ import annotations

import argparse
import ast
import json
import operator
import re
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

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


# --- tool executors ---------------------------------------------------------------

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


# --- tool-call parsing + orchestrator-side extraction ----------------------------

@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    raw: str


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


@dataclass
class DatetimeHint:
    op: str
    args: dict[str, Any]


_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_SIGNED_INT_RE = re.compile(r"-?\d+")


def extract_datetime_hint(question: str) -> DatetimeHint | None:
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
    cleaned = question.strip().rstrip("?")
    match = re.search(r"(?:what is|calculate|compute|tell me|how much is)\s+(.+)$", cleaned, flags=re.IGNORECASE)
    if match:
        cleaned = match.group(1)
    allowed = set("0123456789+-*/().% ")
    expr = "".join(ch for ch in cleaned if ch in allowed).strip()
    expr = re.sub(r"\s+", "", expr)
    if any(ch.isdigit() for ch in expr) and any(op in expr for op in "+-*/%"):
        return expr
    return None


# --- model + generation ----------------------------------------------------------

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
    new_ids = output_ids[input_ids.shape[1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    for stop in ("<|endoftext|>", "\nUser:", "\nTool:"):
        text = text.split(stop, 1)[0]
    return text.strip()


# --- the only loop: reflect-bootstrap --------------------------------------------

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
    prompt = f"User:\n{question}\n\nAssistant:\n"
    first = generate(model, tokenizer, prompt, device, max_new_tokens, temperature, top_p)
    try:
        tc = parse_tool_call(first)
    except ToolCallParseError as exc:
        return {"question": question, "first": first, "error": f"invalid_tool_json: {exc}"}

    if tc is None:
        return {"question": question, "tool": "chat", "first": first, "final": first.strip()}
    if tc.name != "reflect":
        return {
            "question": question, "first": first, "tool_call": tc.raw,
            "error": f"expected_reflect_got_{tc.name}",
        }

    embedded = str(tc.arguments.get("question", question))
    dt_hint = extract_datetime_hint(embedded)
    if dt_hint is not None:
        rresult = {"tool": "datetime", "arguments": {"op": dt_hint.op, **dt_hint.args}}
    else:
        expr_hint = extract_arithmetic_expression(embedded)
        if expr_hint is None:
            return {"question": question, "first": first, "error": "reflect_extraction_failed", "embedded": embedded}
        rresult = {"tool": "calculator", "arguments": {"expression": expr_hint}}

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
        return {"question": question, "first": first, "second": second, "error": f"invalid_2nd_tool: {exc}"}
    if tc2 is None or tc2.name != rresult["tool"]:
        return {
            "question": question, "first": first, "second": second,
            "error": f"wrong_2nd_tool:{None if tc2 is None else tc2.name}",
        }

    if tc2.name == "calculator":
        expr = str(tc2.arguments.get("expression", "")).strip()
        try:
            tool_result = safe_calculator(expr)
        except CalculatorError as exc:
            return {"question": question, "tool": "calculator", "tool_call": tc2.raw, "error": str(exc)}
    else:
        op = str(tc2.arguments.get("op", ""))
        call_args = {k: v for k, v in tc2.arguments.items() if k != "op"}
        try:
            tool_result = safe_datetime(op, **call_args)
        except DatetimeError as exc:
            return {"question": question, "tool": "datetime", "tool_call": tc2.raw, "error": str(exc)}

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
        "tool": tc2.name,
        "tool_call": tc2.raw,
        "tool_result": tool_result,
        "final": final,
    }


# --- eval ------------------------------------------------------------------------

def bootstrap_eval_cases(max_cases: int) -> list[tuple[str, str, str]]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Reflect-bootstrap agentic demo/eval for a local SFT checkpoint")
    parser.add_argument("question", nargs="?", default="What is 17 + 25?")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer-o200k"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--eval", action="store_true", help="Run the bootstrap eval suite")
    parser.add_argument("--max-cases", type=int, default=8)
    parser.add_argument("--model-final", action="store_true", help="Ask the model to write the final answer")
    args = parser.parse_args()

    device = pick_device(args.device)
    model, tokenizer = load_model(args.checkpoint, args.tokenizer, device)

    if args.eval:
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
            tail = f" | err={err}" if err else ""
            print(f"{status} | {question} | tool={chosen} (expected={expected_tool}) | final={final!r}{tail}")
        print(f"\nscore={passed}/{len(cases)}")
        return

    result = run_bootstrap_loop(
        model, tokenizer, args.question, device,
        args.max_new_tokens, args.temperature, args.top_p, args.model_final,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
