"""Composable JSONL builders for staged SFT replay mixes."""

from __future__ import annotations

import json
import random
from pathlib import Path

from .config import DatasetMix


CHAT_EXAMPLES = [
    ("Hello", "Hello! How can I help?"),
    ("Hi", "Hi! What would you like to work on?"),
    ("Explain gravity simply.", "Gravity is the force that pulls objects with mass toward each other."),
    ("What is a GPU?", "A GPU is a processor designed to run many parallel computations efficiently."),
    ("Explain fine-tuning in one sentence.", "Fine-tuning adapts a pretrained model to a narrower behavior or dataset."),
    ("Do I need a tool to say hello?", "No. A greeting can be answered directly."),
]

RESPOND_EXAMPLES = [
    ("Hello", "Hello! How can I help?"),
    ("Tell me you are ready.", "Ready."),
    ("I just want to chat.", "Sure. I can chat without using external tools."),
]

REFLECT_TEMPLATES = [
    "calculator expression {bad!r} does not match the user question {question!r}; use the full expression {good!r}",
    "the tool call dropped part of the arithmetic expression; use {good!r}",
    "the calculator argument is incomplete; replace it with {good!r}",
]


def dump_record(handle, prompt: str, completion: str) -> None:
    handle.write(json.dumps({"prompt": prompt.rstrip() + "\n", "completion": completion.strip()}, ensure_ascii=False) + "\n")


def chat_prompt(user: str) -> str:
    return f"User:\n{user}\n\nAssistant:\n"


def calculator_call(expression: str) -> str:
    return '<tool_call>{"name":"calculator","arguments":{"expression":"' + expression + '"}}</tool_call>'


def respond_call(text: str) -> str:
    payload = {"name": "respond", "arguments": {"text": text}}
    return f"<tool_call>{json.dumps(payload, separators=(',', ':'))}</tool_call>"


def reflect_call(question: str, draft_tool_call: str) -> str:
    payload = {
        "name": "reflect",
        "arguments": {
            "question": question,
            "draft_tool_call": draft_tool_call,
            "task": "check whether the calculator expression matches the user question",
        },
    }
    return f"<tool_call>{json.dumps(payload, separators=(',', ':'))}</tool_call>"


def reflect_result(issue: str, corrected_expression: str) -> str:
    payload = {"name": "reflect", "result": {"issue": issue, "corrected_expression": corrected_expression}}
    return json.dumps(payload, separators=(",", ":"))


def fmt_number(value: float | int) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def make_expression(rng: random.Random) -> tuple[str, str, list[str]]:
    a = rng.randint(1, 99)
    b = rng.randint(1, 99)
    c = rng.randint(1, 20)
    kind = rng.randrange(8)
    bad: list[str] = []
    if kind == 0:
        expr, value = f"{a}+{b}", a + b
    elif kind == 1:
        expr, value = f"{a}-{b}", a - b
    elif kind == 2:
        expr, value = f"{a}*{c}", a * c
    elif kind == 3:
        expr, value = f"{a * c}/{c}", a
    elif kind == 4:
        expr, value, bad = f"{a}-{b}+{c}", a - b + c, [f"{a}-{b}", f"{b}+{c}"]
    elif kind == 5:
        expr, value, bad = f"{a}+{b}*{c}", a + b * c, [f"{a}+{b}", f"{b}*{c}"]
    elif kind == 6:
        base = rng.randint(-30, 30)
        b = rng.randint(1, 99)
        a = b + base * c
        expr, value, bad = f"({a}-{b})/{c}", base, [f"{a}-{b}", f"{b}/{c}", f"{c}-{b}"]
    else:
        expr, value, bad = f"({a}+{b})*{c}", (a + b) * c, [f"{a}+{b}", f"{b}*{c}"]
    return expr, fmt_number(value), bad


class StackSFTDatasetBuilder:
    """Build stage JSONL files from weighted named sources."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def build(self, mix: DatasetMix, out: str | Path) -> Path:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        rng = random.Random(mix.seed + self.seed)
        counts = self._counts(mix)
        with out.open("w", encoding="utf-8") as handle:
            for source, count in counts.items():
                writer = getattr(self, f"_write_{source}", None)
                if writer is None:
                    raise ValueError(f"unknown stack SFT source: {source}")
                writer(handle, count, rng)
        lines = out.read_text(encoding="utf-8").splitlines()
        rng.shuffle(lines)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out

    def _counts(self, mix: DatasetMix) -> dict[str, int]:
        total_weight = sum(max(0.0, weight) for weight in mix.sources.values())
        if total_weight <= 0:
            raise ValueError("dataset mix weights must sum to a positive value")
        counts = {name: int(mix.records * max(0.0, weight) / total_weight) for name, weight in mix.sources.items()}
        first = next(iter(counts))
        counts[first] += mix.records - sum(counts.values())
        return counts

    def _write_chat(self, handle, count: int, rng: random.Random) -> None:
        for _ in range(count):
            user, answer = rng.choice(CHAT_EXAMPLES)
            dump_record(handle, chat_prompt(user), answer)

    _write_general = _write_chat
    _write_general_replay = _write_chat
    _write_no_tool = _write_chat
    _write_no_tool_needed = _write_chat

    def _write_respond(self, handle, count: int, rng: random.Random) -> None:
        for _ in range(count):
            user, answer = rng.choice(RESPOND_EXAMPLES)
            dump_record(handle, chat_prompt(user), respond_call(answer))

    def _write_calculator(self, handle, count: int, rng: random.Random) -> None:
        for _ in range(count):
            expr, _, _ = make_expression(rng)
            question = rng.choice([f"What is {expr}?", f"Calculate {expr}.", f"Use the calculator for {expr}."])
            prompt = (
                f"User:\nUse the calculator tool to answer: {question}\n"
                f"Arithmetic expression: {expr}\n"
                "Copy the full arithmetic expression into the calculator expression argument. Do not compute it mentally.\n\n"
                "Assistant:\n"
            )
            dump_record(handle, prompt, calculator_call(expr))

    _write_tool_call = _write_calculator
    _write_tool_call_replay = _write_calculator

    def _write_final(self, handle, count: int, rng: random.Random) -> None:
        for _ in range(count):
            expr, result, _ = make_expression(rng)
            question = f"What is {expr}?"
            prompt = (
                f"User:\nUse the calculator tool to answer: {question}\n"
                f"Arithmetic expression: {expr}\n"
                "Copy the full arithmetic expression into the calculator expression argument. Do not compute it mentally.\n\n"
                f"Assistant:\n{calculator_call(expr)}\n\n"
                f"Tool result from calculator: {result}\n\n"
                "Assistant:\n"
            )
            dump_record(handle, prompt, f"The answer is {result}.")

    _write_tool_result_final = _write_final
    _write_final_replay = _write_final

    def _write_reflect(self, handle, count: int, rng: random.Random) -> None:
        written = 0
        while written < count:
            expr, result, bad_options = make_expression(rng)
            if not bad_options:
                continue
            bad = rng.choice(bad_options)
            question = f"What is {expr}?"
            bad_call = calculator_call(bad)
            issue = rng.choice(REFLECT_TEMPLATES).format(bad=bad, good=expr, question=question)
            prompt = (
                f"User:\nUse the calculator tool to answer: {question}\n"
                f"Arithmetic expression: {expr}\n"
                "Copy the full arithmetic expression into the calculator expression argument. Do not compute it mentally.\n\n"
                f"Assistant:\n{bad_call}\n\n"
                "Tool result from calculator: wrong_or_incomplete\n\n"
                f"Assistant:\n{reflect_call(question, bad_call)}\n\n"
                f"Tool result from reflect: {reflect_result(issue, expr)}\n\n"
                f"Assistant:\n{calculator_call(expr)}\n\n"
                f"Tool result from calculator: {result}\n\n"
                "Assistant:\n"
            )
            dump_record(handle, prompt, f"The answer is {result}.")
            written += 1
