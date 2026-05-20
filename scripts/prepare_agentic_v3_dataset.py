#!/usr/bin/env python3
"""Prepare a mixed assistant/tool SFT dataset.

The v3 mix is meant to repair the current "tool-brain" failure mode:
- normal chat should be natural text, not a forced tool call
- calculator calls should copy the full expression
- tool results should become final answers
- reflect repairs bad calculator expressions
- respond/final_answer is included lightly as an optional pseudo-tool
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


CHAT_EXAMPLES = [
    ("Hello", "Hello! How can I help?"),
    ("Hi", "Hi! What would you like to work on?"),
    ("Hey, are you there?", "Yes, I am here. What do you want to do?"),
    ("Thanks", "You are welcome."),
    ("Explain gravity simply.", "Gravity is the force that pulls objects with mass toward each other."),
    ("What is a GPU?", "A GPU is a processor designed to run many parallel computations efficiently."),
    ("Explain fine-tuning in one sentence.", "Fine-tuning adapts a pretrained model to a narrower behavior or dataset."),
    ("What is an optimizer in training?", "An optimizer updates model weights to reduce the training loss."),
    ("Give me a short answer: what is Python?", "Python is a popular programming language used for scripting, data, and applications."),
    ("Do I need a tool to say hello?", "No. A greeting can be answered directly."),
]

RESPOND_EXAMPLES = [
    ("Hello", "Hello! How can I help?"),
    ("Say hi in one short sentence.", "Hi! What are we building today?"),
    ("I just want to chat.", "Sure. I can chat without using external tools."),
    ("Tell me you are ready.", "Ready."),
]

REFLECT_TEMPLATES = [
    "calculator expression {bad!r} does not match the user question {question!r}; use the full expression {good!r}",
    "the tool call dropped part of the arithmetic expression; use {good!r}",
    "the calculator argument is incomplete; replace it with {good!r}",
]


def dump_record(handle, prompt: str, completion: str) -> None:
    handle.write(
        json.dumps(
            {
                "prompt": prompt.rstrip() + "\n",
                "completion": completion.strip(),
            },
            ensure_ascii=False,
        )
        + "\n"
    )


def chat_prompt(user: str) -> str:
    return f"User:\n{user}\n\nAssistant:\n"


def calculator_call(expression: str) -> str:
    return '<tool_call>{"name":"calculator","arguments":{"expression":"' + expression + '"}}</tool_call>'


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
    payload = {
        "name": "reflect",
        "result": {
            "issue": issue,
            "corrected_expression": corrected_expression,
        },
    }
    return json.dumps(payload, separators=(",", ":"))


def respond_call(text: str) -> str:
    payload = {"name": "respond", "arguments": {"text": text}}
    return f"<tool_call>{json.dumps(payload, separators=(',', ':'))}</tool_call>"


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
        expr = f"{a}+{b}"
        value = a + b
    elif kind == 1:
        expr = f"{a}-{b}"
        value = a - b
    elif kind == 2:
        expr = f"{a}*{c}"
        value = a * c
    elif kind == 3:
        value = a
        expr = f"{a * c}/{c}"
    elif kind == 4:
        expr = f"{a}-{b}+{c}"
        value = a - b + c
        bad = [f"{a}-{b}", f"{b}+{c}"]
    elif kind == 5:
        expr = f"{a}+{b}*{c}"
        value = a + b * c
        bad = [f"{a}+{b}", f"{b}*{c}"]
    elif kind == 6:
        base = rng.randint(-30, 30)
        b = rng.randint(1, 99)
        a = b + base * c
        expr = f"({a}-{b})/{c}"
        value = base
        bad = [f"{a}-{b}", f"{b}/{c}", f"{c}-{b}"]
    else:
        expr = f"({a}+{b})*{c}"
        value = (a + b) * c
        bad = [f"{a}+{b}", f"{b}*{c}"]
    return expr, fmt_number(value), bad


def write_chat(handle, count: int, rng: random.Random) -> None:
    for _ in range(count):
        user, answer = rng.choice(CHAT_EXAMPLES)
        dump_record(handle, chat_prompt(user), answer)


def write_respond(handle, count: int, rng: random.Random) -> None:
    for _ in range(count):
        user, answer = rng.choice(RESPOND_EXAMPLES)
        dump_record(handle, chat_prompt(user), respond_call(answer))


def write_calculator_call(handle, count: int, rng: random.Random) -> None:
    for _ in range(count):
        expr, _, _ = make_expression(rng)
        question = rng.choice(
            [
                f"What is {expr}?",
                f"Calculate {expr}.",
                f"Use the calculator for {expr}.",
            ]
        )
        prompt = (
            f"User:\nUse the calculator tool to answer: {question}\n"
            f"Arithmetic expression: {expr}\n"
            "Copy the full arithmetic expression into the calculator expression argument. "
            "Do not compute it mentally.\n\n"
            "Assistant:\n"
        )
        dump_record(handle, prompt, calculator_call(expr))


def write_tool_result_final(handle, count: int, rng: random.Random) -> None:
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


def write_reflect_repair(handle, count: int, rng: random.Random) -> None:
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
            f"Tool result from calculator: wrong_or_incomplete\n\n"
            f"Assistant:\n{reflect_call(question, bad_call)}\n\n"
            f"Tool result from reflect: {reflect_result(issue, expr)}\n\n"
            "Assistant:\n"
        )
        dump_record(handle, prompt, calculator_call(expr))

        final_prompt = (
            f"User:\nUse the calculator tool to answer: {question}\n"
            f"Arithmetic expression: {expr}\n"
            "Copy the full arithmetic expression into the calculator expression argument. Do not compute it mentally.\n\n"
            f"Assistant:\n{bad_call}\n\n"
            f"Tool result from calculator: wrong_or_incomplete\n\n"
            f"Assistant:\n{reflect_call(question, bad_call)}\n\n"
            f"Tool result from reflect: {reflect_result(issue, expr)}\n\n"
            f"Assistant:\n{calculator_call(expr)}\n\n"
            f"Tool result from calculator: {result}\n\n"
            "Assistant:\n"
        )
        dump_record(handle, final_prompt, f"The answer is {result}.")
        written += 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare mixed natural-chat + agentic tool SFT data")
    parser.add_argument("--out", default="data/sft/agentic_v3_mixed.jsonl")
    parser.add_argument("--records", type=int, default=80_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chat-ratio", type=float, default=0.40)
    parser.add_argument("--calculator-ratio", type=float, default=0.25)
    parser.add_argument("--final-ratio", type=float, default=0.25)
    parser.add_argument("--reflect-ratio", type=float, default=0.08)
    parser.add_argument("--respond-ratio", type=float, default=0.02)
    args = parser.parse_args()

    ratios = [
        ("chat", args.chat_ratio),
        ("calculator", args.calculator_ratio),
        ("final", args.final_ratio),
        ("reflect", args.reflect_ratio),
        ("respond", args.respond_ratio),
    ]
    total_ratio = sum(ratio for _, ratio in ratios)
    if total_ratio <= 0:
        raise ValueError("ratios must sum to a positive value")

    counts = {name: int(args.records * ratio / total_ratio) for name, ratio in ratios}
    counts["chat"] += args.records - sum(counts.values())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    with out.open("w", encoding="utf-8") as handle:
        writers = [
            (write_chat, counts["chat"]),
            (write_calculator_call, counts["calculator"]),
            (write_tool_result_final, counts["final"]),
            (write_reflect_repair, counts["reflect"]),
            (write_respond, counts["respond"]),
        ]
        for writer, count in writers:
            writer(handle, count, rng)

    # Shuffle the finished file so the SFT stream is mixed before per-epoch shuffling.
    lines = out.read_text(encoding="utf-8").splitlines()
    rng.shuffle(lines)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {len(lines):,} records to {out}")
    for name in sorted(counts):
        print(f"{name}: {counts[name]:,}")


if __name__ == "__main__":
    main()
