#!/usr/bin/env python3
"""Prepare a v5 dataset for "reflect-as-bootstrap" agentic behavior.

Goal: teach the model to accept *unscaffolded* natural-language questions
and bootstrap its own scaffold by emitting a reflect tool call.

Pipeline the model learns:

    User:
    What is 17 + 25?

    Assistant:
    <tool_call>{"name":"reflect","arguments":{"question":"What is 17 + 25?","task":"identify_tool_and_args"}}</tool_call>

    Tool result from reflect: {"tool":"calculator","arguments":{"expression":"17+25"}}

    Assistant:
    <tool_call>{"name":"calculator","arguments":{"expression":"17+25"}}</tool_call>

    Tool result from calculator: 42

    Assistant:
    The answer is 42.

The orchestrator (Python side) is responsible for the actual extraction
in the reflect step. The model only needs to:
- recognize a tool-shaped question and emit a reflect call (copy the
  question verbatim)
- copy the "arguments" object from the reflect result into a real tool
  call
- finalize the tool result in natural language

Chat-shaped questions skip reflect entirely (model answers directly).
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path


WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

CHAT_EXAMPLES = [
    ("Hello", "Hello! How can I help?"),
    ("Hi", "Hi! What would you like to work on?"),
    ("Hey, are you there?", "Yes, I am here. What do you want to do?"),
    ("Thanks", "You are welcome."),
    ("Explain gravity simply.", "Gravity is the force that pulls objects with mass toward each other."),
    ("What is a GPU?", "A GPU is a processor designed to run many parallel computations efficiently."),
    ("Explain fine-tuning in one sentence.", "Fine-tuning adapts a pretrained model to a narrower behavior or dataset."),
    ("Give me a short answer: what is Python?", "Python is a popular programming language used for scripting, data, and applications."),
    ("Do I need a tool to say hello?", "No. A greeting can be answered directly."),
]

# Date- or number-shaped chat that must stay chat (no reflect, no tool).
ROUTING_NEGATIVES = [
    ("I have 30 days of vacation this year.", "That sounds great. Enjoy the time off."),
    ("My birthday is in May.", "Nice. May is a pleasant month."),
    ("I was born in 1990.", "Thanks for sharing."),
    ("Mondays are tough.", "They can be. A short plan helps."),
    ("The meeting is next week.", "Got it. Anything you want to prepare?"),
    ("I love numbers.", "Numbers are useful for many things."),
    ("January is cold.", "Yes, January tends to be cold in many regions."),
    ("Time flies.", "It does. Staying focused helps."),
]


def dump_record(handle, prompt: str, completion: str) -> None:
    handle.write(
        json.dumps(
            {"prompt": prompt.rstrip() + "\n", "completion": completion.strip()},
            ensure_ascii=False,
        )
        + "\n"
    )


def chat_prompt(user: str) -> str:
    return f"User:\n{user}\n\nAssistant:\n"


def reflect_call(question: str) -> str:
    payload = {
        "name": "reflect",
        "arguments": {"question": question, "task": "identify_tool_and_args"},
    }
    return f"<tool_call>{json.dumps(payload, separators=(',', ':'))}</tool_call>"


def reflect_result(tool: str, arguments: dict) -> str:
    payload = {"tool": tool, "arguments": arguments}
    return json.dumps(payload, separators=(",", ":"))


def real_tool_call(tool: str, arguments: dict) -> str:
    payload = {"name": tool, "arguments": arguments}
    return f"<tool_call>{json.dumps(payload, separators=(',', ':'))}</tool_call>"


def fmt_number(value: float | int) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


# --- calculator NL questions ------------------------------------------------------

def make_calc_case(rng: random.Random) -> dict:
    a = rng.randint(1, 99)
    b = rng.randint(1, 99)
    c = rng.randint(1, 20)
    kind = rng.randrange(6)
    if kind == 0:
        expr, value = f"{a}+{b}", a + b
    elif kind == 1:
        expr, value = f"{a}-{b}", a - b
    elif kind == 2:
        expr, value = f"{a}*{c}", a * c
    elif kind == 3:
        expr, value = f"{a * c}/{c}", a
    elif kind == 4:
        expr, value = f"{a}-{b}+{c}", a - b + c
    else:
        expr, value = f"{a}+{b}*{c}", a + b * c
    pretty = expr.replace("*", " * ").replace("/", " / ").replace("+", " + ").replace("-", " - ")
    question = rng.choice(
        [
            f"What is {pretty}?",
            f"Calculate {pretty}.",
            f"Compute {pretty}.",
            f"How much is {pretty}?",
            f"Tell me {pretty}.",
        ]
    )
    return {
        "question": question,
        "tool": "calculator",
        "arguments": {"expression": expr},
        "result": fmt_number(value),
        "final": f"The answer is {fmt_number(value)}.",
    }


# --- datetime NL questions --------------------------------------------------------

def random_date(rng: random.Random) -> date:
    start = date(2020, 1, 1).toordinal()
    end = date(2029, 12, 31).toordinal()
    return date.fromordinal(rng.randint(start, end))


def make_datetime_case(rng: random.Random) -> dict:
    op = rng.choice(["diff", "add_days", "weekday"])
    if op == "diff":
        a = random_date(rng)
        b = a + timedelta(days=rng.randint(-365, 365))
        if rng.random() < 0.5:
            a, b = b, a
        args = {"a": a.isoformat(), "b": b.isoformat()}
        result = str((b - a).days)
        question = rng.choice(
            [
                f"How many days between {a.isoformat()} and {b.isoformat()}?",
                f"What is the day difference between {a.isoformat()} and {b.isoformat()}?",
                f"From {a.isoformat()} to {b.isoformat()}, how many days?",
            ]
        )
        final = f"The answer is {result}."
    elif op == "add_days":
        base = random_date(rng)
        days = rng.randint(-180, 365)
        target = base + timedelta(days=days)
        args = {"date": base.isoformat(), "days": days}
        result = target.isoformat()
        question = rng.choice(
            [
                f"What is {base.isoformat()} plus {days} days?",
                f"Add {days} days to {base.isoformat()}.",
                f"Which date is {days} days after {base.isoformat()}?",
            ]
        )
        final = f"The answer is {result}."
    else:  # weekday
        d = random_date(rng)
        name = WEEKDAYS[d.weekday()]
        args = {"date": d.isoformat()}
        result = name
        question = rng.choice(
            [
                f"What day of the week is {d.isoformat()}?",
                f"Which weekday falls on {d.isoformat()}?",
            ]
        )
        final = f"The answer is {result}."
    return {
        "question": question,
        "tool": "datetime",
        "arguments": {"op": op, **args},
        "result": result,
        "final": final,
    }


# --- three-slice writers per tool case --------------------------------------------

def slice_reflect_call(case: dict) -> tuple[str, str]:
    prompt = chat_prompt(case["question"])
    completion = reflect_call(case["question"])
    return prompt, completion


def slice_tool_call_from_reflect(case: dict) -> tuple[str, str]:
    rresult = reflect_result(case["tool"], case["arguments"])
    prompt = (
        chat_prompt(case["question"])
        + f"{reflect_call(case['question'])}\n\n"
        + f"Tool result from reflect: {rresult}\n\n"
        + "Assistant:\n"
    )
    completion = real_tool_call(case["tool"], case["arguments"])
    return prompt, completion


def slice_final_after_tool(case: dict) -> tuple[str, str]:
    rresult = reflect_result(case["tool"], case["arguments"])
    prompt = (
        chat_prompt(case["question"])
        + f"{reflect_call(case['question'])}\n\n"
        + f"Tool result from reflect: {rresult}\n\n"
        + f"Assistant:\n{real_tool_call(case['tool'], case['arguments'])}\n\n"
        + f"Tool result from {case['tool']}: {case['result']}\n\n"
        + "Assistant:\n"
    )
    completion = case["final"]
    return prompt, completion


def write_bootstrap(handle, count: int, rng: random.Random, factory) -> None:
    """Write balanced triples (reflect, tool_from_reflect, finalize) for one tool."""
    written = 0
    while written < count:
        case = factory(rng)
        for slicer in (slice_reflect_call, slice_tool_call_from_reflect, slice_final_after_tool):
            if written >= count:
                break
            prompt, completion = slicer(case)
            dump_record(handle, prompt, completion)
            written += 1


def write_chat(handle, count: int, rng: random.Random) -> None:
    for _ in range(count):
        user, answer = rng.choice(CHAT_EXAMPLES)
        dump_record(handle, chat_prompt(user), answer)


def write_routing_negatives(handle, count: int, rng: random.Random) -> None:
    for _ in range(count):
        user, answer = rng.choice(ROUTING_NEGATIVES)
        dump_record(handle, chat_prompt(user), answer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare v5 reflect-bootstrap SFT dataset")
    parser.add_argument("--out", default="data/sft/agentic_v5_bootstrap_mixed.jsonl")
    parser.add_argument("--records", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calc-bootstrap-ratio", type=float, default=0.35)
    parser.add_argument("--datetime-bootstrap-ratio", type=float, default=0.35)
    parser.add_argument("--chat-ratio", type=float, default=0.20)
    parser.add_argument("--routing-neg-ratio", type=float, default=0.10)
    args = parser.parse_args()

    ratios = [
        ("calc_bootstrap", args.calc_bootstrap_ratio),
        ("datetime_bootstrap", args.datetime_bootstrap_ratio),
        ("chat", args.chat_ratio),
        ("routing_neg", args.routing_neg_ratio),
    ]
    total = sum(r for _, r in ratios)
    counts = {name: int(args.records * r / total) for name, r in ratios}
    counts["chat"] += args.records - sum(counts.values())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    with out.open("w", encoding="utf-8") as handle:
        write_bootstrap(handle, counts["calc_bootstrap"], rng, make_calc_case)
        write_bootstrap(handle, counts["datetime_bootstrap"], rng, make_datetime_case)
        write_chat(handle, counts["chat"], rng)
        write_routing_negatives(handle, counts["routing_neg"], rng)

    lines = out.read_text(encoding="utf-8").splitlines()
    rng.shuffle(lines)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {len(lines):,} records to {out}")
    for name in sorted(counts):
        print(f"  {name}: {counts[name]:,}")


if __name__ == "__main__":
    main()
