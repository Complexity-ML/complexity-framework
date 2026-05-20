#!/usr/bin/env python3
"""Prepare a v4 mixed SFT dataset that adds a `datetime` tool on top of v3.

Goals:
- teach the model to route between two tools (calculator vs datetime)
- keep calculator + chat behaviour from v3 (refresh, anti-forgetting)
- include finalization step (Tool result -> natural-language answer)
- include routing negatives (date-shaped chat that should NOT call a tool)

datetime tool surface (3 ops, deterministic, easy to dataset-generate):
- diff      args: {a: YYYY-MM-DD, b: YYYY-MM-DD}  -> integer days (b - a)
- add_days  args: {date: YYYY-MM-DD, days: int}   -> YYYY-MM-DD
- weekday   args: {date: YYYY-MM-DD}              -> Monday..Sunday
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path


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

# Date/number-shaped chat that must NOT trigger a tool call.
ROUTING_NEGATIVES = [
    ("I have 30 days of vacation this year.", "That sounds great. Enjoy the time off."),
    ("My birthday is in May.", "Nice. May is a pleasant month."),
    ("I was born in 1990.", "Thanks for sharing."),
    ("Mondays are tough.", "They can be. A short plan helps."),
    ("The meeting is next week.", "Got it. Anything you want to prepare?"),
    ("I love numbers.", "Numbers are useful for many things."),
    ("January is cold.", "Yes, January tends to be cold in many regions."),
    ("Time flies.", "It does. Staying focused helps."),
    ("What is your favorite day?", "I do not have preferences, but I can help with dates."),
    ("Do you know what year it is?", "I do not have a real-time clock; ask me to compute from a known date."),
]

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


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


def datetime_call(op: str, **args) -> str:
    payload = {"name": "datetime", "arguments": {"op": op, **args}}
    return f"<tool_call>{json.dumps(payload, separators=(',', ':'))}</tool_call>"


def calculator_call(expression: str) -> str:
    return (
        '<tool_call>{"name":"calculator","arguments":{"expression":"'
        + expression
        + '"}}</tool_call>'
    )


def fmt_number(value: float | int) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def random_date(rng: random.Random) -> date:
    # Stay inside a fixed window so weekday math is deterministic and the model
    # sees a bounded distribution. 2020-01-01 .. 2029-12-31.
    start = date(2020, 1, 1).toordinal()
    end = date(2029, 12, 31).toordinal()
    return date.fromordinal(rng.randint(start, end))


# --- datetime tool-call generation -------------------------------------------------

def make_diff_case(rng: random.Random) -> tuple[str, dict, str]:
    a = random_date(rng)
    b = a + timedelta(days=rng.randint(-365, 365))
    if rng.random() < 0.5:
        a, b = b, a
    days = (b - a).days
    args = {"a": a.isoformat(), "b": b.isoformat()}
    question = rng.choice(
        [
            f"How many days between {a.isoformat()} and {b.isoformat()}?",
            f"What is the day difference between {a.isoformat()} and {b.isoformat()}?",
            f"From {a.isoformat()} to {b.isoformat()}, how many days?",
        ]
    )
    hint = f"Operation: diff. Date A: {a.isoformat()}. Date B: {b.isoformat()}."
    final_answer = f"The answer is {days} days."
    return question, {"op": "diff", "args": args, "hint": hint, "result": str(days), "final": final_answer}


def make_add_days_case(rng: random.Random) -> tuple[str, dict]:
    base = random_date(rng)
    days = rng.randint(-180, 365)
    target = base + timedelta(days=days)
    args = {"date": base.isoformat(), "days": days}
    question = rng.choice(
        [
            f"What is {base.isoformat()} plus {days} days?",
            f"Add {days} days to {base.isoformat()}.",
            f"Which date is {days} days after {base.isoformat()}?",
        ]
    )
    hint = f"Operation: add_days. Date: {base.isoformat()}. Days: {days}."
    final_answer = f"The answer is {target.isoformat()}."
    return question, {"op": "add_days", "args": args, "hint": hint, "result": target.isoformat(), "final": final_answer}


def make_weekday_case(rng: random.Random) -> tuple[str, dict]:
    d = random_date(rng)
    name = WEEKDAYS[d.weekday()]
    args = {"date": d.isoformat()}
    question = rng.choice(
        [
            f"What day of the week is {d.isoformat()}?",
            f"Which weekday falls on {d.isoformat()}?",
            f"Is {d.isoformat()} a weekend day?",
        ]
    )
    hint = f"Operation: weekday. Date: {d.isoformat()}."
    final_answer = f"The answer is {name}."
    return question, {"op": "weekday", "args": args, "hint": hint, "result": name, "final": final_answer}


DATETIME_FACTORIES = [make_diff_case, make_add_days_case, make_weekday_case]


def datetime_prompt(question: str, hint: str) -> str:
    return (
        f"User:\nUse the datetime tool to answer: {question}\n"
        f"{hint}\n"
        "Copy the dates and numbers exactly into the datetime arguments. "
        "Do not compute mentally.\n\n"
        "Assistant:\n"
    )


def write_datetime_call(handle, count: int, rng: random.Random) -> None:
    for _ in range(count):
        factory = rng.choice(DATETIME_FACTORIES)
        question, spec = factory(rng)
        prompt = datetime_prompt(question, spec["hint"])
        dump_record(handle, prompt, datetime_call(spec["op"], **spec["args"]))


def write_datetime_final(handle, count: int, rng: random.Random) -> None:
    for _ in range(count):
        factory = rng.choice(DATETIME_FACTORIES)
        question, spec = factory(rng)
        prompt = (
            datetime_prompt(question, spec["hint"])
            + datetime_call(spec["op"], **spec["args"])
            + f"\n\nTool result from datetime: {spec['result']}\n\nAssistant:\n"
        )
        dump_record(handle, prompt, spec["final"])


# --- calculator refresh (subset of v3 patterns) -----------------------------------

def make_expression(rng: random.Random) -> tuple[str, str]:
    a = rng.randint(1, 99)
    b = rng.randint(1, 99)
    c = rng.randint(1, 20)
    kind = rng.randrange(6)
    if kind == 0:
        return f"{a}+{b}", fmt_number(a + b)
    if kind == 1:
        return f"{a}-{b}", fmt_number(a - b)
    if kind == 2:
        return f"{a}*{c}", fmt_number(a * c)
    if kind == 3:
        return f"{a * c}/{c}", fmt_number(a)
    if kind == 4:
        return f"{a}-{b}+{c}", fmt_number(a - b + c)
    return f"{a}+{b}*{c}", fmt_number(a + b * c)


def write_calculator_call(handle, count: int, rng: random.Random) -> None:
    for _ in range(count):
        expr, _ = make_expression(rng)
        question = f"What is {expr}?"
        prompt = (
            f"User:\nUse the calculator tool to answer: {question}\n"
            f"Arithmetic expression: {expr}\n"
            "Copy the full arithmetic expression into the calculator expression argument. "
            "Do not compute it mentally.\n\n"
            "Assistant:\n"
        )
        dump_record(handle, prompt, calculator_call(expr))


def write_calculator_final(handle, count: int, rng: random.Random) -> None:
    for _ in range(count):
        expr, result = make_expression(rng)
        question = f"What is {expr}?"
        prompt = (
            f"User:\nUse the calculator tool to answer: {question}\n"
            f"Arithmetic expression: {expr}\n"
            "Copy the full arithmetic expression into the calculator expression argument. "
            "Do not compute it mentally.\n\n"
            f"Assistant:\n{calculator_call(expr)}\n\n"
            f"Tool result from calculator: {result}\n\n"
            "Assistant:\n"
        )
        dump_record(handle, prompt, f"The answer is {result}.")


# --- chat + routing negatives ------------------------------------------------------

def write_chat(handle, count: int, rng: random.Random) -> None:
    for _ in range(count):
        user, answer = rng.choice(CHAT_EXAMPLES)
        dump_record(handle, chat_prompt(user), answer)


def write_routing_negatives(handle, count: int, rng: random.Random) -> None:
    for _ in range(count):
        user, answer = rng.choice(ROUTING_NEGATIVES)
        dump_record(handle, chat_prompt(user), answer)


# --- main --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare v4 mixed dataset (calculator + datetime)")
    parser.add_argument("--out", default="data/sft/agentic_v4_datetime_mixed.jsonl")
    parser.add_argument("--records", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datetime-call-ratio", type=float, default=0.27)
    parser.add_argument("--datetime-final-ratio", type=float, default=0.20)
    parser.add_argument("--calc-call-ratio", type=float, default=0.17)
    parser.add_argument("--calc-final-ratio", type=float, default=0.13)
    parser.add_argument("--chat-ratio", type=float, default=0.16)
    parser.add_argument("--routing-neg-ratio", type=float, default=0.07)
    args = parser.parse_args()

    ratios = [
        ("datetime_call", args.datetime_call_ratio),
        ("datetime_final", args.datetime_final_ratio),
        ("calc_call", args.calc_call_ratio),
        ("calc_final", args.calc_final_ratio),
        ("chat", args.chat_ratio),
        ("routing_neg", args.routing_neg_ratio),
    ]
    total = sum(r for _, r in ratios)
    if total <= 0:
        raise ValueError("ratios must sum to > 0")
    counts = {name: int(args.records * r / total) for name, r in ratios}
    counts["chat"] += args.records - sum(counts.values())

    writers = {
        "datetime_call": write_datetime_call,
        "datetime_final": write_datetime_final,
        "calc_call": write_calculator_call,
        "calc_final": write_calculator_final,
        "chat": write_chat,
        "routing_neg": write_routing_negatives,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    with out.open("w", encoding="utf-8") as handle:
        for name, _ in ratios:
            writers[name](handle, counts[name], rng)

    lines = out.read_text(encoding="utf-8").splitlines()
    rng.shuffle(lines)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {len(lines):,} records to {out}")
    for name in sorted(counts):
        print(f"  {name}: {counts[name]:,}")


if __name__ == "__main__":
    main()
