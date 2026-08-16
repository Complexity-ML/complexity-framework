#!/usr/bin/env python3
"""Drive the shared online-RL loop on synthetic verified tool questions.

Loads a Complexity SFT checkpoint, streams random tool-shaped prompts through
``MPSOnlineRLEngine.infer_with_verified_tools``, and prints the RL stats each
time the shared buffer triggers an update.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from complexity.inference.mps_online_rl_engine import (
    MPSOnlineRLEngine,
    MPSOnlineRLEngineConfig,
)


OPS = ["+", "-", "*"]


def sample_calculator_question(rng: random.Random) -> str:
    a = rng.randint(1, 99)
    b = rng.randint(1, 99)
    op = rng.choice(OPS)
    template = rng.choice([
        "What is {a} {op} {b}?",
        "Calculate {a} {op} {b}.",
        "Compute {a}{op}{b}.",
        "How much is {a} {op} {b}?",
    ])
    return template.format(a=a, b=b, op=op)


def sample_datetime_question(rng: random.Random) -> str:
    base = date(2026, 1, 1) + timedelta(days=rng.randint(0, 730))
    mode = rng.choice(["diff", "add_days", "weekday"])
    if mode == "diff":
        other = base + timedelta(days=rng.randint(1, 365))
        return f"How many days between {base.isoformat()} and {other.isoformat()}?"
    if mode == "add_days":
        days = rng.randint(1, 120)
        return f"What is {base.isoformat()} plus {days} days?"
    return f"What day of the week is {base.isoformat()}?"


def sample_question(rng: random.Random, tool_mix: str) -> tuple[str, str]:
    if tool_mix == "calculator":
        return sample_calculator_question(rng), "calculator"
    if tool_mix == "datetime":
        return sample_datetime_question(rng), "datetime"
    if rng.random() < 0.65:
        return sample_calculator_question(rng), "calculator"
    return sample_datetime_question(rng), "datetime"


def compact_response(text: str, max_chars: int = 220) -> str:
    for stop in ("</tool_call>", "<|endoftext|>", "\nUser:", "\nTool:"):
        if stop in text:
            idx = text.index(stop) + len(stop)
            text = text[:idx]
            break
    text = text.replace("\n", "\\n")
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, default=REPO_ROOT / "tokenizer-o200k")
    ap.add_argument("--output-dir", type=Path, default=Path("runs/online_rl_calculator"))
    ap.add_argument("--iterations", type=int, default=128, help="0 = run forever")
    ap.add_argument("--min-events", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-7)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--tool-mix", choices=["mixed", "calculator", "datetime"], default="mixed")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    engine = MPSOnlineRLEngine(MPSOnlineRLEngineConfig(
        checkpoint=args.checkpoint,
        tokenizer=args.tokenizer,
        output_dir=args.output_dir,
        min_events_before_update=args.min_events,
        learning_rate=args.lr,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    ))

    rng = random.Random(args.seed)
    import itertools
    step_iter = itertools.count(1) if args.iterations == 0 else range(1, args.iterations + 1)
    for step in step_iter:
        question, expected_tool = sample_question(rng, args.tool_mix)
        response, stats, ckpt = engine.infer_with_verified_tools(question)
        print(
            f"[{step:04d}] tool={expected_tool:<10} q={question!r:<58} "
            f"-> {compact_response(response)!r}",
            flush=True,
        )
        if stats is not None:
            print(
                f"        update: events={stats['num_events']:.0f} "
                f"mean_reward={stats['mean_reward']:.3f} "
                f"baseline={stats['baseline']:.3f} "
                f"loss={stats['loss']:.4f} "
                f"grad_norm={stats['grad_norm']:.3f}",
                flush=True,
            )
            if ckpt is not None:
                print(f"        ckpt: {ckpt}", flush=True)


if __name__ == "__main__":
    main()
