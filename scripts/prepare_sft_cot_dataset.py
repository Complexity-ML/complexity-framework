#!/usr/bin/env python3
"""Convert CoT instruction datasets to the JSONL format used by local SFT."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def message_content(message: Any) -> str:
    if isinstance(message, dict):
        return as_text(message.get("content"))
    return as_text(message)


def openr1_record(example: dict[str, Any], trace_index: int) -> tuple[str, str] | None:
    problem = as_text(example.get("problem") or example.get("question") or example.get("prompt"))
    if not problem and isinstance(example.get("messages"), list):
        for message in example["messages"]:
            if isinstance(message, dict) and message.get("role") in {"user", "human"}:
                problem = message_content(message)
                break

    solution = as_text(example.get("solution"))
    generations = example.get("generations")
    if not solution and isinstance(generations, list) and generations:
        generation = generations[min(trace_index, len(generations) - 1)]
        solution = message_content(generation)

    if not solution and isinstance(example.get("messages"), list):
        for message in reversed(example["messages"]):
            if isinstance(message, dict) and message.get("role") in {"assistant", "gpt"}:
                solution = message_content(message)
                break

    answer = as_text(example.get("answer"))
    if not problem or not solution:
        return None
    if answer and answer not in solution:
        solution = f"{solution}\n\nFinal answer: {answer}"
    return problem, solution


def generic_record(example: dict[str, Any], trace_index: int) -> tuple[str, str] | None:
    parsed = openr1_record(example, trace_index)
    if parsed is not None:
        return parsed

    instruction = as_text(example.get("instruction"))
    extra_input = as_text(example.get("input"))
    output = as_text(example.get("output") or example.get("response") or example.get("completion"))
    if instruction and output:
        prompt = instruction if not extra_input else f"{instruction}\n\n{extra_input}"
        return prompt, output
    return None


def main():
    parser = argparse.ArgumentParser(description="Prepare CoT SFT JSONL")
    parser.add_argument("--dataset", default="open-r1/OpenR1-Math-220k")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--trace-index",
        type=int,
        default=0,
        help="Which generation trace to use when the dataset has multiple CoT traces.",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, args.config, split=args.split)
    indices = list(range(len(dataset)))
    if args.shuffle:
        random.Random(args.seed).shuffle(indices)
    if args.max_samples is not None:
        indices = indices[: args.max_samples]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for idx in indices:
            parsed = generic_record(dataset[idx], args.trace_index)
            if parsed is None:
                skipped += 1
                continue
            problem, solution = parsed
            record = {
                "prompt": f"User:\n{problem}\n\nAssistant:\n",
                "completion": solution,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written:,} records to {out_path}")
    if skipped:
        print(f"Skipped {skipped:,} malformed records")


if __name__ == "__main__":
    main()
