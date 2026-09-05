#!/usr/bin/env python3
"""Evaluate a native TR-HASH checkpoint on generative GSM8K."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pyarrow.parquet as pq
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.chat_generate_local import (  # noqa: E402
    build_prompt,
    generate_chat,
    load_model,
    pick_device,
)

FINAL_PATTERNS = (
    re.compile(r"final\s+answer\s*[:=]\s*\$?\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"####\s*([^\n]+)"),
    re.compile(r"\\boxed\{([^{}]+)\}"),
)
NUMBER_PATTERN = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?(?:/[+-]?\d[\d,]*(?:\.\d+)?)?")


def normalize_number(value: str) -> Decimal | None:
    cleaned = value.strip().replace("$", "").replace(",", "")
    cleaned = cleaned.rstrip(". ")
    if "/" in cleaned:
        numerator, denominator = cleaned.split("/", 1)
        try:
            return Decimal(numerator) / Decimal(denominator)
        except (InvalidOperation, ZeroDivisionError):
            return None
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def extract_answer(text: str, *, allow_last_number: bool = True) -> Decimal | None:
    for pattern in FINAL_PATTERNS:
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        numbers = NUMBER_PATTERN.findall(matches[-1].group(1))
        if numbers:
            return normalize_number(numbers[-1])
    if allow_last_number:
        numbers = NUMBER_PATTERN.findall(text)
        if numbers:
            return normalize_number(numbers[-1])
    return None


DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_examples(
    probe: Path | None,
    test_parquet: Path | None,
    *,
    split: str,
) -> list[dict[str, str]]:
    if split != "test":
        raise ValueError("supervised GSM8K checkpoint evaluation is restricted to split=test")
    if probe is not None and test_parquet is not None:
        raise ValueError("pass either --probe or --test-parquet, not both")
    if probe is not None:
        return [
            json.loads(line)
            for line in probe.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if test_parquet is not None:
        return [
            {"question": str(row["question"]), "answer": str(row["answer"])}
            for row in pq.read_table(test_parquet, columns=["question", "answer"]).to_pylist()
        ]
    from datasets import load_dataset

    return list(load_dataset("openai/gsm8k", "main", split=split))


def ground_truth(answer: str) -> Decimal:
    marker = answer.rsplit("####", 1)[-1]
    parsed = extract_answer("#### " + marker, allow_last_number=False)
    if parsed is None:
        raise ValueError(f"cannot parse GSM8K reference answer: {answer!r}")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--test-parquet", type=Path)
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument(
        "--experiment-label",
        default="supervised_gsm8k_sft",
        choices=("refinement_baseline", "supervised_gsm8k_sft"),
    )
    args = parser.parse_args()

    if args.limit < 1 or args.offset < 0:
        parser.error("--limit must be positive and --offset must be non-negative")
    device = pick_device(args.device)
    model, tokenizer, chat_template = load_model(args.checkpoint, args.tokenizer, device)
    all_examples = load_examples(args.probe, args.test_parquet, split=args.split)
    examples = all_examples[args.offset : args.offset + args.limit]
    if not examples:
        raise ValueError("GSM8K selection is empty")

    started = time.monotonic()
    traces = []
    correct = 0
    parsed_count = 0
    prompt_contract = (
        "Solve the following math problem. Explain briefly, then end with "
        "`Final answer: <number>`.\n\n"
    )
    for local_index, example in enumerate(examples, start=1):
        prompt = build_prompt(prompt_contract + example["question"], False, chat_template)
        torch.manual_seed(0)
        response = generate_chat(
            model,
            tokenizer,
            prompt,
            device,
            args.max_new_tokens,
            0.0,
            1.0,
            0,
            args.repetition_penalty,
            128,
        )
        predicted = extract_answer(response)
        expected = ground_truth(example["answer"])
        is_correct = predicted == expected
        parsed_count += predicted is not None
        correct += is_correct
        trace = {
            "index": args.offset + local_index - 1,
            "question": example["question"],
            "expected": str(expected),
            "predicted": None if predicted is None else str(predicted),
            "correct": is_correct,
            "response": response,
        }
        traces.append(trace)
        elapsed = time.monotonic() - started
        print(
            f"{local_index:>3}/{len(examples)} expected={expected} "
            f"predicted={predicted} correct={is_correct} "
            f"acc={correct / local_index:.3f} elapsed={elapsed:.1f}s",
            flush=True,
        )

    elapsed = time.monotonic() - started
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "benchmark": "openai/gsm8k",
        "benchmark_revision": DATASET_REVISION,
        "split": args.split,
        "experiment_label": args.experiment_label,
        "supervised_on_gsm8k_train": args.experiment_label == "supervised_gsm8k_sft",
        "protocol": "zero_shot_chat_greedy_final_numeric_answer",
        "chat_template_applied": True,
        "selection": {"offset": args.offset, "limit": len(examples)},
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0.0,
            "repetition_penalty": args.repetition_penalty,
        },
        "metrics": {
            "examples": len(examples),
            "parsed": parsed_count,
            "parse_rate": parsed_count / len(examples),
            "correct": correct,
            "accuracy": correct / len(examples),
            "elapsed_seconds": round(elapsed, 3),
        },
        "traces": traces,
    }
    if args.test_parquet is not None:
        report["test_parquet_sha256"] = sha256_file(args.test_parquet)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
