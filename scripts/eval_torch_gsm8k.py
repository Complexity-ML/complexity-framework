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

from complexity.inference.chat_template import render_messages_before_assistant  # noqa: E402
from scripts.chat_generate_local import (  # noqa: E402
    generate_chat,
    load_model,
    pick_device,
)

FINAL_PATTERNS = (
    re.compile(r"the\s+answer\s+is\s*\$?\s*([^\n]+)", re.IGNORECASE),
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
LM_EVAL_REVISION = "b954108c9baaaa934b4ad842033b31a97ee30816"
GSM8K_SYSTEM_PROMPT = "Solve step by step. End with: The answer is <number>."
GSM8K_COT_FEWSHOT = (
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the grove "
        "today. After they are done, there will be 21 trees. How many trees did the "
        "grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some more were "
        "planted. So there must have been 21 - 15 = 6. The answer is 6.",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars "
        "are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces "
        "do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had "
        "32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 "
        "lollipops. How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. "
        "So he gave Denny 20 - 12 = 8. The answer is 8.",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. "
        "How many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then "
        "that is 4 more toys. 5 + 4 = 9. The answer is 9.",
    ),
    (
        "There were nine computers in the server room. Five more computers were "
        "installed each day, from Monday to Thursday. How many computers are now in "
        "the server room?",
        "There were originally 9 computers. For each of 4 days, 5 more computers were "
        "added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.",
    ),
    (
        "Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, "
        "he lost 2 more. How many golf balls did he have at the end of Wednesday?",
        "Michael started with 58 golf balls. After losing 23 on Tuesday, he had "
        "58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer "
        "is 33.",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money does she "
        "have left?",
        "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
        "So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.",
    ),
)


def fewshot_digest() -> str:
    payload = json.dumps(GSM8K_COT_FEWSHOT, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def build_gsm8k_prompt(
    question: str,
    chat_template: dict,
    *,
    num_fewshot: int,
) -> str:
    if not 0 <= num_fewshot <= len(GSM8K_COT_FEWSHOT):
        raise ValueError(f"num_fewshot must be between 0 and {len(GSM8K_COT_FEWSHOT)}")
    messages = [{"role": "system", "content": GSM8K_SYSTEM_PROMPT}]
    for demo_question, demo_answer in GSM8K_COT_FEWSHOT[:num_fewshot]:
        messages.extend(
            (
                {"role": "user", "content": demo_question},
                {"role": "assistant", "content": demo_answer},
            )
        )
    messages.append({"role": "user", "content": question})
    return render_messages_before_assistant(messages, chat_template)


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
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--num-fewshot", type=int, choices=range(9), default=8)
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
    end_of_turn_ids = tokenizer.encode("<|end_of_turn|>", add_special_tokens=False)
    if len(end_of_turn_ids) != 1:
        raise ValueError(f"native end-of-turn marker is not atomic: {end_of_turn_ids}")
    end_of_turn_id = end_of_turn_ids[0]
    all_examples = load_examples(args.probe, args.test_parquet, split=args.split)
    examples = all_examples[args.offset : args.offset + args.limit]
    if not examples:
        raise ValueError("GSM8K selection is empty")

    started = time.monotonic()
    traces = []
    correct = 0
    parsed_count = 0
    for local_index, example in enumerate(examples, start=1):
        prompt = build_gsm8k_prompt(
            example["question"],
            chat_template,
            num_fewshot=args.num_fewshot,
        )
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
            stop_token_ids=(end_of_turn_id,),
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
            "prompt_tokens": len(tokenizer.encode(prompt, add_special_tokens=False)),
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
        "protocol": "gsm8k_cot_8shot_native_chat_short_system_greedy",
        "chat_template_applied": True,
        "fresh_context_per_test_example": True,
        "generation_stops": ["tokenizer_eos", "<|end_of_turn|>"],
        "system_prompt": GSM8K_SYSTEM_PROMPT,
        "fewshot": {
            "count": args.num_fewshot,
            "sampler": "canonical_fixed_samples",
            "source": "EleutherAI/lm-evaluation-harness/lm_eval/tasks/gsm8k/gsm8k-cot.yaml",
            "source_revision": LM_EVAL_REVISION,
            "sha256": fewshot_digest(),
        },
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
