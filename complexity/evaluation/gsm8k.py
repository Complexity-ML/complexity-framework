"""Generative GSM8K prompts and numeric grading with pinned test provenance.

Dataset and Parquet runtimes are imported only for their requested loader;
generation is supplied by a callable, never a native decoding loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

FINAL_PATTERNS = (
    re.compile(r"final\s+answer\s*[:=]\s*\$?\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"####\s*([^\n]+)"),
    re.compile(r"\\boxed\{([^{}]+)\}"),
    re.compile(r"the\s+answer\s+is\s*\$?\s*([^\n]+)", re.IGNORECASE),
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
    from .arc import final_answer_segment

    text = final_answer_segment(text)
    for marker in ("<|think_end|>", "</think>"):
        if marker in text:
            text = text.rsplit(marker, 1)[1]
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
        "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
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
    system_prompt: str = GSM8K_SYSTEM_PROMPT,
) -> str:
    from complexity.inference.chat_template import render_messages_before_assistant

    if not 0 <= num_fewshot <= len(GSM8K_COT_FEWSHOT):
        raise ValueError(f"num_fewshot must be between 0 and {len(GSM8K_COT_FEWSHOT)}")
    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
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
        import pyarrow.parquet as pq

        return [
            {"question": str(row["question"]), "answer": str(row["answer"])}
            for row in pq.read_table(test_parquet, columns=["question", "answer"]).to_pylist()
        ]
    from datasets import load_dataset

    return list(load_dataset("openai/gsm8k", "main", split=split, revision=DATASET_REVISION))


def ground_truth(answer: str) -> Decimal:
    marker = answer.rsplit("####", 1)[-1]
    parsed = extract_answer("#### " + marker, allow_last_number=False)
    if parsed is None:
        raise ValueError(f"cannot parse GSM8K reference answer: {answer!r}")
    return parsed


def evaluate(
    examples, generator, chat_template, *, num_fewshot=8, seed=0, output=None, provenance=None
):
    """Grade numeric answers with retained prompts/traces and explicit protocol."""
    from .arc import generate_completion, write_report

    if not examples:
        raise ValueError("GSM8K selection is empty")
    started, rows = time.monotonic(), []
    for index, example in enumerate(examples):
        prompt = build_gsm8k_prompt(example["question"], chat_template, num_fewshot=num_fewshot)
        completion = generate_completion(generator, prompt, seed=seed + index, allowed=())
        predicted, expected = extract_answer(completion), ground_truth(example["answer"])
        rows.append(
            {
                "index": index,
                "question": example["question"],
                "prompt": prompt,
                "response": completion,
                "expected": str(expected),
                "predicted": str(predicted) if predicted is not None else None,
                "correct": predicted == expected,
            }
        )
    report = {
        "schema_version": 1,
        "benchmark": "openai/gsm8k",
        "split": "test",
        "complete": True,
        "protocol": "native_chat_numeric_answer_extraction",
        "chat_template_applied": True,
        "num_fewshot": num_fewshot,
        "fewshot_sha256": fewshot_digest(),
        "seed": seed,
        "provenance": dict(provenance or {}),
        "metrics": {
            "examples": len(rows),
            "parsed": sum(row["predicted"] is not None for row in rows),
            "parse_rate": sum(row["predicted"] is not None for row in rows) / len(rows),
            "correct": sum(row["correct"] for row in rows),
            "accuracy": sum(row["correct"] for row in rows) / len(rows),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        },
    }
    if output is not None:
        write_report(output, report, rows)
    else:
        report["records"] = rows
    return report


def main() -> None:
    from .arc import serving_generator

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--test-parquet", type=Path)
    parser.add_argument("--chat-template", type=Path, required=True)
    parser.add_argument("--backend", choices=("vllm", "sglang"), default="vllm")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-fewshot", type=int, choices=range(9), default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Generation length and sample limit must be positive")
    if args.probe and args.probe.resolve() == args.output.with_suffix(".jsonl").resolve():
        parser.error("Output traces must differ from the source probe")
    examples = load_examples(args.probe, args.test_parquet, split="test")
    selected = examples if args.limit is None else examples[: args.limit]
    template = json.loads(args.chat_template.read_text())
    generator = serving_generator(
        backend=args.backend,
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_new_tokens,
        stop=list(
            dict.fromkeys(
                filter(None, (template.get("end_of_turn_token"), template.get("eos_token")))
            )
        ),
    )
    source = args.probe or args.test_parquet
    provenance = {
        "backend": args.backend,
        "model": args.model,
        "dataset_revision": DATASET_REVISION if source is None else None,
        "local_source": str(source.resolve()) if source else None,
        "local_source_sha256": sha256_file(source) if source else None,
        "chat_template_sha256": sha256_file(args.chat_template),
        "decoding": {"max_new_tokens": args.max_new_tokens, "temperature": 0.0, "top_p": 1.0},
    }
    print(
        json.dumps(
            evaluate(
                selected,
                generator,
                template,
                num_fewshot=args.num_fewshot,
                output=args.output,
                provenance=provenance,
            ),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
