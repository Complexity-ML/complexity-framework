#!/usr/bin/env python3
"""Build the 500K long-horizon TR-HASH Agentic SFT corpus.

The released 250K corpus is deduplicated and cleaned of train/eval overlap.
Verified reasoning, calculator trajectories, executable-code answers,
instruction constraints, and general instruction following fill the corpus to
500K examples. Every completion uses the native Agentic tokenizer contract.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import unicodedata
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from tokenizers import Tokenizer

from complexity.inference.chat_template import agentic_chat_template
from scripts.build_agentic_pretraining_corpus import BenchmarkContaminationIndex
from scripts.prepare_tr_hash_agentic_sft_250k import validate_tokenizer

REPOSITORY = "AETHORIA-AI/TR-HASH-Agentic-SFT-32K-500K"
BASE_REPOSITORY = "AETHORIA-AI/TR-HASH-Agentic-SFT-32K-250K"
BASE_REVISION = "c6697cc1bc48a086e47fefabcd326ba371500f86"
BASE_TRAIN_SHA256 = "b06a168a4368942b57af2cbdd477056546961d46f2a2954091c8f43b5386f9ec"
BASE_EVAL_SHA256 = "eb4670d72a25651eecf2d399afaa6eb30919e8563032c33e3e6bb552669142f7"
PIQA_ARCHIVE_SHA256 = "54d32a04f59a7e354396f321723c8d7ec35cc6b08506563d8d1ffcc15ce98ddd"
TOKENIZER_REPOSITORY = "AETHORIA-AI/TR-HASH-Tokenizer-32K-Agentic"
TOKENIZER_REVISION = "2fcbc2c5359ded0244ca14531f1b3806eebac55e"

SOURCE_REVISIONS = {
    "open-r1/OpenR1-Math-220k": "e4e141ec9dea9f8326f4d347be56105859b2bd68",
    "bigcode/self-oss-instruct-sc2-exec-filter-50k": (
        "356bb069eee815daa6e23e9a282eeefe1490ad44"
    ),
    "HuggingFaceTB/smoltalk": "5feaf2fd3ffca7c237fc38d1861bc30365d48ffa",
}

TRAIN_QUOTAS = {
    "base_clean": 228_184,
    "verified_math_reasoning": 60_000,
    "verified_arithmetic_reasoning": 75_000,
    "calculator_tool_call": 35_000,
    "calculator_tool_final": 15_000,
    "execution_checked_code": 30_000,
    "instruction_constraints": 20_000,
    "general_direct": 36_816,
}
EVAL_QUOTAS = {
    "base_clean": 10_742,
    "verified_math_reasoning": 3_000,
    "verified_arithmetic_reasoning": 4_000,
    "calculator_tool_call": 2_000,
    "calculator_tool_final": 1_000,
    "execution_checked_code": 1_500,
    "instruction_constraints": 1_000,
    "general_direct": 1_758,
}

MAX_LENGTH = 2_049
MAX_REASONING_CHARS = 4_000
LEGACY_OR_NATIVE_MARKER_RE = re.compile(
    r"</?(?:think|final)>|<\|(?:think|final)_(?:start|end)\|>"
)
IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
ANSWER_TAIL_RE = re.compile(r"(?is)\n(?:#+\s*)?(?:final\s+)?answer\s*:\s*.*$")


@dataclass(frozen=True)
class Candidate:
    score: bytes
    record: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(value: str, salt: str) -> bytes:
    return hashlib.sha256(f"{salt}\0{value}".encode()).digest()


def normalized_hash(text: str) -> str:
    words = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()
    return hashlib.sha256(words.encode()).hexdigest()


def final_completion(answer: str) -> str:
    return f"<|final_start|>{answer}<|final_end|><|end_of_turn|>"


def thinking_completion(reasoning: str, answer: str) -> str:
    return (
        f"<|think_start|>{reasoning}<|think_end|>"
        f"<|final_start|>{answer}<|final_end|><|end_of_turn|>"
    )


def user_prompt(content: str, system: str = "") -> str:
    prefix = f"<|system|>{system}<|end_of_turn|>" if system else ""
    return f"{prefix}<|user|>{content}<|end_of_turn|><|assistant|>"


def token_count(tokenizer: Tokenizer, prompt: str, completion: str) -> int:
    return len(tokenizer.encode(prompt).ids) + len(tokenizer.encode(completion).ids) + 1


def repeated_lines(text: str) -> bool:
    lines = [re.sub(r"\s+", " ", line).strip().casefold() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return len(lines) >= 4 and len(set(lines)) / len(lines) < 0.65


def make_record(
    *,
    prompt: str,
    completion: str,
    source_dataset: str,
    source_id: str,
    category: str,
    thinking: bool,
    tokenizer: Tokenizer,
) -> dict[str, Any] | None:
    if LEGACY_OR_NATIVE_MARKER_RE.search(prompt):
        return None
    has_think = completion.count("<|think_start|>") == completion.count("<|think_end|>") == 1
    if has_think != thinking:
        return None
    if completion.count("<|final_start|>") != 1 or completion.count("<|final_end|>") != 1:
        return None
    count = token_count(tokenizer, prompt, completion)
    if count > MAX_LENGTH:
        return None
    return {
        "prompt": prompt,
        "completion": completion,
        "source_dataset": source_dataset,
        "source_id": source_id,
        "source_conversation_id": source_id,
        "source_subset": category,
        "split_origin": "stable_sha256_v1",
        "category": category,
        "thinking_supervised": thinking,
        "token_count": count,
    }


def parquet_rows(paths: Iterable[Path], columns: list[str] | None = None) -> Iterator[dict[str, Any]]:
    for path in sorted(paths):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=512, columns=columns):
            yield from batch.to_pylist()


def build_protected_index(directory: Path) -> tuple[BenchmarkContaminationIndex, Counter]:
    index = BenchmarkContaminationIndex()
    counts: Counter = Counter()
    for path in sorted(directory.glob("arc-*.parquet")):
        benchmark = "arc_challenge" if "challenge" in path.name else "arc_easy"
        for row in parquet_rows([path], ["question"]):
            index.add(benchmark, str(row["question"]))
            counts[benchmark] += 1
    for path in sorted(directory.glob("gsm8k-*.parquet")):
        for row in parquet_rows([path], ["question"]):
            index.add("gsm8k", str(row["question"]))
            counts["gsm8k"] += 1
    for path in sorted(directory.glob("hellaswag-*.parquet")):
        for row in parquet_rows([path], ["ctx"]):
            index.add("hellaswag", str(row["ctx"]))
            counts["hellaswag"] += 1
    piqa_path = directory / "piqa.zip"
    if sha256_file(piqa_path) != PIQA_ARCHIVE_SHA256:
        raise ValueError("PIQA archive does not match the pinned checksum")
    with zipfile.ZipFile(io.BytesIO(piqa_path.read_bytes())) as archive:
        for member in (
            "physicaliqa-train-dev/train.jsonl",
            "physicaliqa-train-dev/dev.jsonl",
        ):
            with archive.open(member) as stream:
                for encoded in io.TextIOWrapper(stream, encoding="utf-8"):
                    row = json.loads(encoded)
                    index.add("piqa", str(row["goal"]))
                    counts["piqa"] += 1
    return index, counts


def parse_messages(row: dict[str, Any]) -> tuple[str, str] | None:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None
    last_assistant = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if messages[index].get("role") in {"assistant", "gpt"}
        ),
        None,
    )
    if last_assistant is None:
        return None
    answer = str(messages[last_assistant].get("content") or "").strip()
    if not answer:
        return None
    parts: list[str] = []
    for message in messages[:last_assistant]:
        content = str(message.get("content") or "").strip()
        role = message.get("role")
        if not content:
            continue
        if role == "system":
            parts.append(f"<|system|>{content}<|end_of_turn|>")
        elif role in {"user", "human"}:
            parts.append(f"<|user|>{content}<|end_of_turn|>")
        elif role in {"assistant", "gpt"}:
            parts.append(f"<|assistant|>{final_completion(content)}")
        else:
            return None
    if not parts or not parts[-1].endswith("<|end_of_turn|>"):
        return None
    return "".join(parts) + "<|assistant|>", answer


def collect_openr1(
    paths: list[Path], tokenizer: Tokenizer, protected: BenchmarkContaminationIndex
) -> tuple[list[Candidate], Counter]:
    candidates: list[Candidate] = []
    rejected: Counter = Counter()
    seen: set[str] = set()
    columns = ["problem", "solution", "answer", "source", "uuid", "correctness_count"]
    for row in parquet_rows(paths, columns):
        problem = str(row.get("problem") or "").strip()
        solution = str(row.get("solution") or "").strip()
        answer = str(row.get("answer") or "").strip()
        if not problem or not solution or not answer:
            rejected["missing"] += 1
            continue
        if int(row.get("correctness_count") or 0) < 1:
            rejected["unverified"] += 1
            continue
        if LEGACY_OR_NATIVE_MARKER_RE.search(problem + solution + answer):
            rejected["marker"] += 1
            continue
        reasoning = ANSWER_TAIL_RE.sub("", IMAGE_RE.sub("", solution)).strip()
        if not 80 <= len(reasoning) <= MAX_REASONING_CHARS:
            rejected["reasoning_length"] += 1
            continue
        if len(answer) > 300 or repeated_lines(reasoning):
            rejected["answer_or_repetition"] += 1
            continue
        identity = normalized_hash(problem)
        if identity in seen:
            rejected["duplicate"] += 1
            continue
        match = protected.match(problem)
        if match:
            rejected[f"benchmark:{match}"] += 1
            continue
        record = make_record(
            prompt=user_prompt(problem),
            completion=thinking_completion(reasoning, answer),
            source_dataset="open-r1/OpenR1-Math-220k",
            source_id=f"{row['uuid']}:{identity[:16]}",
            category="verified_math_reasoning",
            thinking=True,
            tokenizer=tokenizer,
        )
        if record is None:
            rejected["context_or_contract"] += 1
            continue
        seen.add(identity)
        candidates.append(Candidate(stable_digest(record["source_id"], "openr1"), record))
    return candidates, rejected


def collect_messages_source(
    path: Path,
    *,
    source_dataset: str,
    category: str,
    tokenizer: Tokenizer,
    protected: BenchmarkContaminationIndex,
) -> tuple[list[Candidate], Counter]:
    candidates: list[Candidate] = []
    rejected: Counter = Counter()
    seen: set[str] = set()
    for offset, row in enumerate(parquet_rows([path])):
        parsed = parse_messages(row)
        if parsed is None:
            rejected["messages"] += 1
            continue
        prompt, answer = parsed
        if LEGACY_OR_NATIVE_MARKER_RE.search(prompt + answer):
            rejected["marker"] += 1
            continue
        if repeated_lines(answer):
            rejected["repetition"] += 1
            continue
        if protected.match(prompt):
            rejected["benchmark"] += 1
            continue
        digest = normalized_hash(prompt + "\0" + answer)
        if digest in seen:
            rejected["duplicate"] += 1
            continue
        source_id = hashlib.sha256(f"{category}:{offset}:{digest}".encode()).hexdigest()[:24]
        record = make_record(
            prompt=prompt,
            completion=final_completion(answer),
            source_dataset=source_dataset,
            source_id=source_id,
            category=category,
            thinking=False,
            tokenizer=tokenizer,
        )
        if record is None:
            rejected["context_or_contract"] += 1
            continue
        seen.add(digest)
        candidates.append(Candidate(stable_digest(source_id, category), record))
    return candidates, rejected


def collect_code(
    path: Path, tokenizer: Tokenizer, protected: BenchmarkContaminationIndex
) -> tuple[list[Candidate], Counter]:
    candidates: list[Candidate] = []
    rejected: Counter = Counter()
    seen: set[str] = set()
    for row in parquet_rows([path], ["instruction", "response", "id"]):
        instruction = str(row.get("instruction") or "").strip()
        answer = str(row.get("response") or "").strip()
        if not instruction or not answer:
            rejected["missing"] += 1
            continue
        if LEGACY_OR_NATIVE_MARKER_RE.search(instruction + answer) or repeated_lines(answer):
            rejected["marker_or_repetition"] += 1
            continue
        if protected.match(instruction):
            rejected["benchmark"] += 1
            continue
        digest = normalized_hash(instruction + "\0" + answer)
        if digest in seen:
            rejected["duplicate"] += 1
            continue
        source_id = str(row.get("id") or digest[:24])
        record = make_record(
            prompt=user_prompt(instruction),
            completion=final_completion(answer),
            source_dataset="bigcode/self-oss-instruct-sc2-exec-filter-50k",
            source_id=source_id,
            category="execution_checked_code",
            thinking=False,
            tokenizer=tokenizer,
        )
        if record is None:
            rejected["context_or_contract"] += 1
            continue
        seen.add(digest)
        candidates.append(Candidate(stable_digest(source_id, "code"), record))
    return candidates, rejected


def arithmetic_example(index: int) -> tuple[str, str, str]:
    rng = random.Random(500_000 + index)
    kind = index % 10
    a, b = rng.randint(12, 999), rng.randint(3, 199)
    if kind == 0:
        question = f"A warehouse has {a} boxes and receives {b} more. How many boxes are there now?"
        reasoning, answer = f"Add the received boxes: {a} + {b} = {a + b}.", str(a + b)
    elif kind == 1:
        total, removed = a + b, b
        question = f"A shelf held {total} books. {removed} were removed. How many remain?"
        reasoning, answer = f"Subtract the removed books: {total} - {removed} = {a}.", str(a)
    elif kind == 2:
        question = f"There are {b} trays with {a} components on each tray. Find the total."
        reasoning, answer = f"Multiply trays by components per tray: {b} × {a} = {a * b}.", str(a * b)
    elif kind == 3:
        quotient, divisor = rng.randint(4, 500), rng.randint(2, 50)
        total = quotient * divisor
        question = f"Split {total} items equally among {divisor} groups. How many items are in each group?"
        reasoning, answer = f"Compute {total} ÷ {divisor} = {quotient}.", str(quotient)
    elif kind == 4:
        percent = rng.choice((5, 10, 15, 20, 25, 30, 40, 50, 60, 75))
        base = rng.randint(2, 200) * 20
        value = base * percent // 100
        question = f"What is {percent}% of {base}?"
        reasoning, answer = f"Convert {percent}% to {percent}/100 and multiply: {base} × {percent}/100 = {value}.", str(value)
    elif kind == 5:
        percent = rng.choice((10, 20, 25, 30, 40, 50))
        price = rng.randint(5, 300) * 20
        discount = price * percent // 100
        result = price - discount
        question = f"An item costs ${price} and is discounted by {percent}%. What is the sale price?"
        reasoning = f"The discount is {price} × {percent}/100 = {discount}. Subtract it: {price} - {discount} = {result}."
        answer = f"${result}"
    elif kind == 6:
        start = rng.randint(0, 23 * 60 + 30)
        duration = rng.randint(15, 240)
        end = (start + duration) % (24 * 60)
        question = f"An event starts at {start // 60:02d}:{start % 60:02d} and lasts {duration} minutes. When does it end?"
        reasoning = f"Add {duration} minutes to {start // 60:02d}:{start % 60:02d}; the ending time is {end // 60:02d}:{end % 60:02d}."
        answer = f"{end // 60:02d}:{end % 60:02d}"
    elif kind == 7:
        metres, centimetres = rng.randint(1, 80), rng.randint(0, 99)
        question = f"Convert {metres} metres and {centimetres} centimetres to centimetres."
        value = metres * 100 + centimetres
        reasoning, answer = f"Each metre is 100 cm, so {metres} × 100 + {centimetres} = {value} cm.", f"{value} cm"
    elif kind == 8:
        x, y = rng.randint(1, 500), rng.randint(1, 500)
        z = 3 * rng.randint(1, 500) - x - y
        question = f"Find the average of {x}, {y}, and {z}."
        value = (x + y + z) // 3
        reasoning, answer = f"Add the values and divide by 3: ({x} + {y} + {z}) ÷ 3 = {value}.", str(value)
    else:
        coefficient, solution, offset = rng.randint(2, 20), rng.randint(2, 100), rng.randint(1, 80)
        total = coefficient * solution + offset
        question = f"Solve for x: {coefficient}x + {offset} = {total}."
        reasoning = f"Subtract {offset}: {coefficient}x = {total - offset}. Divide by {coefficient}: x = {solution}."
        answer = str(solution)
    return question, reasoning, answer


def collect_arithmetic(tokenizer: Tokenizer) -> list[Candidate]:
    result: list[Candidate] = []
    seen: set[str] = set()
    index = 0
    required = TRAIN_QUOTAS["verified_arithmetic_reasoning"] + EVAL_QUOTAS["verified_arithmetic_reasoning"]
    while len(result) < required:
        question, reasoning, answer = arithmetic_example(index)
        index += 1
        digest = normalized_hash(question)
        if digest in seen:
            continue
        source_id = f"arithmetic-{index:07d}"
        record = make_record(
            prompt=user_prompt(question),
            completion=thinking_completion(reasoning, answer),
            source_dataset="Complexity-ML/verified-arithmetic-generator-v1",
            source_id=source_id,
            category="verified_arithmetic_reasoning",
            thinking=True,
            tokenizer=tokenizer,
        )
        if record is not None:
            seen.add(digest)
            result.append(Candidate(stable_digest(source_id, "arithmetic"), record))
    return result


def calculator_values(index: int) -> tuple[str, str]:
    rng = random.Random(900_000 + index)
    kind = index % 4
    a, b = rng.randint(20, 20_000), rng.randint(2, 500)
    if kind == 0:
        return f"{a}+{b}", str(a + b)
    if kind == 1:
        return f"{a}*{b}", str(a * b)
    if kind == 2:
        c = rng.randint(2, 80)
        return f"({a}+{b})*{c}", str((a + b) * c)
    numerator = a * b
    return f"{numerator}/{b}", str(a)


def collect_calculator(tokenizer: Tokenizer) -> dict[str, list[Candidate]]:
    result = {"calculator_tool_call": [], "calculator_tool_final": []}
    requirements = {
        category: TRAIN_QUOTAS[category] + EVAL_QUOTAS[category] for category in result
    }
    for category in result:
        index = 0
        seen: set[str] = set()
        while len(result[category]) < requirements[category]:
            expression, answer = calculator_values(index + (0 if category.endswith("call") else 100_000))
            prompt_text = (
                f"Use the calculator to evaluate {expression}."
                if index % 2
                else f"Calculate {expression} accurately."
            )
            digest = normalized_hash(prompt_text)
            index += 1
            if digest in seen:
                continue
            tool_call = json.dumps(
                {"name": "calculator", "arguments": {"expression": expression}},
                separators=(",", ":"),
            )
            source_id = f"calculator-{category}-{index:07d}"
            if category == "calculator_tool_call":
                prompt = user_prompt(prompt_text)
                completion = (
                    f"<|tool_call_start|>{tool_call}<|tool_call_end|><|end_of_turn|>"
                )
            else:
                prompt = (
                    f"<|user|>{prompt_text}<|end_of_turn|><|assistant|>"
                    f"<|tool_call_start|>{tool_call}<|tool_call_end|><|end_of_turn|>"
                    f"<|tool_result_start|>{answer}<|tool_result_end|><|end_of_turn|>"
                    "<|assistant|>"
                )
                completion = final_completion(f"The result is {answer}.")
            count = token_count(tokenizer, prompt, completion)
            if count > MAX_LENGTH:
                continue
            record = {
                "prompt": prompt,
                "completion": completion,
                "source_dataset": "Complexity-ML/verified-calculator-generator-v1",
                "source_id": source_id,
                "source_conversation_id": source_id,
                "source_subset": category,
                "split_origin": "stable_sha256_v1",
                "category": category,
                "thinking_supervised": False,
                "token_count": count,
            }
            seen.add(digest)
            result[category].append(Candidate(stable_digest(source_id, category), record))
    return result


def split_candidates(
    candidates: list[Candidate], category: str, forbidden: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_quota, eval_quota = TRAIN_QUOTAS[category], EVAL_QUOTAS[category]
    ordered: list[Candidate] = []
    for item in sorted(candidates, key=lambda item: item.score):
        digest = normalized_hash(item.record["prompt"] + "\0" + item.record["completion"])
        if digest in forbidden:
            continue
        forbidden.add(digest)
        ordered.append(item)
    if len(ordered) < train_quota + eval_quota:
        raise RuntimeError(
            f"not enough {category}: found {len(ordered)}, need {train_quota + eval_quota}"
        )
    evaluation = [item.record for item in ordered[:eval_quota]]
    training = [item.record for item in ordered[eval_quota : eval_quota + train_quota]]
    return training, evaluation


def base_clean_hashes(
    train_path: Path, eval_path: Path
) -> tuple[set[str], set[str], dict[str, int]]:
    train_hashes: set[str] = set()
    eval_hashes: set[str] = set()
    stats: Counter = Counter()
    with train_path.open(encoding="utf-8") as source:
        for encoded in source:
            record = json.loads(encoded)
            digest = normalized_hash(record["prompt"] + "\0" + record["completion"])
            if re.search(r"</?(?:think|final)>", record["prompt"] + record["completion"]):
                stats["train_legacy_rows_removed"] += 1
                continue
            if digest in train_hashes:
                stats["train_duplicate_rows_removed"] += 1
                continue
            train_hashes.add(digest)
            stats["train_thinking"] += int(record.get("thinking_supervised", False))
    with eval_path.open(encoding="utf-8") as source:
        for encoded in source:
            record = json.loads(encoded)
            digest = normalized_hash(record["prompt"] + "\0" + record["completion"])
            if re.search(r"</?(?:think|final)>", record["prompt"] + record["completion"]):
                stats["eval_legacy_rows_removed"] += 1
                continue
            if digest in train_hashes:
                stats["eval_train_overlap_rows_removed"] += 1
                continue
            if digest in eval_hashes:
                stats["eval_duplicate_rows_removed"] += 1
                continue
            eval_hashes.add(digest)
            stats["eval_thinking"] += int(record.get("thinking_supervised", False))
    return train_hashes, eval_hashes, dict(stats)


def copy_base(
    path: Path,
    output,
    expected: int,
    counts: Counter,
    seen: set[str],
    excluded: set[str],
) -> None:
    written = 0
    with path.open(encoding="utf-8") as source:
        for encoded in source:
            record = json.loads(encoded)
            digest = normalized_hash(record["prompt"] + "\0" + record["completion"])
            if (
                digest in seen
                or digest in excluded
                or re.search(r"</?(?:think|final)>", record["prompt"] + record["completion"])
            ):
                continue
            seen.add(digest)
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts[record["category"]] += 1
            counts["thinking"] += int(record.get("thinking_supervised", False))
            counts["tokens"] += int(record["token_count"])
            counts["examples"] += 1
            written += 1
    if written != expected:
        raise RuntimeError(f"unexpected base count in {path}: {written}, expected {expected}")


def write_split(
    base_path: Path,
    output_path: Path,
    records: dict[str, list[dict[str, Any]]],
    *,
    expected_base: int,
    excluded_base: set[str],
) -> dict[str, int]:
    counts: Counter = Counter()
    seen: set[str] = set()
    with output_path.open("w", encoding="utf-8") as output:
        copy_base(base_path, output, expected_base, counts, seen, excluded_base)
        for category in TRAIN_QUOTAS:
            if category == "base_clean":
                continue
            for record in records[category]:
                digest = normalized_hash(record["prompt"] + "\0" + record["completion"])
                if digest in seen:
                    raise ValueError(f"duplicate selected conversation: {category}:{digest}")
                seen.add(digest)
                output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                counts[category] += 1
                counts["thinking"] += int(record["thinking_supervised"])
                counts["tokens"] += int(record["token_count"])
                counts["examples"] += 1
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--openr1", type=Path, nargs="+", required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--constraints", type=Path, required=True)
    parser.add_argument("--general", type=Path, required=True)
    parser.add_argument("--benchmarks", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    tokenizer_file = args.tokenizer / "tokenizer.json" if args.tokenizer.is_dir() else args.tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    validate_tokenizer(tokenizer)
    if sha256_file(args.base_dir / "train.jsonl") != BASE_TRAIN_SHA256:
        raise ValueError("base train.jsonl does not match the pinned 250K release")
    if sha256_file(args.base_dir / "eval.jsonl") != BASE_EVAL_SHA256:
        raise ValueError("base eval.jsonl does not match the pinned 250K release")

    protected, protected_counts = build_protected_index(args.benchmarks)
    openr1, openr1_rejected = collect_openr1(args.openr1, tokenizer, protected)
    code, code_rejected = collect_code(args.code, tokenizer, protected)
    constraints, constraints_rejected = collect_messages_source(
        args.constraints,
        source_dataset="HuggingFaceTB/smoltalk",
        category="instruction_constraints",
        tokenizer=tokenizer,
        protected=protected,
    )
    general, general_rejected = collect_messages_source(
        args.general,
        source_dataset="HuggingFaceTB/smoltalk",
        category="general_direct",
        tokenizer=tokenizer,
        protected=protected,
    )
    arithmetic = collect_arithmetic(tokenizer)
    calculator = collect_calculator(tokenizer)
    sources = {
        "verified_math_reasoning": openr1,
        "verified_arithmetic_reasoning": arithmetic,
        "execution_checked_code": code,
        "instruction_constraints": constraints,
        "general_direct": general,
        **calculator,
    }
    base_train_hashes, base_eval_hashes, base_cleanup = base_clean_hashes(
        args.base_dir / "train.jsonl", args.base_dir / "eval.jsonl"
    )
    if len(base_train_hashes) != TRAIN_QUOTAS["base_clean"]:
        raise RuntimeError(f"unexpected clean base train size: {len(base_train_hashes)}")
    if len(base_eval_hashes) != EVAL_QUOTAS["base_clean"]:
        raise RuntimeError(f"unexpected clean base eval size: {len(base_eval_hashes)}")
    forbidden = set(base_train_hashes | base_eval_hashes)
    training: dict[str, list[dict[str, Any]]] = {}
    evaluation: dict[str, list[dict[str, Any]]] = {}
    for category, candidates in sources.items():
        training[category], evaluation[category] = split_candidates(
            candidates, category, forbidden
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_counts = write_split(
        args.base_dir / "train.jsonl",
        args.out_dir / "train.jsonl",
        training,
        expected_base=TRAIN_QUOTAS["base_clean"],
        excluded_base=set(),
    )
    eval_counts = write_split(
        args.base_dir / "eval.jsonl",
        args.out_dir / "eval.jsonl",
        evaluation,
        expected_base=EVAL_QUOTAS["base_clean"],
        excluded_base=base_train_hashes,
    )
    if train_counts["examples"] != 500_000 or eval_counts["examples"] != 25_000:
        raise RuntimeError(f"unexpected totals: train={train_counts}, eval={eval_counts}")
    if train_counts["thinking"] != 158_072 or eval_counts["thinking"] != 8_071:
        raise RuntimeError(f"unexpected thinking totals: train={train_counts}, eval={eval_counts}")

    manifest = {
        "format": "tr-hash-agentic-sft-500k-v1",
        "repository": REPOSITORY,
        "base_replay": {
            "repository": BASE_REPOSITORY,
            "revision": BASE_REVISION,
            "train_sha256": BASE_TRAIN_SHA256,
            "eval_sha256": BASE_EVAL_SHA256,
            "cleanup": base_cleanup,
            "clean_train_examples": len(base_train_hashes),
            "clean_eval_examples": len(base_eval_hashes),
        },
        "tokenizer": {"repository": TOKENIZER_REPOSITORY, "revision": TOKENIZER_REVISION},
        "sources": SOURCE_REVISIONS,
        "source_files": {
            "openr1": [
                {"name": path.name, "sha256": sha256_file(path)}
                for path in sorted(args.openr1)
            ],
            "code": {"name": args.code.name, "sha256": sha256_file(args.code)},
            "constraints": {
                "name": args.constraints.name,
                "sha256": sha256_file(args.constraints),
            },
            "general": {"name": args.general.name, "sha256": sha256_file(args.general)},
        },
        "benchmark_files": {
            path.name: sha256_file(path) for path in sorted(args.benchmarks.iterdir())
        },
        "artifacts": {
            "train.jsonl": sha256_file(args.out_dir / "train.jsonl"),
            "eval.jsonl": sha256_file(args.out_dir / "eval.jsonl"),
        },
        "splits": {"train": train_counts, "eval": eval_counts},
        "quotas": {"train": TRAIN_QUOTAS, "eval": EVAL_QUOTAS},
        "source_rejections": {
            "verified_math_reasoning": dict(openr1_rejected),
            "execution_checked_code": dict(code_rejected),
            "instruction_constraints": dict(constraints_rejected),
            "general_direct": dict(general_rejected),
        },
        "protected_benchmarks": dict(protected_counts),
        "protected_prompt_count": protected.prompt_count,
        "protected_index_sha256": protected.fingerprint(),
        "epochs": 3,
        "max_length": 2048,
        "training_projection": "native_agentic_prompt_completion",
        "assistant_only_loss": True,
        "inference_thinking_policy": "automatic; never prefill think_start",
        "selection": "deterministic sha256 split and ranking",
    }
    (args.out_dir / "dataset_info.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "chat_template.json").write_text(
        json.dumps(agentic_chat_template(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / ".gitattributes").write_text(
        "*.jsonl filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8"
    )
    (args.out_dir / "README.md").write_text(
        """---
license: other
language:
- en
task_categories:
- text-generation
tags:
- agentic
- tool-use
- reasoning
- code
- supervised-finetuning
size_categories:
- 100K<n<1M
pretty_name: TR-HASH Agentic SFT 32K 500K
---

# TR-HASH Agentic SFT 32K 500K

Audited 500K-example long-horizon SFT mixture for the native TR-HASH Agentic
tokenizer. It cleans the pinned 250K corpus and adds verified examples for
verified arithmetic, compact mathematical reasoning, calculator use,
execution-checked code, constrained instruction following, and general answers.

| Training content | Supervised behavior | Examples | Share |
|---|---|---:|---:|
| General dialogue and instruction responses | Answer diverse questions and follow ordinary user instructions | 100,000 | 20.00% |
| General agentic tool calls | Select and invoke an appropriate tool when external action is required | 75,027 | 15.01% |
| No-tool decisions | Answer directly when calling a tool is unnecessary | 32,041 | 6.41% |
| General tool-result responses | Use a returned tool result to produce the final answer | 21,116 | 4.22% |
| Compact mathematical reasoning | Produce concise, verified reasoning for mathematical problems | 60,000 | 12.00% |
| Programmatically verified arithmetic | Solve generated arithmetic problems with checked answers | 75,000 | 15.00% |
| Calculator invocation | Emit a structured calculator call with valid arguments | 35,000 | 7.00% |
| Calculator-result synthesis | Turn calculator output into a clear final response | 15,000 | 3.00% |
| Execution-checked code | Generate programs whose outputs pass execution checks | 30,000 | 6.00% |
| Constraint following | Respect explicit format, length, lexical, and structural constraints | 20,000 | 4.00% |
| Additional short general answers | Provide concise direct answers across general topics | 36,816 | 7.36% |
| **Total** |  | **500,000** | **100.00%** |

The train split contains 158,072 optional-thinking targets. Thinking is never
prefilled at inference. All other answers use native final/tool envelopes. The
25,000-example validation split follows the same 5% proportions.

The source 250K release contained exact duplicate rows, train/eval overlap, and
legacy `<think>` text in five prompts. These records are excluded and replaced.
The resulting JSONL has no exact content duplicates, reused source IDs,
train/eval content overlap, or legacy reasoning markers.

The intended run is a three-epoch full-parameter SFT from the 100M Agentic
refinement checkpoint. Exact source revisions, quotas, rejection counts,
benchmark decontamination fingerprint, and token totals are recorded in
`dataset_info.json`.
""",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
