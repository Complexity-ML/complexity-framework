#!/usr/bin/env python3
"""Materialize the clean 2M-example TR-HASH Agentic SFT corpus.

The builder keeps general instruction following dominant, embeds tool-use and
no-tool decisions in ordinary conversations, and stores complete native
agentic trajectories as one record. It never replays the 250K or 500K releases.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from datasets import load_dataset
from datasets.download.download_manager import DownloadManager
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

from scripts.prepare_tr_hash_agentic_sft_250k import validate_tokenizer

RECIPE_PATH = Path("configs/tr_hash_agentic_100m_sft_2m.json")
MAX_TOKENS = 2_049
LEGACY_MARKERS = re.compile(r"</?(?:think|final)>|<TOOLCALL>|</TOOLCALL>")
LEADING_LEGACY_THINK = re.compile(r"(?is)^\s*<think>.*?</think>\s*")
NATIVE_MARKERS = re.compile(
    r"<\|(?:system|user|assistant|end_of_turn|think_start|think_end|final_start|final_end|tool_call_start|tool_call_end|tool_result_start|tool_result_end)\|>"
)
IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
ANSWER_TAIL = re.compile(r"(?is)\n(?:#+\s*)?(?:final\s+)?answer\s*:\s*.*$")

SOURCE_FILES = {
    "smol": {
        "repository": "HuggingFaceTB/smol-smoltalk",
        "revision": "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc",
        "patterns": ["data/*.parquet"],
    },
    "luciole": {
        "repository": "OpenLLM-France/Luciole-PostTraining-Dataset-1.1",
        "revision": "95df4f0dfd1abefafd9a7baac795c3e4304a9f99",
        "patterns": [
            "sft_instruct/nemotron-instruction-following-chat-v1/*.jsonl",
            "sft_instruct/nemotron-stem/*.jsonl",
            "sft_instruct/dolci-python-algorithms/*.jsonl",
            "sft_instruct/nemotron-code/*.jsonl",
            "sft_instruct/dolci-instruct-precise-if/*.jsonl",
        ],
    },
    "numina": {
        "repository": "AI-MO/NuminaMath-1.5",
        "revision": "1b05109f9e5c1ad06c0663519502416c30b300f8",
        "patterns": ["data/*.parquet"],
    },
    "bigcode": {
        "repository": "bigcode/self-oss-instruct-sc2-exec-filter-50k",
        "revision": "356bb069eee815daa6e23e9a282eeefe1490ad44",
        "patterns": ["data/*.parquet"],
    },
    "smoltalk": {
        "repository": "HuggingFaceTB/smoltalk",
        "revision": "5feaf2fd3ffca7c237fc38d1861bc30365d48ffa",
        "patterns": [
            "data/everyday-conversations/*.parquet",
            "data/openhermes-100k/*.parquet",
            "data/systemchats-30k/*.parquet",
            "data/smol-magpie-ultra/train-00000-of-00006.parquet",
            "data/smol-magpie-ultra/train-00001-of-00006.parquet",
            "data/smol-magpie-ultra/train-00002-of-00006.parquet",
        ],
    },
    "ultrachat": {
        "repository": "HuggingFaceH4/ultrachat_200k",
        "revision": "8049631c405ae6576f93f445c6b8166f76f5505a",
        "patterns": ["data/train_sft-*.parquet", "data/test_sft-*.parquet"],
    },
}

TOOLS = {
    "calculator": {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate arithmetic.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    "search_knowledge_base": {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the available knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    "date_time": {
        "type": "function",
        "function": {
            "name": "date_time",
            "description": "Get the current date and time.",
            "parameters": {
                "type": "object",
                "properties": {"timezone": {"type": "string"}},
            },
        },
    },
}

PROTECTED_BENCHMARKS = {
    "piqa": {
        "source": "https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip",
        "sha256": "54d32a04f59a7e354396f321723c8d7ec35cc6b08506563d8d1ffcc15ce98ddd",
    },
    "arc_easy": {
        "repository": "allenai/ai2_arc",
        "config": "ARC-Easy",
        "revision": "210d026faf9955653af8916fad021475a3f00453",
        "field": "question",
    },
    "arc_challenge": {
        "repository": "allenai/ai2_arc",
        "config": "ARC-Challenge",
        "revision": "210d026faf9955653af8916fad021475a3f00453",
        "field": "question",
    },
    "hellaswag": {
        "repository": "Rowan/hellaswag",
        "revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
        "field": "ctx",
    },
    "gsm8k": {
        "repository": "openai/gsm8k",
        "config": "main",
        "revision": "740312add88f781978c0658806c59bc2815b9866",
        "field": "question",
    },
}


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def stable_hash(value: Any) -> str:
    text = value if isinstance(value, str) else compact_json(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_hash(value: str) -> str:
    return stable_hash(" ".join(value.casefold().split()))


def load_benchmark_fingerprints() -> dict[str, set[str]]:
    fingerprints: dict[str, set[str]] = {}
    piqa = PROTECTED_BENCHMARKS["piqa"]
    archive = Path(DownloadManager().download(piqa["source"]))
    if sha256_file(archive) != piqa["sha256"]:
        raise RuntimeError("PIQA archive checksum mismatch")
    piqa_hashes: set[str] = set()
    with zipfile.ZipFile(archive) as bundle:
        for member in (
            "physicaliqa-train-dev/train.jsonl",
            "physicaliqa-train-dev/dev.jsonl",
        ):
            for line in bundle.read(member).decode("utf-8").splitlines():
                goal = str(json.loads(line).get("goal") or "").strip()
                if goal:
                    piqa_hashes.add(normalized_hash(goal))
    fingerprints["piqa"] = piqa_hashes
    for name, spec in PROTECTED_BENCHMARKS.items():
        if name == "piqa":
            continue
        args = [spec["repository"]]
        if spec.get("config"):
            args.append(spec["config"])
        dataset = load_dataset(*args, revision=spec["revision"])
        fingerprints[name] = {
            normalized_hash(str(row[spec["field"]]))
            for split in dataset.values()
            for row in split
            if str(row[spec["field"]]).strip()
        }
    return fingerprints


def scaled(count: int, scale: float) -> int:
    return max(1, int(round(count * scale)))


def final_completion(answer: str) -> str:
    return f"<|final_start|>{answer.strip()}<|final_end|><|end_of_turn|>"


def thinking_completion(reasoning: str, answer: str) -> str:
    return (
        f"<|think_start|>{reasoning.strip()}<|think_end|>"
        f"<|final_start|>{answer.strip()}<|final_end|><|end_of_turn|>"
    )


def tool_call_completion(name: str, arguments: dict[str, Any], reasoning: str = "") -> str:
    prefix = f"<|think_start|>{reasoning.strip()}<|think_end|>" if reasoning.strip() else ""
    payload = compact_json({"name": name, "arguments": arguments})
    return f"{prefix}<|tool_call_start|>{payload}<|tool_call_end|><|end_of_turn|>"


def tools_system(names: Iterable[str]) -> str:
    definitions = [TOOLS[name] for name in names]
    return "Available tools:\n" + compact_json(definitions)


def repeated_lines(text: str) -> bool:
    lines = [" ".join(line.casefold().split()) for line in text.splitlines() if line.strip()]
    return len(lines) >= 4 and len(set(lines)) / len(lines) < 0.65


def evaluate_arithmetic(expression: str) -> int | float:
    operators = {
        ast.Add: lambda left, right: left + right,
        ast.Sub: lambda left, right: left - right,
        ast.Mult: lambda left, right: left * right,
        ast.Div: lambda left, right: left / right,
        ast.FloorDiv: lambda left, right: left // right,
        ast.Mod: lambda left, right: left % right,
        ast.Pow: lambda left, right: left**right,
    }

    def visit(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return operators[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        raise ValueError("unsupported calculator expression")

    if len(expression) > 128:
        raise ValueError("calculator expression is too long")
    return visit(ast.parse(expression, mode="eval"))


def validate_agentic_trajectory(trajectory: Any) -> None:
    if not isinstance(trajectory, list) or len(trajectory) < 5:
        raise ValueError("agentic trajectory is incomplete")
    if trajectory[0].get("role") != "system" or trajectory[1].get("role") != "user":
        raise ValueError("agentic trajectory must start with system and user turns")
    system = str(trajectory[0].get("content") or "")
    if not system.startswith("Available tools:\n"):
        raise ValueError("agentic trajectory lacks the compact tool system")
    schemas = json.loads(system.split("\n", 1)[1])
    offered = {item["function"]["name"] for item in schemas}
    if not offered <= set(TOOLS):
        raise ValueError("agentic trajectory offers an unknown tool")
    if trajectory[-1].get("role") != "assistant":
        raise ValueError("agentic trajectory must end with an assistant answer")
    for index, message in enumerate(trajectory):
        role = message.get("role")
        content = str(message.get("content") or "")
        if not content:
            raise ValueError("agentic trajectory contains empty content")
        if role != "assistant" and (
            LEGACY_MARKERS.search(content) or NATIVE_MARKERS.search(content)
        ):
            raise ValueError("non-assistant trajectory turn contains a control marker")
        if role != "assistant":
            continue
        if not content.endswith("<|end_of_turn|>"):
            raise ValueError("assistant trajectory action lacks end_of_turn")
        if content.count("<|think_start|>") != content.count("<|think_end|>"):
            raise ValueError("assistant thinking markers are unbalanced")
        has_call = "<|tool_call_start|>" in content
        has_final = "<|final_start|>" in content
        if has_call == has_final:
            raise ValueError("assistant action must contain exactly one call or final envelope")
        if has_call:
            if content.count("<|tool_call_start|>") != 1 or content.count("<|tool_call_end|>") != 1:
                raise ValueError("tool call markers are unbalanced")
            payload = content.split("<|tool_call_start|>", 1)[1].split("<|tool_call_end|>", 1)[0]
            call = json.loads(payload)
            if call.get("name") not in offered or not isinstance(call.get("arguments"), dict):
                raise ValueError("tool call does not match the offered schema")
            if index + 1 >= len(trajectory) or trajectory[index + 1].get("role") != "tool":
                raise ValueError("tool call is not followed by a tool result")
            if call["name"] == "calculator":
                expression = call["arguments"].get("expression")
                if not isinstance(expression, str):
                    raise ValueError("calculator call lacks a string expression")
                expected = evaluate_arithmetic(expression)
                observed = str(trajectory[index + 1].get("content") or "").strip()
                if observed != str(expected):
                    raise ValueError("calculator result does not match its expression")
        elif index != len(trajectory) - 1:
            raise ValueError("final answer appears before the end of the trajectory")
        elif content.count("<|final_start|>") != 1 or content.count("<|final_end|>") != 1:
            raise ValueError("final markers are unbalanced")


def iter_rows(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    for path in sorted(paths):
        if path.suffix == ".parquet":
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=512):
                yield from batch.to_pylist()
        else:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)


def source_identity(repository: str, subset: str, row: dict[str, Any]) -> str:
    explicit = row.get("id") or row.get("uuid")
    identity = str(explicit) if explicit else stable_hash(row)[:32]
    return f"{repository}:{subset}:{identity}"


def parse_messages(messages: Any) -> tuple[str, str, str] | None:
    if not isinstance(messages, list) or not messages:
        return None
    assistant_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if isinstance(messages[index], dict)
            and str(messages[index].get("role", "")).lower() == "assistant"
        ),
        None,
    )
    if assistant_index is None:
        return None
    answer = str(messages[assistant_index].get("content") or "").strip()
    if (
        not answer
        or LEGACY_MARKERS.search(answer)
        or NATIVE_MARKERS.search(answer)
        or repeated_lines(answer)
    ):
        return None
    pieces: list[str] = []
    last_user = ""
    for message in messages[:assistant_index]:
        if not isinstance(message, dict):
            return None
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        if LEGACY_MARKERS.search(content) or NATIVE_MARKERS.search(content):
            return None
        if role == "system":
            pieces.append(f"<|system|>{content}<|end_of_turn|>")
        elif role == "user":
            pieces.append(f"<|user|>{content}<|end_of_turn|>")
            last_user = content
        elif role == "assistant":
            pieces.append(f"<|assistant|>{final_completion(content)}")
        else:
            return None
    if not pieces or not last_user:
        return None
    return "".join(pieces) + "<|assistant|>", answer, last_user


def extract_python(text: str) -> str | None:
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    code = max(blocks, key=len).strip() if blocks else text.strip()
    try:
        ast.parse(code)
    except SyntaxError:
        return None
    return code


def trajectory_token_count(tokenizer: Tokenizer, trajectory: list[dict[str, str]]) -> int:
    count = 1
    for message in trajectory:
        role, content = message["role"], message["content"]
        if role == "system":
            text = f"<|system|>{content}<|end_of_turn|>"
        elif role == "user":
            text = f"<|user|>{content}<|end_of_turn|>"
        elif role == "tool":
            text = f"<|tool_result_start|>{content}<|tool_result_end|><|end_of_turn|>"
        elif role == "assistant":
            text = f"<|assistant|>{content}"
        else:
            raise ValueError(f"unsupported role: {role}")
        count += len(tokenizer.encode(text).ids)
    return count


class DatasetWriter:
    def __init__(
        self,
        out_dir: Path,
        tokenizer: Tokenizer,
        max_tokens: int,
        benchmark_fingerprints: dict[str, set[str]] | None = None,
        resume: bool = False,
    ):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.paths = {"train": out_dir / "train.jsonl", "validation": out_dir / "eval.jsonl"}
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.seen_records: set[str] = set()
        self.seen_sources: set[str] = set()
        self.counts = {"train": Counter(), "validation": Counter()}
        self.rejected: Counter[str] = Counter()
        self.benchmark_fingerprints = benchmark_fingerprints or {}
        self.benchmark_matches: Counter[str] = Counter()
        if resume:
            for split, path in self.paths.items():
                if not path.exists():
                    raise FileNotFoundError(f"cannot resume without {path}")
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        record = json.loads(line)
                        source_id = str(record["source_conversation_id"])
                        if "trajectory" in record:
                            identity_text = compact_json(record["trajectory"])
                        else:
                            identity_text = record["prompt"] + "\0" + record["completion"]
                        self.seen_sources.add(source_id)
                        self.seen_records.add(normalized_hash(identity_text))
                        partition = str(record["partition"])
                        token_count = int(record["token_count"])
                        self.counts[split]["examples"] += 1
                        self.counts[split]["tokens"] += token_count
                        self.counts[split][partition] += 1
                        self.counts[split][f"tokens:{partition}"] += token_count
        mode = "a" if resume else "w"
        self.files = {name: path.open(mode, encoding="utf-8") for name, path in self.paths.items()}

    def close(self) -> None:
        for handle in self.files.values():
            handle.close()

    def add(self, split: str, record: dict[str, Any], partition: str) -> bool:
        source_id = str(record["source_conversation_id"])
        if source_id in self.seen_sources:
            self.rejected["source_reuse"] += 1
            return False
        benchmark_texts = record.pop("_benchmark_check_texts", [])
        if "trajectory" in record:
            benchmark_texts.extend(
                message["content"] for message in record["trajectory"] if message["role"] == "user"
            )
        benchmark_hashes = {normalized_hash(text) for text in benchmark_texts if text.strip()}
        for benchmark, fingerprints in self.benchmark_fingerprints.items():
            if benchmark_hashes & fingerprints:
                self.rejected["benchmark_contamination"] += 1
                self.benchmark_matches[benchmark] += 1
                return False
        if "trajectory" in record:
            validate_agentic_trajectory(record["trajectory"])
            token_count = trajectory_token_count(self.tokenizer, record["trajectory"])
            identity_text = compact_json(record["trajectory"])
        else:
            token_count = (
                len(self.tokenizer.encode(record["prompt"]).ids)
                + len(self.tokenizer.encode(record["completion"]).ids)
                + 1
            )
            identity_text = record["prompt"] + "\0" + record["completion"]
        digest = normalized_hash(identity_text)
        if digest in self.seen_records:
            self.rejected["normalized_duplicate"] += 1
            return False
        if token_count > self.max_tokens:
            self.rejected["overlength"] += 1
            return False
        record["token_count"] = token_count
        record["partition"] = partition
        self.files[split].write(compact_json(record) + "\n")
        self.seen_records.add(digest)
        self.seen_sources.add(source_id)
        self.counts[split]["examples"] += 1
        self.counts[split]["tokens"] += token_count
        self.counts[split][partition] += 1
        self.counts[split][f"tokens:{partition}"] += token_count
        return True


def public_record(
    row: dict[str, Any],
    *,
    repository: str,
    subset: str,
    category: str,
    tools_available: bool = False,
) -> dict[str, Any] | None:
    parsed = parse_messages(row.get("messages"))
    if parsed is None:
        return None
    prompt, answer, last_user = parsed
    if tools_available:
        digest = stable_hash(prompt)
        choices = (
            ("calculator", "search_knowledge_base"),
            ("search_knowledge_base", "date_time"),
            ("calculator", "date_time"),
        )
        prompt = (
            f"<|system|>{tools_system(choices[int(digest[:2], 16) % 3])}<|end_of_turn|>" + prompt
        )
    source_id = source_identity(repository, subset, row)
    return {
        "prompt": prompt,
        "completion": final_completion(answer),
        "source_dataset": repository,
        "source_subset": subset,
        "source_id": source_id,
        "source_conversation_id": source_id,
        "category": category,
        "thinking_supervised": False,
        "_benchmark_check_texts": [last_user],
    }


def reasoning_record(row: dict[str, Any]) -> dict[str, Any] | None:
    if str(row.get("problem_is_valid", "Yes")) != "Yes":
        return None
    if str(row.get("solution_is_valid", "Yes")) != "Yes":
        return None
    problem = str(row.get("problem") or "").strip()
    solution = str(row.get("solution") or "").strip()
    answer = str(row.get("answer") or "").strip()
    if not problem or not solution or not answer or answer.casefold() == "proof":
        return None
    solution = ANSWER_TAIL.sub("", IMAGE.sub("", solution)).strip()
    solution = re.sub(r"(?is)<think>.*?</think>", "", solution).strip()
    if not 80 <= len(solution) <= 4_000 or len(answer) > 300:
        return None
    if LEGACY_MARKERS.search(problem + solution + answer) or repeated_lines(solution):
        return None
    source_id = source_identity(SOURCE_FILES["numina"]["repository"], "train", row)
    return {
        "prompt": f"<|user|>{problem}<|end_of_turn|><|assistant|>",
        "completion": thinking_completion(solution, answer),
        "source_dataset": SOURCE_FILES["numina"]["repository"],
        "source_subset": "train",
        "source_id": source_id,
        "source_conversation_id": source_id,
        "category": "verified_compact_reasoning",
        "thinking_supervised": True,
        "_benchmark_check_texts": [problem],
    }


def code_record(
    row: dict[str, Any],
    *,
    repository: str,
    subset: str,
) -> dict[str, Any] | None:
    messages = row.get("messages")
    if isinstance(messages, list):
        messages = [dict(message) if isinstance(message, dict) else message for message in messages]
        assistant_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if isinstance(messages[index], dict)
                and str(messages[index].get("role", "")).lower() == "assistant"
            ),
            None,
        )
        if assistant_index is not None:
            content = str(messages[assistant_index].get("content") or "")
            messages[assistant_index]["content"] = LEADING_LEGACY_THINK.sub("", content)
    parsed = parse_messages(messages)
    if parsed is None:
        instruction = str(row.get("instruction") or "").strip()
        answer = LEADING_LEGACY_THINK.sub("", str(row.get("response") or "")).strip()
        if not instruction or not answer:
            return None
        prompt = f"<|user|>{instruction}<|end_of_turn|><|assistant|>"
    else:
        prompt, answer, last_user = parsed
        benchmark_text = last_user
    if parsed is None:
        benchmark_text = instruction
    if extract_python(answer) is None:
        return None
    source_id = source_identity(repository, subset, row)
    return {
        "prompt": prompt,
        "completion": final_completion(answer),
        "source_dataset": repository,
        "source_subset": subset,
        "source_id": source_id,
        "source_conversation_id": source_id,
        "category": "python_code",
        "thinking_supervised": False,
        "_benchmark_check_texts": [benchmark_text],
    }


def generated_reasoning_record(index: int) -> dict[str, Any]:
    rng = random.Random(20_000_000 + index)
    a, b, c = rng.randint(12, 9_999), rng.randint(2, 199), rng.randint(1, 999)
    kind = index % 5
    if kind == 0:
        question = f"A depot receives {b} crates with {a} components each, then ships {c}. How many remain?"
        answer = a * b - c
        reasoning = f"Multiply the crate count by the components per crate, then subtract the shipment: {b}*{a}-{c}={answer}."
    elif kind == 1:
        total = a * b
        question = f"{total} samples are divided equally among {b} teams. How many samples does each team receive?"
        answer = a
        reasoning = f"Equal division gives {total}/{b}={answer} samples per team."
    elif kind == 2:
        question = f"A counter starts at {a}, increases by {b}, and then decreases by {c}. What is its final value?"
        answer = a + b - c
        reasoning = f"Apply both changes in order: {a}+{b}-{c}={answer}."
    elif kind == 3:
        question = f"Find the perimeter of a rectangle measuring {a} cm by {b} cm."
        answer = 2 * (a + b)
        reasoning = f"A rectangle has two sides of each length, so 2*({a}+{b})={answer} cm."
    else:
        percent = rng.choice((10, 20, 25, 40, 50, 75))
        base = a * 100
        answer = base * percent // 100
        question = f"For inventory batch B-{index}, what is {percent}% of {base}?"
        reasoning = f"Multiply by {percent}/100: {base}*{percent}/100={answer}."
    source_id = f"generated-reasoning-2m:{index:08d}"
    return {
        "prompt": f"<|user|>{question}<|end_of_turn|><|assistant|>",
        "completion": thinking_completion(reasoning, str(answer)),
        "source_dataset": "Complexity-ML/verified-reasoning-generator-2m-v1",
        "source_subset": "deterministic",
        "source_id": source_id,
        "source_conversation_id": source_id,
        "category": "verified_compact_reasoning",
        "thinking_supervised": True,
        "_benchmark_check_texts": [question],
    }


def generated_constraint_record(index: int) -> dict[str, Any]:
    rng = random.Random(30_000_000 + index)
    left, right = rng.randint(10, 99_999), rng.randint(10, 99_999)
    mode = index % 4
    if mode == 0:
        prompt = f"Return only valid JSON with keys left, right, and total for {left}+{right}."
        answer = compact_json({"left": left, "right": right, "total": left + right})
    elif mode == 1:
        prompt = f"Answer with exactly three words confirming ticket {left}."
        answer = f"Ticket {left} confirmed"
    elif mode == 2:
        prompt = f"Write the numbers {left} and {right} separated only by a comma."
        answer = f"{left},{right}"
    else:
        prompt = f"Return one Markdown bullet stating that batch {left} contains {right} items."
        answer = f"- Batch {left} contains {right} items."
    source_id = f"generated-constraint-2m:{index:08d}"
    return {
        "prompt": f"<|user|>{prompt}<|end_of_turn|><|assistant|>",
        "completion": final_completion(answer),
        "source_dataset": "Complexity-ML/verified-constraints-generator-2m-v1",
        "source_subset": "deterministic",
        "source_id": source_id,
        "source_conversation_id": source_id,
        "category": "constraints_and_formats",
        "thinking_supervised": False,
        "_benchmark_check_texts": [prompt],
    }


def generated_code_record(index: int) -> dict[str, Any]:
    rng = random.Random(35_000_000 + index)
    value = rng.randint(2, 10_000)
    mode = index % 5
    if mode == 0:
        name = f"scale_{index}"
        factor = rng.randint(2, 20)
        prompt = f"Write a Python function named {name} that multiplies an integer by {factor}."
        code = f"def {name}(value: int) -> int:\n    return value * {factor}"
    elif mode == 1:
        name = f"is_multiple_{index}"
        divisor = rng.randint(2, 19)
        prompt = (
            f"Write a Python function named {name} that returns whether an integer "
            f"is divisible by {divisor}."
        )
        code = f"def {name}(value: int) -> bool:\n    return value % {divisor} == 0"
    elif mode == 2:
        name = f"offset_values_{index}"
        offset = rng.randint(-50, 50)
        prompt = (
            f"Write a Python function named {name} that adds {offset} to every integer in a list."
        )
        code = (
            f"def {name}(values: list[int]) -> list[int]:\n"
            f"    return [value + ({offset}) for value in values]"
        )
    elif mode == 3:
        name = f"count_long_{index}"
        minimum = rng.randint(2, 12)
        prompt = (
            f"Write a Python function named {name} that counts strings whose length "
            f"is at least {minimum}."
        )
        code = (
            f"def {name}(values: list[str]) -> int:\n"
            f"    return sum(len(value) >= {minimum} for value in values)"
        )
    else:
        name = f"bounded_{index}"
        upper = value + rng.randint(1, 500)
        prompt = (
            f"Write a Python function named {name} that clamps an integer to the "
            f"inclusive range [{value}, {upper}]."
        )
        code = f"def {name}(number: int) -> int:\n    return max({value}, min(number, {upper}))"
    if extract_python(code) is None:
        raise AssertionError("deterministic Python generator produced invalid syntax")
    source_id = f"generated-python-2m:{index:08d}"
    return {
        "prompt": f"<|user|>{prompt}<|end_of_turn|><|assistant|>",
        "completion": final_completion(f"```python\n{code}\n```"),
        "source_dataset": "Complexity-ML/verified-python-generator-2m-v1",
        "source_subset": "deterministic",
        "source_id": source_id,
        "source_conversation_id": source_id,
        "category": "python_code",
        "thinking_supervised": False,
        "_benchmark_check_texts": [prompt],
    }


def agentic_record(index: int, subtype: str) -> dict[str, Any]:
    rng = random.Random(40_000_000 + index)
    a, b, c = rng.randint(12, 999), rng.randint(2, 99), rng.randint(1, 500)
    source_id = f"generated-agentic-2m:{subtype}:{index:08d}"
    if subtype == "single_tool_end_to_end":
        mode = index % 3
        if mode == 0:
            result = a * b - c
            expression = f"{a}*{b}-{c}"
            user = (
                f"For request CALC-{index}, a workshop packs {a} boxes of {b} parts, "
                f"then removes {c} parts. How many remain?"
            )
            call = tool_call_completion("calculator", {"expression": expression})
            tool_result, final = str(result), str(result)
            names = ["calculator", "search_knowledge_base"]
        elif mode == 1:
            project = f"Project-{index}-{a}-{b}"
            user = f"What is the verified owner of {project}?"
            call = tool_call_completion("search_knowledge_base", {"query": user})
            tool_result = compact_json({"project": project, "owner": f"Team {c}"})
            final = f"The verified owner of {project} is Team {c}."
            names = ["search_knowledge_base", "date_time"]
        else:
            timezone = "Europe/Paris" if index % 2 else "UTC"
            user = f"For request DT-{index}, what is the current date and time in {timezone}?"
            call = tool_call_completion("date_time", {"timezone": timezone})
            tool_result = f"2031-06-{index % 28 + 1:02d} 14:{index % 60:02d} {timezone}"
            final = tool_result
            names = ["date_time", "calculator"]
        trajectory = [
            {"role": "system", "content": tools_system(names)},
            {"role": "user", "content": user},
            {"role": "assistant", "content": call},
            {"role": "tool", "content": tool_result},
            {"role": "assistant", "content": final_completion(final)},
        ]
    elif subtype == "multi_tool_end_to_end":
        unit_price, quantity = a, b
        user = f"Find the unit price and quantity for order O-{index}, then calculate its total."
        first = tool_call_completion(
            "search_knowledge_base", {"query": f"order O-{index} unit price and quantity"}
        )
        found = compact_json({"unit_price": unit_price, "quantity": quantity})
        expression = f"{unit_price}*{quantity}"
        second = tool_call_completion("calculator", {"expression": expression})
        total = unit_price * quantity
        trajectory = [
            {
                "role": "system",
                "content": tools_system(["search_knowledge_base", "calculator"]),
            },
            {"role": "user", "content": user},
            {"role": "assistant", "content": first},
            {"role": "tool", "content": found},
            {"role": "assistant", "content": second},
            {"role": "tool", "content": str(total)},
            {"role": "assistant", "content": final_completion(f"Order O-{index} totals {total}.")},
        ]
    elif subtype == "tool_failure_and_recovery":
        broad = f"asset {index}-{a}"
        precise = f"asset {index}-{a} in register R-{b}"
        trajectory = [
            {"role": "system", "content": tools_system(["search_knowledge_base"])},
            {"role": "user", "content": f"Who owns {precise}?"},
            {
                "role": "assistant",
                "content": tool_call_completion("search_knowledge_base", {"query": broad}),
            },
            {"role": "tool", "content": "No unique match; provide the register identifier."},
            {
                "role": "assistant",
                "content": tool_call_completion("search_knowledge_base", {"query": precise}),
            },
            {"role": "tool", "content": compact_json({"owner": f"Operator-{c}"})},
            {
                "role": "assistant",
                "content": final_completion(f"{precise} is owned by Operator-{c}."),
            },
        ]
    else:
        expression = f"({a}+{b})*{c}"
        result = (a + b) * c
        reasoning = "The request requires adding the two quantities before multiplication."
        trajectory = [
            {"role": "system", "content": tools_system(["calculator"])},
            {
                "role": "user",
                "content": (
                    f"For request CALC-{index}, combine batches of {a} and {b} items, "
                    f"then make {c} identical sets. How many items are required?"
                ),
            },
            {
                "role": "assistant",
                "content": tool_call_completion(
                    "calculator", {"expression": expression}, reasoning=reasoning
                ),
            },
            {"role": "tool", "content": str(result)},
            {"role": "assistant", "content": final_completion(str(result))},
        ]
    return {
        "trajectory": trajectory,
        "source_dataset": "Complexity-ML/verified-agentic-trajectory-generator-2m-v1",
        "source_subset": subtype,
        "source_id": source_id,
        "source_conversation_id": source_id,
        "category": "complete_agentic_trajectories",
        "trajectory_subtype": subtype,
        "thinking_supervised": subtype == "reasoning_code_and_tools",
    }


def collect_public(
    writer: DatasetWriter,
    paths: list[Path],
    *,
    repository: str,
    subset: str,
    category: str,
    train_quota: int,
    validation_quota: int,
    tools_available: bool = False,
    projector=public_record,
) -> None:
    accepted = Counter(
        {
            "train": writer.counts["train"][category],
            "validation": writer.counts["validation"][category],
        }
    )
    for row in iter_rows(paths):
        identity = source_identity(repository, subset, row)
        split = "validation" if int(stable_hash(identity)[:8], 16) % 20 == 0 else "train"
        quota = validation_quota if split == "validation" else train_quota
        if accepted[split] >= quota:
            if accepted["train"] >= train_quota and accepted["validation"] >= validation_quota:
                break
            continue
        if projector is reasoning_record:
            record = reasoning_record(row)
        elif projector is code_record:
            record = code_record(row, repository=repository, subset=subset)
        else:
            record = public_record(
                row,
                repository=repository,
                subset=subset,
                category=category,
                tools_available=tools_available,
            )
        if record is None:
            writer.rejected[f"{category}:projection"] += 1
            continue
        if writer.add(split, record, category):
            accepted[split] += 1
            if accepted[split] % 10_000 == 0:
                print(f"{category}/{subset}/{split}: {accepted[split]:,}/{quota:,}", flush=True)
    if accepted["train"] != train_quota or accepted["validation"] != validation_quota:
        raise RuntimeError(
            f"{category}/{subset} capacity shortfall: {dict(accepted)}, "
            f"required train={train_quota}, validation={validation_quota}"
        )


def collect_generated(
    writer: DatasetWriter,
    generator,
    *,
    partition: str,
    train_quota: int,
    validation_quota: int,
    offset: int,
) -> None:
    for split, quota, start in (
        ("validation", validation_quota, offset),
        ("train", train_quota, offset + validation_quota),
    ):
        accepted = 0
        index = start
        while accepted < quota:
            record = generator(index)
            index += 1
            accepted += int(writer.add(split, record, partition))


def collect_public_pool(
    writer: DatasetWriter,
    sources: list[tuple[list[Path], str, str]],
    *,
    category: str,
    train_quota: int,
    validation_quota: int,
    tools_available: bool = False,
    projector=public_record,
) -> None:
    accepted = Counter(
        {
            "train": writer.counts["train"][category],
            "validation": writer.counts["validation"][category],
        }
    )
    for paths, repository, subset in sources:
        for row in iter_rows(paths):
            identity = source_identity(repository, subset, row)
            split = "validation" if int(stable_hash(identity)[:8], 16) % 20 == 0 else "train"
            quota = validation_quota if split == "validation" else train_quota
            if accepted[split] >= quota:
                if accepted["train"] >= train_quota and accepted["validation"] >= validation_quota:
                    return
                continue
            if projector is code_record:
                record = code_record(row, repository=repository, subset=subset)
            else:
                record = public_record(
                    row,
                    repository=repository,
                    subset=subset,
                    category=category,
                    tools_available=tools_available,
                )
            if record is None:
                writer.rejected[f"{category}:projection"] += 1
                continue
            if writer.add(split, record, category):
                accepted[split] += 1
                if accepted[split] % 10_000 == 0:
                    print(f"{category}/{split}: {accepted[split]:,}/{quota:,}", flush=True)
    if accepted["train"] != train_quota or accepted["validation"] != validation_quota:
        raise RuntimeError(
            f"{category} pooled capacity shortfall: {dict(accepted)}, "
            f"required train={train_quota}, validation={validation_quota}"
        )


def resolve_sources(*, local_files_only: bool = False) -> dict[str, Path]:
    result = {}
    for name, source in SOURCE_FILES.items():
        result[name] = Path(
            snapshot_download(
                source["repository"],
                repo_type="dataset",
                revision=source["revision"],
                allow_patterns=source["patterns"],
                local_files_only=local_files_only,
            )
        )
    return result


def write_source_cache_manifest(out_dir: Path, sources: dict[str, Path]) -> None:
    payload = {
        "format": "tr-hash-source-cache-v1",
        "sources": {
            name: {
                "repository": SOURCE_FILES[name]["repository"],
                "revision": SOURCE_FILES[name]["revision"],
                "local_snapshot": str(path),
                "files": [
                    str(file.relative_to(path))
                    for file in sorted(path.rglob("*"))
                    if file.is_file()
                ],
            }
            for name, path in sources.items()
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "source_cache.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def source_paths(root: Path, pattern: str) -> list[Path]:
    paths = sorted(root.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no source files match {root / pattern}")
    return paths


def write_manifest(
    out_dir: Path,
    writer: DatasetWriter,
    recipe: dict[str, Any],
    scale: float,
) -> dict[str, Any]:
    partitions = recipe["token_distribution_policy"]["partitions"]
    train_tokens = writer.counts["train"]["tokens"]
    token_audit = {}
    for name, policy in partitions.items():
        tokens = sum(writer.counts["train"][f"tokens:{item}"] for item in policy["components"])
        share = tokens / max(1, train_tokens)
        token_audit[name] = {
            "tokens": tokens,
            "share": share,
            "pilot_observed_share": policy["pilot_observed_share"],
        }
    expected: dict[str, dict[str, int]] = {"train": {}, "validation": {}}
    for component in recipe["components"]:
        name = component["name"]
        if scale != 1.0:
            expected["train"][name] = writer.counts["train"][name]
            expected["validation"][name] = writer.counts["validation"][name]
        elif name == "complete_agentic_trajectories":
            for split, field in (
                ("train", "train_examples"),
                ("validation", "validation_examples"),
            ):
                expected[split][name] = sum(
                    scaled(item[field], scale) for item in component["submix"].values()
                )
        else:
            expected["train"][name] = scaled(component["train_examples"], scale)
            expected["validation"][name] = scaled(component["validation_examples"], scale)
    count_audit = {
        split: {
            name: {
                "expected": count,
                "observed": writer.counts[split][name],
                "passed": writer.counts[split][name] == count,
            }
            for name, count in split_expected.items()
        }
        for split, split_expected in expected.items()
    }
    counts_passed = all(item["passed"] for split in count_audit.values() for item in split.values())
    manifest = {
        "format": "tr-hash-agentic-sft-2m-v1",
        "quality_status": "passed" if counts_passed else "failed",
        "scale": scale,
        "recipe": recipe,
        "splits": {name: dict(counts) for name, counts in writer.counts.items()},
        "example_count_audit": count_audit,
        "token_distribution_audit": token_audit,
        "benchmark_contamination_audit": {
            "method": "exact_normalized_user_prompt_fingerprint",
            "benchmarks": {
                name: {
                    "fingerprints": len(fingerprints),
                    "rejected_matches": writer.benchmark_matches[name],
                }
                for name, fingerprints in writer.benchmark_fingerprints.items()
            },
            "accepted_matches": 0,
            "passed": True,
        },
        "rejected": dict(writer.rejected),
        "artifacts": {
            name: {"path": path.name, "sha256": sha256_file(path)}
            for name, path in writer.paths.items()
        },
    }
    (out_dir / "dataset_info.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if manifest["quality_status"] != "passed":
        raise RuntimeError(f"example count audit failed: {count_audit}")
    return manifest


def write_release_files(out_dir: Path, manifest: dict[str, Any], tokenizer: Tokenizer) -> None:
    recipe = manifest["recipe"]
    template = {
        "id": "tr-hash-agentic-chat-v1",
        "version": 1,
        "system_prompt": "",
        "system_format": "<|system|>{content}<|end_of_turn|>",
        "user_format": "<|user|>{content}<|end_of_turn|>",
        "assistant_prefix": "<|assistant|>",
        "turn_separator": "",
        "eos_token": tokenizer.id_to_token(1),
        "end_of_turn_token": "<|end_of_turn|>",
        "assistant_only_loss": True,
        "training_projection": "native_agentic_full_trajectory",
        "assistant_envelope": {
            "type": "optional_think_final",
            "think_start": "<|think_start|>",
            "think_end": "<|think_end|>",
            "final_start": "<|final_start|>",
            "final_end": "<|final_end|>",
            "scope": "reasoning_tasks",
        },
    }
    (out_dir / "chat_template.json").write_text(
        json.dumps(template, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / ".gitattributes").write_text(
        "*.jsonl filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )
    train = manifest["splits"]["train"]
    validation = manifest["splits"]["validation"]
    rows = []
    for component in recipe["components"]:
        name = component["name"]
        rows.append(f"| `{name}` | {train[name]:,} | {validation[name]:,} |")
    shares = manifest["token_distribution_audit"]
    benchmark = manifest["benchmark_contamination_audit"]
    readme = f"""---
pretty_name: TR-HASH Agentic SFT 2M
language:
- en
- fr
license: other
task_categories:
- text-generation
size_categories:
- 1M<n<10M
---

# TR-HASH Agentic SFT 2M

Clean full-parameter SFT corpus for the TR-HASH 100M Agentic line. This build contains {train["examples"]:,} training examples and {validation["examples"]:,} validation examples. It does not replay the previous 250K or 500K SFT releases.

## Composition

| Component | Train | Validation |
|---|---:|---:|
{chr(10).join(rows)}

The tool examples remain part of general conversations. A complete trajectory stores the user request, each assistant tool call, each tool result, and the grounded final answer in one record. Tools may also be offered when the correct behavior is to answer directly.

## Native format

- Tokenizer: `{recipe["tokenizer"]["repository"]}` at `{recipe["tokenizer"]["revision"]}`
- Starting checkpoint: `{recipe["starting_checkpoint"]["repository"]}` at commit `{recipe["starting_checkpoint"]["revision"]}`, subfolder `{recipe["starting_checkpoint"]["subfolder"]}`
- Context limit: {recipe["sequence_length"]} tokens with overlength examples rejected rather than truncated
- Assistant-only loss with every assistant action in complete trajectories supervised
- Optional native thinking envelope; thinking is never prefilled or forced by this dataset

## Measured token distribution

The signed example counts are authoritative. Token shares are reported after tokenization and are not padded with verbose traces:

- General, reasoning, code, and constraints: {shares["general_reasoning_code_and_constraints"]["share"]:.2%}
- Complete agentic trajectories: {shares["complete_agentic_trajectories"]["share"]:.2%}
- Tools offered without a call: {shares["tools_available_without_call"]["share"]:.2%}

## Quality gates

- Exact and normalized deduplication across train and validation
- Conversation-level split isolation
- Native marker and tool-call validation
- Calculator expression/result verification
- Python syntax parsing, with upstream execution-filter provenance retained when available
- No token truncation
- Exact normalized user-prompt fingerprints against PIQA, ARC-Easy, ARC-Challenge, HellaSwag, and GSM8K; {sum(item["rejected_matches"] for item in benchmark["benchmarks"].values()):,} matching candidates were rejected and zero were retained

`dataset_info.json` records the pinned sources, revisions, checksums, rejection counters, component counts, token measurements, and contamination audit. `chat_template.json` contains the native Agentic training contract.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, default=RECIPE_PATH)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Resolve every pinned source from the local Hugging Face cache only.",
    )
    args = parser.parse_args()
    if not 0 < args.scale <= 1:
        raise ValueError("scale must be in (0, 1]")
    tokenizer_file = (
        args.tokenizer / "tokenizer.json" if args.tokenizer.is_dir() else args.tokenizer
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    validate_tokenizer(tokenizer)
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    sources = resolve_sources(local_files_only=args.offline)
    write_source_cache_manifest(args.out_dir, sources)
    benchmark_fingerprints = load_benchmark_fingerprints()
    if set(benchmark_fingerprints) != set(recipe["protected_benchmarks"]):
        raise RuntimeError("protected benchmark fingerprint set does not match recipe")
    writer = DatasetWriter(
        args.out_dir,
        tokenizer,
        MAX_TOKENS,
        benchmark_fingerprints=benchmark_fingerprints,
        resume=args.resume,
    )
    try:
        collect_public_pool(
            writer,
            [
                (
                    source_paths(sources["smol"], "data/train-*.parquet"),
                    SOURCE_FILES["smol"]["repository"],
                    "train",
                ),
                (
                    source_paths(
                        sources["luciole"],
                        "sft_instruct/nemotron-instruction-following-chat-v1/*.jsonl",
                    ),
                    SOURCE_FILES["luciole"]["repository"],
                    "nemotron_instruction_following_chat_v1",
                ),
                (
                    source_paths(sources["luciole"], "sft_instruct/nemotron-stem/*.jsonl"),
                    SOURCE_FILES["luciole"]["repository"],
                    "nemotron_stem",
                ),
                (
                    source_paths(sources["ultrachat"], "data/*_sft-*.parquet"),
                    SOURCE_FILES["ultrachat"]["repository"],
                    "sft",
                ),
                (
                    source_paths(sources["smoltalk"], "data/openhermes-100k/*.parquet"),
                    SOURCE_FILES["smoltalk"]["repository"],
                    "openhermes_100k",
                ),
                (
                    source_paths(sources["smoltalk"], "data/systemchats-30k/*.parquet"),
                    SOURCE_FILES["smoltalk"]["repository"],
                    "systemchats_30k",
                ),
                (
                    source_paths(sources["smoltalk"], "data/everyday-conversations/*.parquet"),
                    SOURCE_FILES["smoltalk"]["repository"],
                    "everyday_conversations",
                ),
                (
                    source_paths(sources["smoltalk"], "data/smol-magpie-ultra/*.parquet"),
                    SOURCE_FILES["smoltalk"]["repository"],
                    "smol_magpie_ultra",
                ),
            ],
            category="general_conversations",
            train_quota=scaled(800_000, args.scale),
            validation_quota=scaled(40_000, args.scale),
        )

        collect_public(
            writer,
            source_paths(sources["numina"], "data/train-*.parquet"),
            repository=SOURCE_FILES["numina"]["repository"],
            subset="train",
            category="verified_compact_reasoning",
            train_quota=scaled(180_000, args.scale),
            validation_quota=scaled(9_000, args.scale),
            projector=reasoning_record,
        )
        collect_generated(
            writer,
            generated_reasoning_record,
            partition="verified_compact_reasoning",
            train_quota=scaled(270_000, args.scale),
            validation_quota=scaled(13_500, args.scale),
            offset=0,
        )

        collect_public_pool(
            writer,
            [
                (
                    source_paths(
                        sources["luciole"],
                        "sft_instruct/dolci-python-algorithms/*.jsonl",
                    ),
                    SOURCE_FILES["luciole"]["repository"],
                    "dolci_python_algorithms",
                ),
                (
                    source_paths(sources["luciole"], "sft_instruct/nemotron-code/*.jsonl"),
                    SOURCE_FILES["luciole"]["repository"],
                    "nemotron_code",
                ),
                (
                    source_paths(sources["bigcode"], "data/*.parquet"),
                    SOURCE_FILES["bigcode"]["repository"],
                    "train",
                ),
            ],
            category="python_code",
            train_quota=scaled(300_000, args.scale),
            validation_quota=scaled(15_000, args.scale),
            projector=code_record,
        )
        collect_generated(
            writer,
            generated_code_record,
            partition="python_code",
            train_quota=scaled(50_000, args.scale),
            validation_quota=scaled(2_500, args.scale),
            offset=0,
        )

        collect_public(
            writer,
            source_paths(sources["luciole"], "sft_instruct/dolci-instruct-precise-if/*.jsonl"),
            repository=SOURCE_FILES["luciole"]["repository"],
            subset="dolci_instruct_precise_if",
            category="constraints_and_formats",
            train_quota=scaled(100_000, args.scale),
            validation_quota=scaled(5_000, args.scale),
        )
        collect_generated(
            writer,
            generated_constraint_record,
            partition="constraints_and_formats",
            train_quota=scaled(50_000, args.scale),
            validation_quota=scaled(2_500, args.scale),
            offset=0,
        )

        collect_public_pool(
            writer,
            [
                (
                    source_paths(sources["smol"], "data/train-*.parquet"),
                    SOURCE_FILES["smol"]["repository"],
                    "train",
                ),
                (
                    source_paths(sources["smol"], "data/test-*.parquet"),
                    SOURCE_FILES["smol"]["repository"],
                    "test",
                ),
                (
                    source_paths(
                        sources["luciole"],
                        "sft_instruct/nemotron-instruction-following-chat-v1/*.jsonl",
                    ),
                    SOURCE_FILES["luciole"]["repository"],
                    "nemotron_instruction_following_chat_v1",
                ),
                (
                    source_paths(sources["luciole"], "sft_instruct/nemotron-stem/*.jsonl"),
                    SOURCE_FILES["luciole"]["repository"],
                    "nemotron_stem",
                ),
                (
                    source_paths(sources["ultrachat"], "data/*_sft-*.parquet"),
                    SOURCE_FILES["ultrachat"]["repository"],
                    "sft",
                ),
                (
                    source_paths(sources["smoltalk"], "data/openhermes-100k/*.parquet"),
                    SOURCE_FILES["smoltalk"]["repository"],
                    "openhermes_100k",
                ),
                (
                    source_paths(sources["smoltalk"], "data/systemchats-30k/*.parquet"),
                    SOURCE_FILES["smoltalk"]["repository"],
                    "systemchats_30k",
                ),
                (
                    source_paths(sources["smoltalk"], "data/everyday-conversations/*.parquet"),
                    SOURCE_FILES["smoltalk"]["repository"],
                    "everyday_conversations",
                ),
                (
                    source_paths(sources["smoltalk"], "data/smol-magpie-ultra/*.parquet"),
                    SOURCE_FILES["smoltalk"]["repository"],
                    "smol_magpie_ultra",
                ),
            ],
            category="tool_available_direct_answer",
            train_quota=scaled(150_000, args.scale),
            validation_quota=scaled(7_500, args.scale),
            tools_available=True,
        )

        submix = recipe["components"][-1]["submix"]
        offset = 0
        for subtype, counts in submix.items():
            collect_generated(
                writer,
                lambda index, subtype=subtype: agentic_record(index, subtype),
                partition="complete_agentic_trajectories",
                train_quota=scaled(counts["train_examples"], args.scale),
                validation_quota=scaled(counts["validation_examples"], args.scale),
                offset=offset,
            )
            offset += counts["train_examples"] + counts["validation_examples"]
    finally:
        writer.close()

    manifest = write_manifest(args.out_dir, writer, recipe, args.scale)
    write_release_files(args.out_dir, manifest, tokenizer)
    print(json.dumps(manifest["token_distribution_audit"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
