#!/usr/bin/env python3
"""Recompile an audited text SFT release with the TR-HASH 32,004 protocol.

This script consumes JSONL text, never an existing token shard.  It preserves
the selected examples and provenance, removes protected benchmark overlaps,
wraps every assistant turn with the four canonical special tokens, and drops
only rows that no longer fit after the four-token envelope is added.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import unicodedata
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset

from complexity.inference.chat_template import (
    align_chat_template_eos,
    reasoning_chat_template_32004,
)
from complexity.tokenizer import Tokenizer

try:
    from scripts.sft_500m_32k_tr import format_record
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from sft_500m_32k_tr import format_record


THINK_START = "<|think_start|>"
THINK_END = "<|think_end|>"
FINAL_START = "<|final_start|>"
FINAL_END = "<|final_end|>"
SPECIAL_TOKEN_IDS = {
    THINK_START: 32_000,
    THINK_END: 32_001,
    FINAL_START: 32_002,
    FINAL_END: 32_003,
}
FORMAT_ID = "tr-hash-think-final-32004-v1"
_ALL_MARKERS = tuple(SPECIAL_TOKEN_IDS)
_LEGACY_TAG = re.compile(r"</?(?:think|analysis|reasoning|final|answer)\s*>", re.IGNORECASE)
_EXPLICIT = re.compile(
    r"<(?:think|analysis|reasoning)\s*>(?P<think>.*?)"
    r"</(?:think|analysis|reasoning)\s*>\s*"
    r"<(?:final|answer)\s*>(?P<final>.*?)"
    r"</(?:final|answer)\s*>",
    re.IGNORECASE | re.DOTALL,
)
_FINAL_LINE = re.compile(
    r"(?:^|\n)\s*(?:final\s+answer|answer)\s*:\s*(?P<answer>[^\n]+)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AssistantEnvelope:
    reasoning: str
    final: str
    extraction: str


@dataclass(frozen=True)
class BenchmarkGuard:
    exact: frozenset[str]
    sixteen_token_prefixes: frozenset[tuple[str, ...]]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_for_dedup(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+", text, flags=re.UNICODE))


def make_benchmark_guard(prompts: set[str]) -> BenchmarkGuard:
    prefixes = {tuple(prompt.split()[:16]) for prompt in prompts if len(prompt.split()) >= 16}
    return BenchmarkGuard(frozenset(prompts), frozenset(prefixes))


def _benchmark_text(row: dict[str, Any], field: str) -> str:
    return str(row.get(field, "") or "").strip()


def load_protected_benchmark_prompts(recipe: dict[str, Any]) -> BenchmarkGuard:
    protected: set[str] = set()
    for benchmark in recipe.get("protected_benchmarks", []):
        if "archive_url" in benchmark:
            with urllib.request.urlopen(benchmark["archive_url"], timeout=120) as response:
                payload = response.read()
            actual_sha256 = hashlib.sha256(payload).hexdigest()
            if actual_sha256 != benchmark["archive_sha256"]:
                raise RuntimeError(f"{benchmark['name']} archive hash mismatch: {actual_sha256}")
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                for member in benchmark["members"]:
                    with archive.open(member) as handle:
                        for raw_line in handle:
                            row = json.loads(raw_line.decode("utf-8"))
                            text = normalize_for_dedup(
                                _benchmark_text(row, benchmark["text_field"])
                            )
                            if text:
                                protected.add(text)
            continue
        for split in benchmark["splits"]:
            dataset = load_dataset(
                benchmark["dataset"],
                benchmark.get("config"),
                split=split,
                revision=benchmark["revision"],
                streaming=True,
            )
            for row in dataset:
                text = normalize_for_dedup(_benchmark_text(row, benchmark["text_field"]))
                if text:
                    protected.add(text)
    return make_benchmark_guard(protected)


def benchmark_overlap(
    messages: list[dict[str, str]], protected_prompts: BenchmarkGuard | set[str]
) -> bool:
    guard = (
        protected_prompts
        if isinstance(protected_prompts, BenchmarkGuard)
        else make_benchmark_guard(protected_prompts)
    )
    user_text = normalize_for_dedup(
        "\n".join(message["content"] for message in messages if message["role"] == "user")
    )
    if user_text in guard.exact:
        return True
    words = user_text.split()
    windows = {tuple(words[index : index + 16]) for index in range(len(words) - 15)}
    return not windows.isdisjoint(guard.sixteen_token_prefixes)


def _encoded_length(
    messages: list[dict[str, str]],
    tokenizer: Tokenizer,
    chat_template: dict[str, Any],
) -> int:
    prompt, completion = format_record({"messages": messages}, chat_template)
    return (
        len(tokenizer.encode(prompt, add_special_tokens=False))
        + len(tokenizer.encode(completion, add_special_tokens=False))
        + 1
    )


def _drop_oldest_complete_turn(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    systems = [message for message in messages if message["role"] == "system"]
    dialogue = [message for message in messages if message["role"] != "system"]
    if len(dialogue) <= 2:
        return messages
    next_user = next(
        (index for index in range(1, len(dialogue)) if dialogue[index]["role"] == "user"),
        None,
    )
    if next_user is None:
        return messages
    return systems + dialogue[next_user:]


def fit_complete_turns(
    messages: list[dict[str, str]],
    tokenizer: Tokenizer,
    sequence_length: int,
    chat_template: dict[str, Any],
) -> list[dict[str, str]] | None:
    candidate = [dict(message) for message in messages]
    while _encoded_length(candidate, tokenizer, chat_template) > sequence_length:
        reduced = _drop_oldest_complete_turn(candidate)
        if reduced == candidate:
            return None
        candidate = reduced
    return candidate


def validate_tokenizer_32004(tokenizer: Tokenizer) -> None:
    if len(tokenizer) != 32_004:
        raise ValueError(f"expected canonical 32,004 tokenizer, got {len(tokenizer):,}")
    for token, expected_id in SPECIAL_TOKEN_IDS.items():
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if encoded != [expected_id]:
            raise ValueError(f"{token} encoded as {encoded}, expected [{expected_id}]")


def _strip_markers(text: str) -> str:
    for marker in _ALL_MARKERS:
        text = text.replace(marker, "")
    return _LEGACY_TAG.sub("", text).strip()


def _last_boxed(text: str) -> str | None:
    """Return the last balanced ``\\boxed{...}`` expression, if present."""

    positions = [match.start() for match in re.finditer(r"\\boxed\s*\{", text)]
    for start in reversed(positions):
        brace = text.find("{", start)
        depth = 0
        for index in range(brace, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1].strip()
    return None


def _extract_explicit(text: str) -> AssistantEnvelope | None:
    match = _EXPLICIT.search(text)
    if match is None:
        return None
    reasoning = _strip_markers(match.group("think"))
    final = _strip_markers(match.group("final"))
    if not final:
        return None
    return AssistantEnvelope(reasoning, final, "source_explicit")


def split_assistant_content(content: str, capability: str) -> AssistantEnvelope:
    """Split only when the source provides a trustworthy final-answer signal."""

    content = str(content).strip()
    explicit = _extract_explicit(content)
    if explicit is not None:
        return explicit
    clean = _strip_markers(content)
    capability_lower = capability.casefold()
    reasoning_capability = "math" in capability_lower or "reasoning" in capability_lower
    if reasoning_capability:
        boxed = _last_boxed(clean)
        if boxed:
            return AssistantEnvelope(clean, boxed, "boxed_math")
        final_line = _FINAL_LINE.search(clean)
        if final_line:
            answer = final_line.group("answer").strip()
            reasoning = clean[: final_line.start()].strip()
            if answer:
                return AssistantEnvelope(reasoning, answer, "labeled_final")
    # A plain response is not silently reinterpreted as hidden reasoning.
    return AssistantEnvelope("", clean, "direct_final")


def render_envelope(envelope: AssistantEnvelope) -> str:
    return f"{THINK_START}{envelope.reasoning}{THINK_END}{FINAL_START}{envelope.final}{FINAL_END}"


def wrap_messages(
    messages: list[dict[str, Any]], capability: str
) -> tuple[list[dict[str, str]], Counter[str]]:
    wrapped: list[dict[str, str]] = []
    extractions: Counter[str] = Counter()
    for message in messages:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if role == "assistant":
            envelope = split_assistant_content(content, capability)
            if not envelope.final:
                raise ValueError("assistant final span is empty")
            content = render_envelope(envelope)
            extractions[envelope.extraction] += 1
        wrapped.append({"role": role, "content": content})
    return wrapped, extractions


def validate_enveloped_messages(messages: list[dict[str, str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for message in messages:
        content = message["content"]
        if message["role"] != "assistant":
            if any(marker in content for marker in _ALL_MARKERS):
                raise ValueError("reasoning marker found outside an assistant turn")
            continue
        for marker in _ALL_MARKERS:
            count = content.count(marker)
            if count != 1:
                raise ValueError(f"assistant turn has {count} occurrences of {marker}")
            counts[marker] += 1
        positions = [content.index(marker) for marker in _ALL_MARKERS]
        if positions != sorted(positions):
            raise ValueError("assistant reasoning markers are not properly ordered")
        if not content.split(FINAL_START, 1)[1].split(FINAL_END, 1)[0].strip():
            raise ValueError("assistant final span is empty")
    return counts


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def recompile_partition(
    source: Path,
    target: Path,
    *,
    tokenizer: Tokenizer,
    sequence_length: int,
    benchmark_guard: BenchmarkGuard,
) -> dict[str, Any]:
    eos_text = tokenizer.decode([tokenizer.eos_token_id], skip_special_tokens=False)
    chat_template = align_chat_template_eos(reasoning_chat_template_32004(), eos_token=eos_text)
    temporary = target.with_suffix(target.suffix + ".partial")
    source_counts: Counter[str] = Counter()
    capability_counts: Counter[str] = Counter()
    extraction_counts: Counter[str] = Counter()
    marker_counts: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    input_examples = output_examples = 0
    with temporary.open("w", encoding="utf-8") as output:
        for _, row in _iter_jsonl(source):
            input_examples += 1
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                rejected["invalid_messages"] += 1
                continue
            if benchmark_overlap(messages, benchmark_guard):
                rejected["benchmark_overlap"] += 1
                continue
            capability = str(row.get("capability", "unknown"))
            try:
                wrapped, extractions = wrap_messages(messages, capability)
            except ValueError:
                rejected["invalid_assistant"] += 1
                continue
            fitted = fit_complete_turns(
                wrapped,
                tokenizer,
                sequence_length,
                chat_template,
            )
            if fitted is None:
                rejected["does_not_fit_after_envelope"] += 1
                continue
            counts = validate_enveloped_messages(fitted)
            payload = {
                **row,
                "messages": fitted,
                "reasoning_format": FORMAT_ID,
            }
            output.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            output_examples += 1
            source_counts[str(row.get("source", "unknown"))] += 1
            capability_counts[capability] += 1
            extraction_counts.update(extractions)
            marker_counts.update(counts)
    temporary.replace(target)
    return {
        "source_file": source.name,
        "source_sha256": sha256(source),
        "output_sha256": sha256(target),
        "input_examples": input_examples,
        "output_examples": output_examples,
        "rejected": dict(sorted(rejected.items())),
        "sources": dict(sorted(source_counts.items())),
        "capabilities": dict(sorted(capability_counts.items())),
        "extractions": dict(sorted(extraction_counts.items())),
        "special_token_text_counts": {
            token: int(marker_counts[token]) for token in SPECIAL_TOKEN_IDS
        },
    }


def recompile_release(
    source: Path,
    output: Path,
    *,
    tokenizer_path: Path,
    protected_recipe: Path,
    source_repo: str,
    source_revision: str,
) -> dict[str, Any]:
    tokenizer = Tokenizer.load(str(tokenizer_path))
    validate_tokenizer_32004(tokenizer)
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    sequence_length = int(source_manifest["sequence_length"])
    benchmark_recipe = json.loads(protected_recipe.read_text(encoding="utf-8"))
    guard = load_protected_benchmark_prompts(benchmark_recipe)
    output.mkdir(parents=True, exist_ok=True)
    partitions = {
        split: recompile_partition(
            source / filename,
            output / filename,
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            benchmark_guard=guard,
        )
        for split, filename in (("train", "train.jsonl"), ("eval", "eval.jsonl"))
    }
    manifest = {
        "schema_version": 3,
        "name": f"{source_manifest['name']}-vocab32004",
        "source_dataset": source_repo,
        "source_revision": source_revision,
        "source_manifest_sha256": sha256(source / "manifest.json"),
        "sequence_length": sequence_length,
        "reasoning_format": FORMAT_ID,
        "tokenizer_repo": "AETHORIA-AI/TR-HASH-Tokenizer-32K",
        "tokenizer_revision": "f9e5969b721a1af704d007061f52c0942f6dc153",
        "tokenizer_sha256": sha256(tokenizer_path / "tokenizer.json"),
        "tokenizer_vocab_size": len(tokenizer),
        "special_token_ids": SPECIAL_TOKEN_IDS,
        "protected_prompt_count": len(guard.exact),
        "protected_benchmarks": [item["name"] for item in benchmark_recipe["protected_benchmarks"]],
        "partitions": partitions,
        "train_examples": partitions["train"]["output_examples"],
        "eval_examples": partitions["eval"]["output_examples"],
        "train_sha256": partitions["train"]["output_sha256"],
        "eval_sha256": partitions["eval"]["output_sha256"],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata = output / "metadata"
    metadata.mkdir(exist_ok=True)
    (metadata / "source-manifest.json").write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (metadata / "recompile-recipe.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_dataset": source_repo,
                "source_revision": source_revision,
                "tokenizer_repo": manifest["tokenizer_repo"],
                "tokenizer_revision": manifest["tokenizer_revision"],
                "reasoning_format": FORMAT_ID,
                "benchmark_guard_recipe_sha256": sha256(protected_recipe),
                "policy": {
                    "ordinary_answers": "empty_think_full_final",
                    "reasoning_answers": "source_explicit_or_verified_final_extraction",
                    "uncertain_reasoning": "empty_think_full_final",
                    "token_truncation": "forbidden",
                    "legacy_binary_reuse": "forbidden",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--protected-recipe",
        type=Path,
        default=Path("configs/tr_hash_200m_reasoning_sft_500m.json"),
    )
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    report = recompile_release(
        args.source,
        args.output,
        tokenizer_path=args.tokenizer,
        protected_recipe=args.protected_recipe,
        source_repo=args.source_repo,
        source_revision=args.source_revision,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
