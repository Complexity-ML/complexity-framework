#!/usr/bin/env python3
"""Build one balanced 500K SFT corpus from the audited 32,004 releases.

The reasoning release already contains the complete general SFT train split.
This builder therefore consumes that train file once, retains every general
example and every non-math reasoning addition, then deterministically samples
the remaining math additions to reach exactly 500,000 unique examples.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

TARGET_EXAMPLES = 500_000
GENERAL_SOURCES = frozenset(
    {
        "bigcode_self_oss_exec",
        "everyday_conversations",
        "luciole_bilingual",
        "luciole_precise_instruction",
        "luciole_python_algorithms",
        "luciole_stem",
        "openr1_math_verified",
        "smoltalk_constraints",
        "smoltalk_magpie_ultra",
        "smoltalk_rewrite",
        "smoltalk_summarize",
    }
)
REASONING_NON_MATH_SOURCES = frozenset(
    {
        "bigcode_self_oss_exec_new",
        "luciole_precise_instruction_new",
        "luciole_stem_new",
        "smoltalk_constraints_new",
    }
)
REASONING_MATH_SOURCES = frozenset(
    {
        "numina_math_15_exact_fill",
        "numina_math_15_validated",
        "openr1_math_verified_new",
    }
)
EXPECTED_SPECIAL_TOKENS = (
    "<|think_start|>",
    "<|think_end|>",
    "<|final_start|>",
    "<|final_end|>",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(seed: int, namespace: str, payload: bytes) -> int:
    digest = hashlib.blake2b(digest_size=16, person=b"trhash-sft-500k")
    digest.update(str(seed).encode("ascii"))
    digest.update(b"\0")
    digest.update(namespace.encode("ascii"))
    digest.update(b"\0")
    digest.update(payload)
    return int.from_bytes(digest.digest(), "big")


def validate_row(row: dict[str, Any]) -> None:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("row has no messages")
    assistant = [message for message in messages if message.get("role") == "assistant"]
    if not assistant:
        raise ValueError("row has no assistant message")
    content = str(assistant[-1].get("content", ""))
    for token in EXPECTED_SPECIAL_TOKENS:
        if content.count(token) != 1:
            raise ValueError(f"final assistant message must contain {token} exactly once")
    positions = [content.index(token) for token in EXPECTED_SPECIAL_TOKENS]
    if positions != sorted(positions):
        raise ValueError("reasoning/final markers are out of order")


def _write_keyed(handle: Any, raw: bytes, *, seed: int) -> None:
    key = stable_key(seed, "shuffle", raw)
    handle.write(f"{key:032x}\t".encode("ascii"))
    handle.write(raw.rstrip(b"\n"))
    handle.write(b"\n")


def _copy_without_sort_key(source: Path, destination: Path) -> None:
    with source.open("rb") as incoming, destination.open("wb") as outgoing:
        for line in incoming:
            _, separator, payload = line.partition(b"\t")
            if not separator:
                raise RuntimeError("sorted row is missing its deterministic key")
            outgoing.write(payload)


def build_train(
    source: Path,
    destination: Path,
    *,
    target_examples: int = TARGET_EXAMPLES,
    seed: int = 42,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    capabilities: Counter[str] = Counter()
    general_count = 0
    non_math_count = 0
    math_seen = 0
    math_heap: list[tuple[int, bytes]] = []

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tr-hash-unified-", dir=destination.parent) as tmp:
        tmp_root = Path(tmp)
        keyed = tmp_root / "selected.keyed.jsonl"
        sorted_keyed = tmp_root / "selected.sorted.jsonl"
        with source.open("rb") as incoming, keyed.open("wb") as selected:
            for line_number, raw in enumerate(incoming, start=1):
                row = json.loads(raw)
                validate_row(row)
                source_name = str(row.get("source", ""))
                capability = str(row.get("capability", "unknown"))
                if source_name in GENERAL_SOURCES:
                    _write_keyed(selected, raw, seed=seed)
                    general_count += 1
                    counts[source_name] += 1
                    capabilities[capability] += 1
                elif source_name in REASONING_NON_MATH_SOURCES:
                    _write_keyed(selected, raw, seed=seed)
                    non_math_count += 1
                    counts[source_name] += 1
                    capabilities[capability] += 1
                elif source_name in REASONING_MATH_SOURCES:
                    math_seen += 1
                    priority = stable_key(seed, "reasoning-math", raw)
                    # Python has a min-heap. Store negative priorities so the
                    # current worst retained row remains at index zero.
                    item = (-priority, raw.rstrip(b"\n"))
                    math_heap.append(item)
                else:
                    raise ValueError(
                        f"unknown source {source_name!r} at {source}:{line_number}"
                    )

        math_target = target_examples - general_count - non_math_count
        if math_target <= 0:
            raise ValueError(
                "target is too small to retain the complete general and non-math splits"
            )
        if math_target > math_seen:
            raise ValueError(f"need {math_target:,} math rows but only saw {math_seen:,}")
        math_heap = heapq.nsmallest(math_target, math_heap, key=lambda item: -item[0])
        with keyed.open("ab") as selected:
            for _, raw in math_heap:
                row = json.loads(raw)
                source_name = str(row["source"])
                capability = str(row.get("capability", "unknown"))
                _write_keyed(selected, raw, seed=seed)
                counts[source_name] += 1
                capabilities[capability] += 1

        sort_env = os.environ.copy()
        sort_env["LC_ALL"] = "C"
        subprocess.run(
            ["sort", "--stable", "-o", str(sorted_keyed), str(keyed)],
            check=True,
            env=sort_env,
        )
        _copy_without_sort_key(sorted_keyed, destination)

    output_examples = general_count + non_math_count + math_target
    if output_examples != target_examples:
        raise RuntimeError(f"built {output_examples:,} rows, expected {target_examples:,}")
    return {
        "examples": output_examples,
        "general_examples": general_count,
        "reasoning_examples": non_math_count + math_target,
        "reasoning_non_math_examples": non_math_count,
        "reasoning_math_examples": math_target,
        "reasoning_math_candidates": math_seen,
        "sources": dict(sorted(counts.items())),
        "capabilities": dict(sorted(capabilities.items())),
        "sha256": sha256(destination),
    }


def combine_eval(general: Path, reasoning: Path, destination: Path, *, seed: int = 42) -> dict:
    counts: Counter[str] = Counter()
    capabilities: Counter[str] = Counter()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tr-hash-unified-eval-", dir=destination.parent) as tmp:
        keyed = Path(tmp) / "eval.keyed.jsonl"
        sorted_keyed = Path(tmp) / "eval.sorted.jsonl"
        examples = 0
        with keyed.open("wb") as selected:
            for source in (general, reasoning):
                with source.open("rb") as incoming:
                    for raw in incoming:
                        row = json.loads(raw)
                        validate_row(row)
                        _write_keyed(selected, raw, seed=seed)
                        examples += 1
                        counts[str(row.get("source", ""))] += 1
                        capabilities[str(row.get("capability", "unknown"))] += 1
        sort_env = os.environ.copy()
        sort_env["LC_ALL"] = "C"
        subprocess.run(
            ["sort", "--stable", "-o", str(sorted_keyed), str(keyed)],
            check=True,
            env=sort_env,
        )
        _copy_without_sort_key(sorted_keyed, destination)
    return {
        "examples": examples,
        "sources": dict(sorted(counts.items())),
        "capabilities": dict(sorted(capabilities.items())),
        "sha256": sha256(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning-train", type=Path, required=True)
    parser.add_argument("--general-eval", type=Path, required=True)
    parser.add_argument("--reasoning-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-examples", type=int, default=TARGET_EXAMPLES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--general-revision", required=True)
    parser.add_argument("--reasoning-revision", required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    train = build_train(
        args.reasoning_train,
        args.output / "train.jsonl",
        target_examples=args.target_examples,
        seed=args.seed,
    )
    evaluation = combine_eval(
        args.general_eval,
        args.reasoning_eval,
        args.output / "eval.jsonl",
        seed=args.seed,
    )
    manifest = {
        "schema_version": 1,
        "name": "tr-hash-moe-200m-unified-sft-v3-32004-500k",
        "sequence_length": 2048,
        "seed": args.seed,
        "train_examples": train["examples"],
        "eval_examples": evaluation["examples"],
        "train_sha256": train["sha256"],
        "eval_sha256": evaluation["sha256"],
        "partitions": {"train": train, "eval": evaluation},
        "special_token_ids": dict(zip(EXPECTED_SPECIAL_TOKENS, range(32000, 32004))),
        "reasoning_format": "tr-hash-think-final-32004-v1",
        "protected_benchmarks": [
            "arc_easy",
            "arc_challenge",
            "piqa",
            "gsm8k",
            "hellaswag",
        ],
        "source_repositories": {
            "general": {
                "repo_id": "AETHORIA-AI/TR-HASH-MoE-200M-SFT-v3-32004-300K",
                "revision": args.general_revision,
            },
            "reasoning": {
                "repo_id": "AETHORIA-AI/TR-HASH-MoE-200M-Reasoning-SFT-v2-32004-500M",
                "revision": args.reasoning_revision,
            },
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
