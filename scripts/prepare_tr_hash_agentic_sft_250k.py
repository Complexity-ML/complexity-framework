#!/usr/bin/env python3
"""Build the 250K TR-HASH Agentic SFT corpus with conditional thinking."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from tokenizers import Tokenizer

REPOSITORY = "AETHORIA-AI/TR-HASH-Agentic-SFT-32K-250K"
GENERAL_REPOSITORY = "AETHORIA-AI/TR-HASH-Agentic-SFT-32K-21K"
GENERAL_REVISION = "c768cbb398f3c406721f06b2af6ee7edd09aeb2c"
NEMOTRON_REPOSITORY = "nvidia/Nemotron-Agentic-v1"
NEMOTRON_REVISION = "650d590978ca35c8f1ecea2faf136e5fac421b62"
TOKENIZER_REPOSITORY = "AETHORIA-AI/TR-HASH-Tokenizer-32K-Agentic"
TOKENIZER_REVISION = "2fcbc2c5359ded0244ca14531f1b3806eebac55e"

TRAIN_QUOTAS = {
    "direct": 100_000,
    "tool_call": 90_000,
    "no_call": 35_000,
    "tool_final": 25_000,
}
EVAL_QUOTAS = {
    "direct": 5_000,
    "tool_call": 4_500,
    "no_call": 1_750,
    "tool_final": 1_250,
}
TRAIN_THINK_QUOTAS = {"tool_call": 20_000, "no_call": 2_000, "tool_final": 3_000}
EVAL_THINK_QUOTAS = {"tool_call": 1_000, "no_call": 100, "tool_final": 150}

SPECIAL_TOKEN_IDS = {
    "<|system|>": 4,
    "<|user|>": 5,
    "<|assistant|>": 6,
    "<|end_of_turn|>": 7,
    "<|tool_call_start|>": 8,
    "<|tool_call_end|>": 9,
    "<|tool_result_start|>": 10,
    "<|tool_result_end|>": 11,
    "<|think_start|>": 16,
    "<|think_end|>": 17,
    "<|final_start|>": 18,
    "<|final_end|>": 19,
}
THINK_RE = re.compile(r"<\|think_start\|>.*?<\|think_end\|>", re.DOTALL)
THINK_CAPTURE_RE = re.compile(r"<\|think_start\|>(.*?)<\|think_end\|>", re.DOTALL)
MAX_THINK_CHARS = 2_000


def stable_score(record: dict[str, Any], salt: str) -> bytes:
    identity = str(record.get("source_id") or record.get("prompt") or "")
    return hashlib.sha256(f"{salt}\0{identity}".encode()).digest()


def validate_tokenizer(tokenizer: Tokenizer) -> None:
    if tokenizer.get_vocab_size() != 32_000:
        raise ValueError(f"expected vocab 32000, got {tokenizer.get_vocab_size()}")
    for token, expected in SPECIAL_TOKEN_IDS.items():
        actual = tokenizer.token_to_id(token)
        if actual != expected:
            raise ValueError(
                f"special token mismatch for {token}: expected {expected}, got {actual}"
            )


def require_decontaminated(directory: Path) -> None:
    manifest = json.loads((directory / "dataset_info.json").read_text())
    if not manifest.get("piqa_exact_exclusion_enabled"):
        raise ValueError(f"PIQA exclusion is not enabled in {directory}")


def index_candidates(
    path: Path,
    classifier: Callable[[dict[str, Any]], str | None],
    quotas: dict[str, int],
    salt: str,
) -> dict[str, list[tuple[bytes, bytes, int]]]:
    candidates: dict[str, list[tuple[bytes, bytes, int]]] = {key: [] for key in quotas}
    with path.open(encoding="utf-8") as handle:
        for line_number, encoded in enumerate(handle):
            record = json.loads(encoded)
            category = classifier(record)
            if category in candidates:
                thinking_match = THINK_CAPTURE_RE.search(str(record.get("completion", "")))
                compact_thinking = bool(
                    thinking_match and len(thinking_match.group(1)) <= MAX_THINK_CHARS
                )
                candidates[category].append(
                    (
                        stable_score(record, f"{salt}:select"),
                        bytes([not compact_thinking]) + stable_score(record, f"{salt}:think"),
                        line_number,
                    )
                )
    selected: dict[str, list[tuple[bytes, bytes, int]]] = {}
    for category, quota in quotas.items():
        values = sorted(candidates[category])[:quota]
        if len(values) != quota:
            raise RuntimeError(
                f"not enough {category} rows in {path}: found {len(candidates[category])}, need {quota}"
            )
        selected[category] = values
    return selected


def general_category(record: dict[str, Any]) -> str | None:
    if (
        record.get("category") == "direct"
        and record.get("source_dataset") == "HuggingFaceTB/smol-smoltalk"
    ):
        return "direct"
    return None


def agentic_category(record: dict[str, Any]) -> str | None:
    category = record.get("category")
    if category == "direct":
        return "no_call"
    if category in {"tool_call", "tool_final"}:
        return str(category)
    return None


def selected_lines(index: dict[str, list[tuple[bytes, bytes, int]]]) -> dict[int, str]:
    return {line_number: category for category, rows in index.items() for _, _, line_number in rows}


def thinking_lines(
    index: dict[str, list[tuple[bytes, bytes, int]]], quotas: dict[str, int]
) -> set[int]:
    result: set[int] = set()
    for category, quota in quotas.items():
        ranked = sorted(index[category], key=lambda item: (item[1], item[2]))
        if len(ranked) < quota:
            raise RuntimeError(f"not enough {category} rows for thinking quota")
        if any(think_score[0] for _, think_score, _ in ranked[:quota]):
            raise RuntimeError(
                f"not enough compact {category} reasoning rows for thinking quota {quota}"
            )
        result.update(line_number for _, _, line_number in ranked[:quota])
    return result


def normalize_record(
    record: dict[str, Any],
    *,
    category: str,
    thinking: bool,
    tokenizer: Tokenizer,
) -> dict[str, Any]:
    prompt = str(record["prompt"])
    completion = str(record["completion"])
    had_thinking = "<|think_start|>" in completion or "<|think_end|>" in completion
    if had_thinking and not (
        completion.count("<|think_start|>") == completion.count("<|think_end|>") == 1
    ):
        raise ValueError("unbalanced or repeated thinking markers")
    if thinking:
        if not had_thinking:
            raise ValueError(f"selected {category} row has no thinking trace")
    else:
        completion = THINK_RE.sub("", completion)
    if ("<|think_start|>" in completion) != ("<|think_end|>" in completion):
        raise ValueError("unbalanced thinking markers after projection")
    token_count = len(tokenizer.encode(prompt).ids) + len(tokenizer.encode(completion).ids) + 1
    if token_count > 2_049:
        raise ValueError(f"projected row exceeds context: {token_count}")
    source_dataset = str(record.get("source_dataset") or NEMOTRON_REPOSITORY)
    conversation_id = str(record.get("source_id"))
    record_hash = hashlib.sha256((prompt + "\0" + completion).encode()).hexdigest()[:16]
    return {
        "prompt": prompt,
        "completion": completion,
        "source_dataset": source_dataset,
        "source_id": f"{conversation_id}:{record_hash}",
        "source_conversation_id": conversation_id,
        "source_subset": str(record.get("source_subset") or "tool_calling"),
        "split_origin": str(record.get("split_origin") or "stable_hash"),
        "category": category,
        "thinking_supervised": thinking,
        "token_count": token_count,
    }


def write_selected(
    source: Path,
    output,
    *,
    index: dict[str, list[tuple[bytes, bytes, int]]],
    think_quotas: dict[str, int],
    tokenizer: Tokenizer,
    counts: Counter,
    source_ids: set[str],
) -> None:
    wanted = selected_lines(index)
    think = thinking_lines(index, think_quotas) if think_quotas else set()
    with source.open(encoding="utf-8") as handle:
        for line_number, encoded in enumerate(handle):
            category = wanted.get(line_number)
            if category is None:
                continue
            record = normalize_record(
                json.loads(encoded),
                category=category,
                thinking=line_number in think,
                tokenizer=tokenizer,
            )
            identity = record["source_dataset"] + ":" + record["source_id"]
            if identity in source_ids:
                raise ValueError(f"duplicate selected source id: {identity}")
            source_ids.add(identity)
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts[category] += 1
            counts["thinking"] += int(record["thinking_supervised"])
            counts["examples"] += 1
            counts["tokens"] += record["token_count"]


def build_split(
    split: str,
    general_dir: Path,
    agentic_dir: Path,
    output: Path,
    tokenizer: Tokenizer,
) -> dict[str, int]:
    source_name = (
        "validation.jsonl" if (general_dir / "validation.jsonl").exists() else "eval.jsonl"
    )
    general_path = general_dir / ("train.jsonl" if split == "train" else source_name)
    agentic_path = agentic_dir / ("train.jsonl" if split == "train" else "eval.jsonl")
    quotas = TRAIN_QUOTAS if split == "train" else EVAL_QUOTAS
    think_quotas = TRAIN_THINK_QUOTAS if split == "train" else EVAL_THINK_QUOTAS
    general_index = index_candidates(
        general_path, general_category, {"direct": quotas["direct"]}, f"{split}:general"
    )
    agentic_index = index_candidates(
        agentic_path,
        agentic_category,
        {key: quotas[key] for key in ("tool_call", "no_call", "tool_final")},
        f"{split}:agentic",
    )
    counts: Counter = Counter()
    source_ids: set[str] = set()
    with output.open("w", encoding="utf-8") as handle:
        write_selected(
            general_path,
            handle,
            index=general_index,
            think_quotas={},
            tokenizer=tokenizer,
            counts=counts,
            source_ids=source_ids,
        )
        write_selected(
            agentic_path,
            handle,
            index=agentic_index,
            think_quotas=think_quotas,
            tokenizer=tokenizer,
            counts=counts,
            source_ids=source_ids,
        )
    return dict(counts)


def conversation_ids(path: Path) -> set[tuple[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return {
            (record["source_dataset"], record["source_conversation_id"])
            for record in map(json.loads, handle)
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--general-dir", type=Path, required=True)
    parser.add_argument("--agentic-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    tokenizer_file = (
        args.tokenizer / "tokenizer.json" if args.tokenizer.is_dir() else args.tokenizer
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    validate_tokenizer(tokenizer)
    require_decontaminated(args.general_dir)
    require_decontaminated(args.agentic_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        split: build_split(
            split, args.general_dir, args.agentic_dir, args.out_dir / filename, tokenizer
        )
        for split, filename in (("train", "train.jsonl"), ("validation", "eval.jsonl"))
    }
    if splits["train"]["examples"] != 250_000 or splits["validation"]["examples"] != 12_500:
        raise RuntimeError(f"unexpected output totals: {splits}")
    if splits["train"]["thinking"] != 25_000 or splits["validation"]["thinking"] != 1_250:
        raise RuntimeError(f"unexpected thinking totals: {splits}")
    leakage = conversation_ids(args.out_dir / "train.jsonl") & conversation_ids(
        args.out_dir / "eval.jsonl"
    )
    if leakage:
        raise RuntimeError(f"cross-split conversation leakage: {len(leakage)} conversations")

    manifest = {
        "format": "tr-hash-agentic-sft-250k-v1",
        "repository": REPOSITORY,
        "tokenizer": {"repository": TOKENIZER_REPOSITORY, "revision": TOKENIZER_REVISION},
        "sources": {
            "general": {"repository": GENERAL_REPOSITORY, "revision": GENERAL_REVISION},
            "agentic": {"repository": NEMOTRON_REPOSITORY, "revision": NEMOTRON_REVISION},
        },
        "splits": splits,
        "epochs": 3,
        "max_length": 2048,
        "piqa_exact_exclusion_enabled": True,
        "inference_thinking_policy": "auto; never force a think_start prefill",
        "max_retained_thinking_characters": MAX_THINK_CHARS,
        "selection": "deterministic sha256 ranking within source split and category",
    }
    (args.out_dir / "dataset_info.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
- function-calling
- tool-use
- reasoning
- supervised-finetuning
size_categories:
- 100K<n<1M
pretty_name: TR-HASH Agentic SFT 32K 250K
---

# TR-HASH Agentic SFT 32K 250K

Public 250K-example SFT mixture for the native TR-HASH Agentic 32K tokenizer.
It is designed for a three-epoch full-parameter SFT of the 100M refinement checkpoint.

| Train target | Examples |
|---|---:|
| General direct answers | 100,000 |
| Tool calls | 90,000 |
| No-call decisions | 35,000 |
| Final answers after tool results | 25,000 |
| **Total** | **250,000** |

Exactly 25,000 train targets retain source reasoning inside native
`<|think_start|>...<|think_end|>` markers. Other targets teach direct answers,
tool calls, or final answers without forcing a thinking prefill. Inference must
leave thinking automatic. Retained reasoning is capped at 2,000 source
characters so long traces do not dominate a 100M model's supervised tokens.

The validation split contains 12,500 examples with the same source-level split
boundaries and 1,250 reasoning targets. Exact tool-call, no-call, thinking-marker,
and final-answer gates should be reported independently at each checkpoint.

Sources are pinned in `dataset_info.json`. NVIDIA Nemotron-Agentic-v1 is CC BY
4.0; SmolTalk-derived general records are Apache-2.0. The dataset therefore uses
mixed-source licensing and preserves source identifiers for attribution.
""",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
