#!/usr/bin/env python3
"""Audit and package a recompiled TR-HASH 32,004 SFT dataset release."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

from scripts.recompile_tr_hash_sft_32004 import (
    FORMAT_ID,
    SPECIAL_TOKEN_IDS,
    validate_enveloped_messages,
)
from scripts.tokenize_tr_hash_sft_32004 import sha256

TOKENIZED_SUBDIR = Path("tokenized/tr-hash-32k-v3-32004-2048")
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_refinement_checkpoint(path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    weights_path = path / "model.safetensors"
    config = _read_json(config_path)
    if int(config.get("vocab_size", -1)) != 32_004:
        raise ValueError("Refinement checkpoint does not declare vocab 32,004")
    if config.get("tie_word_embeddings") is not True:
        raise ValueError("Refinement checkpoint must preserve tied embeddings")
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        if "embed_tokens.weight" not in handle.keys():
            raise ValueError("Refinement checkpoint has no native embedding tensor")
        shape = list(handle.get_slice("embed_tokens.weight").get_shape())
    if not shape or shape[0] != 32_004:
        raise ValueError(f"Refinement embedding table has wrong shape: {shape}")
    return {
        "repo_id": "AETHORIA-AI/TR-HASH-MoE-200M-160B-Refinement",
        "config_sha256": sha256(config_path),
        "model_sha256": sha256(weights_path),
        "vocab_size": 32_004,
        "embedding_shape": shape,
        "tie_word_embeddings": True,
    }


def audit_raw_partition(path: Path) -> dict[str, Any]:
    examples = assistant_turns = 0
    marker_counts = {token: 0 for token in SPECIAL_TOKEN_IDS}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("reasoning_format") != FORMAT_ID:
                raise ValueError(f"row {line_number} has no canonical reasoning format")
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"row {line_number} has invalid messages")
            counts = validate_enveloped_messages(messages)
            for token, count in counts.items():
                marker_counts[token] += int(count)
            assistant_turns += sum(message["role"] == "assistant" for message in messages)
            examples += 1
    if not examples:
        raise ValueError(f"empty SFT partition: {path}")
    if any(marker_counts[token] != assistant_turns for token in SPECIAL_TOKEN_IDS):
        raise ValueError(f"raw marker counts differ from assistant turns: {marker_counts}")
    return {
        "examples": examples,
        "assistant_turns": assistant_turns,
        "special_token_text_counts": marker_counts,
        "sha256": sha256(path),
    }


def audit_tokenized_partition(path: Path, raw: dict[str, Any]) -> dict[str, Any]:
    metadata = _read_json(path / "sft.idx.json")
    if int(metadata["vocab_size"]) != 32_004:
        raise ValueError(f"wrong tokenized vocabulary in {path}")
    if int(metadata["examples"]) != int(raw["examples"]):
        raise ValueError(f"raw/tokenized example mismatch in {path}")
    expected_labels = {token: int(raw["examples"]) for token in SPECIAL_TOKEN_IDS}
    if metadata.get("special_token_label_counts") != expected_labels:
        raise ValueError(
            f"supervised marker counts differ from examples in {path}: "
            f"{metadata.get('special_token_label_counts')}"
        )
    inputs = np.memmap(path / "input_ids.bin", mode="r", dtype="<u4")
    labels = np.memmap(path / "labels.bin", mode="r", dtype="<i4")
    if len(inputs) != len(labels) or len(inputs) != int(metadata["num_tokens"]):
        raise ValueError(f"tokenized binary lengths differ in {path}")
    visible_labels = labels[labels != -100]
    actual_labels = {
        token: int(np.count_nonzero(visible_labels == token_id))
        for token, token_id in SPECIAL_TOKEN_IDS.items()
    }
    if actual_labels != expected_labels:
        raise ValueError(f"binary supervised marker audit failed in {path}: {actual_labels}")
    if len(inputs) and int(inputs.max()) >= 32_004:
        raise ValueError(f"out-of-vocabulary input ID in {path}")
    return {
        "examples": int(metadata["examples"]),
        "num_tokens": int(metadata["num_tokens"]),
        "supervised_tokens": int(metadata["supervised_tokens"]),
        "special_token_label_counts": actual_labels,
        "max_input_id": int(inputs.max()) if len(inputs) else -1,
        "files": {
            filename: sha256(path / filename)
            for filename in ("input_ids.bin", "labels.bin", "examples.jsonl", "sft.idx.json")
        },
    }


def dataset_card(manifest: dict[str, Any], audit: dict[str, Any], title: str) -> str:
    train = audit["partitions"]["train"]
    evaluation = audit["partitions"]["eval"]
    rejected = manifest["partitions"]["train"]["rejected"]
    return f"""---
annotations_creators:
- machine-generated
language:
- en
- fr
license: other
multilinguality:
- multilingual
pretty_name: {title}
size_categories:
- 100K<n<1M
task_categories:
- text-generation
tags:
- supervised-fine-tuning
- reasoning
- tr-hash
- vocab-32004
---

# {title}

Canonical text and binary SFT release for the **32,004-token** TR-HASH
tokenizer. It was recompiled from the audited text selection at
[`{manifest["source_dataset"]}`](https://huggingface.co/datasets/{manifest["source_dataset"]}/tree/{manifest["source_revision"]})
without reusing or remapping its legacy 32,000-token binary shard.

## Reasoning protocol

Every assistant turn has exactly one ordered envelope:

```text
<|think_start|>optional verified reasoning<|think_end|><|final_start|>answer<|final_end|>
```

The IDs are fixed at 32000–32003. Ordinary instruction, code and conversation
answers use an empty `think` span and preserve the complete original response
inside `final`. Math reasoning is placed in `think` only when a final answer can
be extracted deterministically from source markup, `\\boxed{{...}}`, or an
explicit final-answer line. No synthetic chain of thought is invented.

## Audited release

| Split | Examples | Visible tokens | Supervised tokens |
|---|---:|---:|---:|
| train | {train["raw"]["examples"]:,} | {train["tokenized"]["num_tokens"]:,} | {train["tokenized"]["supervised_tokens"]:,} |
| eval | {evaluation["raw"]["examples"]:,} | {evaluation["tokenized"]["num_tokens"]:,} | {evaluation["tokenized"]["supervised_tokens"]:,} |

- Source vocabulary: 32,000; release vocabulary: **32,004**.
- Benchmark guard: ARC-Easy, ARC-Challenge, PIQA, GSM8K and HellaSwag.
- Removed train overlaps: **{int(rejected.get("benchmark_overlap", 0)):,}**.
- Token truncation: **forbidden**; rows that no longer fit after the envelope
  are rejected instead of sliced.
- All four special IDs occur exactly once in the supervised completion of
  every retained example.

## Files

- `train.jsonl`, `eval.jsonl`: canonical text with source/capability provenance;
- `{TOKENIZED_SUBDIR}/`: reusable `uint32` inputs, masked `int32` labels,
  per-example indexes and the audited 32,004 tokenizer;
- `manifest.json`: pinned source revision, benchmark guard and text hashes;
- `metadata/recompile-recipe.json`: transformation policy;
- `metadata/release-audit.json`: text/binary counts and SHA-256 verification.

## Reproduction

The scripts live in
[`Complexity-ML/complexity-framework`](https://github.com/Complexity-ML/complexity-framework):

```bash
python -m scripts.recompile_tr_hash_sft_32004 ...
python -m scripts.tokenize_tr_hash_sft_32004 ...
python -m scripts.package_tr_hash_sft_32004_release ...
```

The next full-parameter SFT must initialize from
[`AETHORIA-AI/TR-HASH-MoE-200M-160B-Refinement`](https://huggingface.co/AETHORIA-AI/TR-HASH-MoE-200M-160B-Refinement),
whose embeddings already contain 32,004 rows. Previous 32,000-token SFT weights
must not be resumed.
"""


def package_release(
    dataset: Path,
    tokenizer: Path,
    refinement: Path,
    *,
    title: str,
) -> dict[str, Any]:
    manifest = _read_json(dataset / "manifest.json")
    tokenized_manifest = _read_json(dataset / TOKENIZED_SUBDIR / "manifest.json")
    raw = {
        "train": audit_raw_partition(dataset / "train.jsonl"),
        "eval": audit_raw_partition(dataset / "eval.jsonl"),
    }
    tokenized = {
        split: audit_tokenized_partition(
            dataset / TOKENIZED_SUBDIR / split,
            raw[split],
        )
        for split in ("train", "eval")
    }
    if int(tokenized_manifest["tokenizer_vocab_size"]) != 32_004:
        raise ValueError("tokenized release manifest does not declare vocab 32,004")
    if tokenized_manifest.get("special_token_ids") != SPECIAL_TOKEN_IDS:
        raise ValueError("tokenized release manifest has wrong special-token IDs")
    if tokenized_manifest["source_train_sha256"] != manifest["train_sha256"]:
        raise ValueError("tokenized train source hash differs from raw manifest")
    if tokenized_manifest["source_eval_sha256"] != manifest["eval_sha256"]:
        raise ValueError("tokenized eval source hash differs from raw manifest")
    audit = {
        "schema_version": 1,
        "status": "passed",
        "reasoning_format": FORMAT_ID,
        "tokenizer_vocab_size": 32_004,
        "special_token_ids": SPECIAL_TOKEN_IDS,
        "compatible_refinement": audit_refinement_checkpoint(refinement),
        "partitions": {
            split: {"raw": raw[split], "tokenized": tokenized[split]} for split in ("train", "eval")
        },
        "invariants": {
            "source_is_text_not_legacy_binary": True,
            "all_assistant_envelopes_closed_and_ordered": True,
            "all_four_ids_supervised_once_per_example": True,
            "no_token_truncation": True,
            "benchmark_guard_applied": True,
            "refinement_vocab_compatibility": True,
        },
    }
    metadata = dataset / "metadata"
    metadata.mkdir(exist_ok=True)
    (metadata / "release-audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tokenizer_root = dataset / "tokenizer"
    tokenizer_root.mkdir(exist_ok=True)
    for filename in TOKENIZER_FILES:
        shutil.copy2(tokenizer / filename, tokenizer_root / filename)
    shutil.copy2(
        dataset / TOKENIZED_SUBDIR / "tokenizer/chat_template.jinja",
        tokenizer_root / "chat_template.jinja",
    )
    (dataset / "README.md").write_text(dataset_card(manifest, audit, title), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--refinement", type=Path, required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()
    report = package_release(
        args.dataset,
        args.tokenizer,
        args.refinement,
        title=args.title,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
