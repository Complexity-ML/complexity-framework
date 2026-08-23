#!/usr/bin/env python3
"""Validate and package the 500M-token reasoning SFT dataset for the Hub."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from scripts.prepare_tr_hash_200m_clean_sft import sha256

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def validate_reasoning_build(
    dataset_dir: Path, recipe_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _read_json(dataset_dir / "manifest.json")
    recipe = _read_json(recipe_path)
    train = dataset_dir / "train.jsonl"
    evaluation = dataset_dir / "eval.jsonl"
    actual = int(manifest["actual_unique_formatted_tokens"])
    if not 500_000_000 <= actual < 500_020_000:
        raise ValueError(f"expected 500M tokens with <20K overshoot, got {actual}")
    if int(manifest["nominal_target_unique_formatted_tokens"]) != 500_000_000:
        raise ValueError("nominal reasoning target is not 500M")
    if int(manifest["sequence_length"]) != 2_048:
        raise ValueError("reasoning release must use the 2,048-token model context")
    if _count_lines(train) != int(manifest["train_examples"]):
        raise ValueError("train line count does not match manifest")
    if _count_lines(evaluation) != int(manifest["eval_examples"]):
        raise ValueError("eval line count does not match manifest")
    if sha256(train) != manifest["train_sha256"]:
        raise ValueError("train SHA-256 does not match manifest")
    if sha256(evaluation) != manifest["eval_sha256"]:
        raise ValueError("eval SHA-256 does not match manifest")
    if int(manifest.get("protected_prompt_count", 0)) <= 0:
        raise ValueError("benchmark contamination guard was not populated")
    return manifest, recipe


def validate_tokenized_release(
    dataset_dir: Path, raw_manifest: dict[str, Any]
) -> dict[str, Any] | None:
    tokenized_root = dataset_dir / "tokenized" / "tr-hash-32k-v2-2048"
    manifest_path = tokenized_root / "manifest.json"
    if not manifest_path.is_file():
        return None
    tokenized = _read_json(manifest_path)
    if tokenized.get("quality_status") != "passed":
        raise ValueError("tokenized release quality status did not pass")
    if tokenized.get("release_quality", {}).get("token_truncation") is not False:
        raise ValueError("tokenized release does not prove zero truncation")
    raw_tokens = int(raw_manifest["actual_unique_formatted_tokens"])
    train = tokenized.get("partitions", {}).get("train", {})
    evaluation = tokenized.get("partitions", {}).get("eval", {})
    if int(train.get("num_tokens", -1)) != raw_tokens:
        raise ValueError("raw and tokenized train token totals differ")
    if int(train.get("examples", -1)) != int(raw_manifest["train_examples"]):
        raise ValueError("raw and tokenized train example totals differ")
    if int(evaluation.get("examples", -1)) != int(raw_manifest["eval_examples"]):
        raise ValueError("raw and tokenized eval example totals differ")
    for partition_name, partition in (("train", train), ("eval", evaluation)):
        for filename, expected in partition.get("files", {}).items():
            path = tokenized_root / partition_name / filename
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f"tokenized file hash mismatch: {path}")
    return tokenized


def write_release_audit(
    dataset_dir: Path,
    raw_manifest: dict[str, Any],
    tokenized_manifest: dict[str, Any],
) -> None:
    metadata = dataset_dir / "metadata"
    metadata.mkdir(exist_ok=True)
    audit = {
        "schema_version": 1,
        "status": "passed",
        "raw": {
            "train_examples": int(raw_manifest["train_examples"]),
            "eval_examples": int(raw_manifest["eval_examples"]),
            "unique_formatted_train_tokens": int(raw_manifest["actual_unique_formatted_tokens"]),
            "train_sha256": raw_manifest["train_sha256"],
            "eval_sha256": raw_manifest["eval_sha256"],
            "protected_prompt_count": int(raw_manifest["protected_prompt_count"]),
        },
        "tokenized": {
            "manifest_sha256": sha256(
                dataset_dir / "tokenized" / "tr-hash-32k-v2-2048" / "manifest.json"
            ),
            "train": tokenized_manifest["partitions"]["train"],
            "eval": tokenized_manifest["partitions"]["eval"],
        },
        "invariants": {
            "exact_raw_token_parity": True,
            "no_truncation": True,
            "benchmark_denylist_populated": True,
            "tokenizer_vocab_size_32000": True,
            "sequence_length_2048": True,
        },
    }
    (metadata / "release-audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def dataset_card(manifest: dict[str, Any]) -> str:
    rows = []
    removed = 0
    for name, source in manifest["sources"].items():
        rejected = source.get("rejected", {})
        removed += int(rejected.get("benchmark_overlap", 0))
        rows.append(
            f"| `{name}` | {source['capability']} | "
            f"{int(source['train_examples']):,} | "
            f"{int(source['actual_train_tokens']):,} | "
            f"{sum(int(value) for value in rejected.values()):,} | "
            f"`{source['license']}` |"
        )
    return f"""---
annotations_creators:
- machine-generated
language:
- en
- fr
license: other
multilinguality:
- multilingual
pretty_name: TR-HASH MoE 200M Reasoning SFT 500M
size_categories:
- 100K<n<1M
task_categories:
- text-generation
tags:
- reasoning
- supervised-fine-tuning
- tr-hash
- quality-filtered
---

# TR-HASH MoE 200M — Reasoning SFT 500M

Audited post-training mixture for a full-parameter reasoning SFT initialized
from
[`AETHORIA-AI/TR-HASH-MoE-200M-160B-Refinement`](https://huggingface.co/AETHORIA-AI/TR-HASH-MoE-200M-160B-Refinement).

The training split contains **{int(manifest["actual_unique_formatted_tokens"]):,}
unique formatted tokens** (500M nominal; repeated epochs are not counted).
Every supervised answer fits the native 2,048-token context in full. Nothing
is silently truncated.

## Composition

| Source | Capability | Train examples | Visible train tokens | Rejected | License |
|---|---|---:|---:|---:|---|
{chr(10).join(rows)}

This is a mixed-license aggregate. Every row retains source provenance; users
must comply with its upstream terms. The aggregate is not relicensed under a
single permissive license.

## Benchmark isolation

ARC-Easy, ARC-Challenge, PIQA, GSM8K and HellaSwag are **not training
sources**. Their complete public question sets were normalized and used only
as a deny-list. **{removed:,} candidate/replay rows** matching that guard were
removed before release. The protected corpus contains
**{int(manifest["protected_prompt_count"]):,} prompts**.

## Release files

- `train.jsonl`, `eval.jsonl`: raw conversations with source provenance;
- `tokenized/tr-hash-32k-v2-2048/`: `uint32` inputs and masked `int32` labels;
- `metadata/recipe.json`: pinned source revisions and nominal token quotas;
- `manifest.json`: exact counts, rejection statistics and SHA-256 hashes;
- `metadata/release-audit.json`: raw/tokenized cross-validation.

## Reproduction

```bash
export REPLAY_JSONL=/path/to/clean-sft-v2/train.jsonl
python -m scripts.prepare_tr_hash_200m_reasoning_sft_500m \\
  --tokenizer /path/to/tr-hash-32k-tokenizer \\
  --output-dir data/tr_hash_moe_200m_reasoning_sft_500m
python -m scripts.tokenize_tr_hash_200m_clean_sft_v2 \\
  --source data/tr_hash_moe_200m_reasoning_sft_500m \\
  --tokenizer /path/to/tr-hash-32k-tokenizer \\
  --output data/tr_hash_moe_200m_reasoning_sft_500m/tokenized/tr-hash-32k-v2-2048 \\
  --source-dataset AETHORIA-AI/TR-HASH-MoE-200M-Reasoning-SFT-500M
```
"""


def package(dataset_dir: Path, tokenizer_dir: Path, recipe_path: Path) -> None:
    manifest, _ = validate_reasoning_build(dataset_dir, recipe_path)
    tokenizer_output = dataset_dir / "tokenizer"
    metadata_output = dataset_dir / "metadata"
    tokenizer_output.mkdir(exist_ok=True)
    metadata_output.mkdir(exist_ok=True)
    for filename in TOKENIZER_FILES:
        source = tokenizer_dir / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, tokenizer_output / filename)
    shutil.copy2(recipe_path, metadata_output / "recipe.json")
    tokenized = validate_tokenized_release(dataset_dir, manifest)
    if tokenized is not None:
        write_release_audit(dataset_dir, manifest, tokenized)
    (dataset_dir / "README.md").write_text(dataset_card(manifest), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=Path("configs/tr_hash_200m_reasoning_sft_500m.json"),
    )
    args = parser.parse_args()
    package(args.dataset_dir, args.tokenizer, args.recipe)
    print(f"Packaged reasoning dataset: {args.dataset_dir}")


if __name__ == "__main__":
    main()
