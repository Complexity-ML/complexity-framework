#!/usr/bin/env python3
"""Build a 150M general + 50M filtered reasoning token SFT shard.

The source releases are already encoded with the same 32K tokenizer and chat
template.  This builder selects complete examples directly from their audited
binary shards, preserving labels exactly and avoiding a second text decoding or
tokenization pass.  Selection and final interleaving are deterministic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from complexity.training.sft_shard import (
    FINAL_ASSISTANT_SUPERVISION,
    MASKED_ASSISTANT_HISTORY,
    SHARD_FORMAT_V2,
)

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_score(seed: int, namespace: str, example_id: str) -> bytes:
    return hashlib.sha256(f"{seed}:{namespace}:{example_id}".encode()).digest()


def load_recipe(path: Path) -> dict[str, Any]:
    recipe = json.loads(path.read_text(encoding="utf-8"))
    general = int(recipe.get("general_replay_tokens", 0))
    reasoning = int(recipe.get("reasoning_tokens", 0))
    total = int(recipe.get("target_formatted_tokens", 0))
    quotas = recipe.get("reasoning_source_quotas")
    if general != 150_000_000 or reasoning != 50_000_000 or total != 200_000_000:
        raise ValueError("preservation recipe must be exactly 150M general + 50M reasoning")
    if general + reasoning != total:
        raise ValueError("general and reasoning budgets do not sum to the target")
    if not isinstance(quotas, list) or not quotas:
        raise ValueError("reasoning source quotas are required")
    names = [str(item["source"]) for item in quotas]
    if len(names) != len(set(names)):
        raise ValueError("reasoning source quota names must be unique")
    quota_total = sum(int(item["tokens"]) for item in quotas)
    if quota_total != reasoning:
        raise ValueError(f"reasoning quotas sum to {quota_total}, expected {reasoning}")
    if int(recipe.get("sequence_length", 0)) != 2_048:
        raise ValueError("preservation recipe must use the released 2,048-token context")
    return recipe


def read_examples(partition: Path, *, origin: str) -> list[dict[str, Any]]:
    index_path = partition / "examples.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    examples: list[dict[str, Any]] = []
    with index_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            num_tokens = int(row["num_tokens"])
            offset = int(row["offset"])
            if num_tokens <= 0 or offset < 0:
                raise ValueError(f"invalid shard row {index_path}:{line_number}")
            row = dict(row)
            row["origin"] = origin
            row["source_partition"] = str(partition.resolve())
            examples.append(row)
    if not examples:
        raise ValueError(f"empty SFT partition: {partition}")
    expected_tokens = max(int(row["offset"]) + int(row["num_tokens"]) for row in examples)
    sizes = {
        "input_ids.bin": (partition / "input_ids.bin").stat().st_size // 4,
        "labels.bin": (partition / "labels.bin").stat().st_size // 4,
    }
    if any(size != expected_tokens for size in sizes.values()):
        raise ValueError(
            f"binary/index token count mismatch for {partition}: {sizes} != {expected_tokens}"
        )
    return examples


def select_budget(
    examples: list[dict[str, Any]],
    target_tokens: int,
    *,
    seed: int,
    namespace: str,
) -> tuple[list[dict[str, Any]], int]:
    ordered = sorted(
        examples,
        key=lambda row: stable_score(seed, namespace, str(row["example_id"])),
    )
    selected: list[dict[str, Any]] = []
    tokens = 0
    for row in ordered:
        selected.append(row)
        tokens += int(row["num_tokens"])
        if tokens >= target_tokens:
            break
    if tokens < target_tokens:
        raise ValueError(
            f"{namespace} only provides {tokens:,} tokens, below target {target_tokens:,}"
        )
    return selected, tokens


def select_training_examples(
    general: list[dict[str, Any]],
    reasoning: list[dict[str, Any]],
    recipe: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed = int(recipe["seed"])
    general_selected, general_tokens = select_budget(
        general,
        int(recipe["general_replay_tokens"]),
        seed=seed,
        namespace="general-replay",
    )
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reasoning:
        by_source[str(row.get("source", "unknown"))].append(row)
    reasoning_selected: list[dict[str, Any]] = []
    source_tokens: dict[str, int] = {}
    for quota in recipe["reasoning_source_quotas"]:
        source = str(quota["source"])
        selected, tokens = select_budget(
            by_source.get(source, []),
            int(quota["tokens"]),
            seed=seed,
            namespace=f"reasoning:{source}",
        )
        reasoning_selected.extend(selected)
        source_tokens[source] = tokens
    reasoning_tokens = sum(source_tokens.values())
    selected = general_selected + reasoning_selected
    selected.sort(
        key=lambda row: stable_score(
            seed,
            "final-interleave",
            f"{row['origin']}:{row['example_id']}",
        )
    )
    return selected, {
        "general_examples": len(general_selected),
        "general_tokens": general_tokens,
        "reasoning_examples": len(reasoning_selected),
        "reasoning_tokens": reasoning_tokens,
        "reasoning_source_tokens": source_tokens,
        "total_examples": len(selected),
        "total_tokens": general_tokens + reasoning_tokens,
    }


def materialize_partition(
    examples: list[dict[str, Any]],
    target: Path,
    *,
    partition_name: str,
) -> dict[str, Any]:
    target.mkdir(parents=True, exist_ok=False)
    input_path = target / "input_ids.bin"
    label_path = target / "labels.bin"
    index_path = target / "examples.jsonl"
    arrays: dict[tuple[str, str], np.memmap] = {}
    total_tokens = supervised_tokens = prompt_tokens = 0
    sources: Counter[str] = Counter()
    capabilities: Counter[str] = Counter()

    def array(row: dict[str, Any], filename: str, dtype: str) -> np.memmap:
        key = (str(row["source_partition"]), filename)
        if key not in arrays:
            arrays[key] = np.memmap(Path(key[0]) / filename, mode="r", dtype=dtype)
        return arrays[key]

    with (
        input_path.open("wb") as input_handle,
        label_path.open("wb") as label_handle,
        index_path.open("w", encoding="utf-8") as index_handle,
    ):
        for output_index, row in enumerate(examples):
            offset = int(row["offset"])
            num_tokens = int(row["num_tokens"])
            inputs = array(row, "input_ids.bin", "<u4")[offset : offset + num_tokens]
            labels = array(row, "labels.bin", "<i4")[offset : offset + num_tokens]
            if len(inputs) != num_tokens or len(labels) != num_tokens:
                raise ValueError(f"short binary slice for {row['example_id']}")
            inputs.tofile(input_handle)
            labels.tofile(label_handle)
            source = str(row.get("source", "unknown"))
            capability = str(row.get("capability", "unknown"))
            prompt_count = int(row.get("prompt_tokens", 0))
            supervised_count = int(row.get("supervised_tokens", np.count_nonzero(labels != -100)))
            output_row = {
                "example_id": f"{partition_name}-{output_index:07d}",
                "origin": row["origin"],
                "source_example_id": row["example_id"],
                "task": capability,
                "source": source,
                "capability": capability,
                "offset": total_tokens,
                "num_tokens": num_tokens,
                "prompt_tokens": prompt_count,
                "supervised_tokens": supervised_count,
            }
            index_handle.write(
                json.dumps(output_row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            total_tokens += num_tokens
            prompt_tokens += prompt_count
            supervised_tokens += supervised_count
            sources[source] += 1
            capabilities[capability] += 1
        for handle in (input_handle, label_handle, index_handle):
            handle.flush()
            os.fsync(handle.fileno())

    metadata = {
        "format": SHARD_FORMAT_V2,
        "assistant_supervision": FINAL_ASSISTANT_SUPERVISION,
        "history_assistant_turns": MASKED_ASSISTANT_HISTORY,
        "examples": len(examples),
        "num_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "supervised_tokens": supervised_tokens,
        "input_dtype": "uint32-le",
        "label_dtype": "int32-le",
        "sources": dict(sorted(sources.items())),
        "capabilities": dict(sorted(capabilities.items())),
        "files": {
            "input_ids.bin": sha256(input_path),
            "labels.bin": sha256(label_path),
            "examples.jsonl": sha256(index_path),
        },
    }
    (target / "sft.idx.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def verify_source_compatibility(general_root: Path, reasoning_root: Path) -> tuple[dict, dict]:
    general_manifest = json.loads((general_root / "manifest.json").read_text(encoding="utf-8"))
    reasoning_manifest = json.loads((reasoning_root / "manifest.json").read_text(encoding="utf-8"))
    keys = (
        "tokenizer_vocab_size",
        "tokenizer_sha256",
        "chat_template_id",
        "chat_template_eos_token",
    )
    mismatched = [key for key in keys if general_manifest.get(key) != reasoning_manifest.get(key)]
    if mismatched:
        raise ValueError(f"source shard tokenizer/chat-template mismatch: {mismatched}")
    if any(
        manifest.get("sequence_length_cap") != 2_048
        for manifest in (general_manifest, reasoning_manifest)
    ):
        raise ValueError("source shards must both use the released 2,048-token context")
    if any(
        manifest.get("release_quality", {}).get("ready") is not True
        for manifest in (general_manifest, reasoning_manifest)
    ):
        raise ValueError("source shards must both be release-ready")
    return general_manifest, reasoning_manifest


def build_release(
    *,
    general_root: Path,
    reasoning_root: Path,
    output: Path,
    recipe_path: Path,
    general_repo: str,
    reasoning_repo: str,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    recipe = load_recipe(recipe_path)
    general_manifest, reasoning_manifest = verify_source_compatibility(general_root, reasoning_root)
    source_index = json.loads((general_root / "train" / "sft.idx.json").read_text(encoding="utf-8"))
    general_train = read_examples(general_root / "train", origin="sft-v2-general")
    reasoning_train = read_examples(reasoning_root / "train", origin="reasoning-500m")
    selected, selection = select_training_examples(general_train, reasoning_train, recipe)
    total = int(selection["total_tokens"])
    if not 200_000_000 <= total < 200_020_000:
        raise ValueError(f"selected token total outside bounded overshoot: {total}")

    output.mkdir(parents=True)
    train = materialize_partition(selected, output / "train", partition_name="train")
    general_eval = read_examples(general_root / "eval", origin="sft-v2-general")
    reasoning_eval = read_examples(reasoning_root / "eval", origin="reasoning-500m")
    eval_examples = general_eval + reasoning_eval
    eval_examples.sort(
        key=lambda row: stable_score(
            int(recipe["seed"]),
            "eval-interleave",
            f"{row['origin']}:{row['example_id']}",
        )
    )
    evaluation = materialize_partition(eval_examples, output / "eval", partition_name="eval")

    inherited_index_keys = (
        "chat_template_eos_token",
        "chat_template_id",
        "eos_token_id",
        "sequence_length_cap",
        "truncation_policy",
        "vocab_size",
    )
    inherited_index = {key: source_index[key] for key in inherited_index_keys}
    for partition_name, metadata in (("train", train), ("eval", evaluation)):
        metadata.update(inherited_index)
        (output / partition_name / "sft.idx.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    shutil.copy2(general_root / "chat_template.json", output / "chat_template.json")
    tokenizer_output = output / "tokenizer"
    tokenizer_output.mkdir()
    for filename in TOKENIZER_FILES:
        source = general_root / "tokenizer" / filename
        if source.is_file():
            shutil.copy2(source, tokenizer_output / filename)
    manifest = {
        "schema_version": 1,
        "format": SHARD_FORMAT_V2,
        "name": recipe["name"],
        "quality_status": "passed",
        "release_quality": {
            "ready": True,
            "token_truncation": False,
            "benchmark_isolation": "inherited-and-source-filtered",
        },
        "recipe_sha256": sha256(recipe_path),
        "source_datasets": {
            "general": general_repo,
            "reasoning": reasoning_repo,
        },
        "source_manifest_sha256": {
            "general": sha256(general_root / "manifest.json"),
            "reasoning": sha256(reasoning_root / "manifest.json"),
        },
        "nominal_target_formatted_tokens": int(recipe["target_formatted_tokens"]),
        "actual_unique_formatted_tokens": total,
        "selection": selection,
        "tokenizer_vocab_size": int(general_manifest["tokenizer_vocab_size"]),
        "tokenizer_sha256": general_manifest["tokenizer_sha256"],
        "chat_template_id": general_manifest["chat_template_id"],
        "chat_template_eos_token": general_manifest["chat_template_eos_token"],
        "sequence_length_cap": 2_048,
        "assistant_supervision": FINAL_ASSISTANT_SUPERVISION,
        "history_assistant_turns": MASKED_ASSISTANT_HISTORY,
        "partitions": {"train": train, "eval": evaluation},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--general-shard", type=Path, required=True)
    parser.add_argument("--reasoning-shard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=Path("configs/tr_hash_200m_reasoning_preservation_50m.json"),
    )
    parser.add_argument("--general-repo", default="AETHORIA-AI/TR-HASH-MoE-200M-SFT-v2-300K")
    parser.add_argument(
        "--reasoning-repo",
        default="AETHORIA-AI/TR-HASH-MoE-200M-Reasoning-SFT-500M",
    )
    args = parser.parse_args()
    manifest = build_release(
        general_root=args.general_shard,
        reasoning_root=args.reasoning_shard,
        output=args.output,
        recipe_path=args.recipe,
        general_repo=args.general_repo,
        reasoning_repo=args.reasoning_repo,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
