#!/usr/bin/env python3
"""Filter audited TR-HASH 32,004-token SFT shards without re-tokenizing.

The raw unified selection preserves source JSON rows byte-for-byte, only
changing their order. This tool matches those rows by SHA-256, copies their
already-audited token segments, and rebuilds offsets and release metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

from complexity.training.sft_shard import (
    FINAL_ASSISTANT_SUPERVISION,
    MASKED_ASSISTANT_HISTORY,
    SHARD_FORMAT_V2,
)

try:
    from scripts.tokenize_tr_hash_sft_32004 import sha256
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from tokenize_tr_hash_sft_32004 import sha256

TOKENIZED_SUBDIR = Path("tokenized/tr-hash-32k-v3-32004-2048")
SPECIAL_TOKEN_IDS = {
    "<|think_start|>": 32_000,
    "<|think_end|>": 32_001,
    "<|final_start|>": 32_002,
    "<|final_end|>": 32_003,
}
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
    "chat_template.jinja",
)


def _digest(raw: bytes) -> bytes:
    return hashlib.sha256(raw.rstrip(b"\r\n")).digest()


def selected_positions(path: Path) -> tuple[dict[bytes, deque[int]], int]:
    positions: dict[bytes, deque[int]] = defaultdict(deque)
    examples = 0
    with path.open("rb") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            positions[_digest(raw)].append(examples)
            examples += 1
    if not examples:
        raise ValueError(f"empty selected partition: {path}")
    return dict(positions), examples


def resolve_selected_records(
    selected: Path,
    sources: list[tuple[Path, Path]],
) -> list[tuple[Path, dict[str, Any]]]:
    pending, count = selected_positions(selected)
    records: list[tuple[Path, dict[str, Any]] | None] = [None] * count

    for raw_path, tokenized_partition in sources:
        with (
            raw_path.open("rb") as raw_handle,
            (tokenized_partition / "examples.jsonl").open(encoding="utf-8") as index_handle,
        ):
            indexed = iter(index_handle)
            source_examples = 0
            for raw in raw_handle:
                if not raw.strip():
                    continue
                try:
                    metadata = json.loads(next(indexed))
                except StopIteration as exc:
                    raise ValueError(f"token index ended before raw source: {raw_path}") from exc
                source_examples += 1
                positions = pending.get(_digest(raw))
                if positions:
                    output_position = positions.popleft()
                    records[output_position] = (tokenized_partition, metadata)
                    if not positions:
                        pending.pop(_digest(raw), None)
            if next(indexed, None) is not None:
                raise ValueError(
                    f"token index has more rows than raw source: {tokenized_partition}"
                )
            source_metadata = json.loads(
                (tokenized_partition / "sft.idx.json").read_text(encoding="utf-8")
            )
            if source_examples != int(source_metadata["examples"]):
                raise ValueError(
                    f"raw/tokenized example mismatch for {raw_path}: "
                    f"{source_examples} != {source_metadata['examples']}"
                )

    if pending:
        missing = sum(len(positions) for positions in pending.values())
        raise ValueError(f"{missing:,} selected rows were not found in tokenized sources")
    if any(record is None for record in records):
        raise RuntimeError("selected record resolution left empty output positions")
    return [record for record in records if record is not None]


def materialize_partition(
    selected: Path,
    sources: list[tuple[Path, Path]],
    target: Path,
) -> dict[str, Any]:
    resolved = resolve_selected_records(selected, sources)
    target.mkdir(parents=True, exist_ok=True)
    temporary = {
        "input_ids.bin": target / "input_ids.bin.partial",
        "labels.bin": target / "labels.bin.partial",
        "examples.jsonl": target / "examples.jsonl.partial",
    }
    arrays: dict[Path, tuple[np.memmap, np.memmap]] = {}
    sources_count: Counter[str] = Counter()
    capabilities: Counter[str] = Counter()
    special_inputs: Counter[str] = Counter()
    special_labels: Counter[str] = Counter()
    total_tokens = prompt_tokens = supervised_tokens = 0

    with (
        temporary["input_ids.bin"].open("wb") as input_handle,
        temporary["labels.bin"].open("wb") as label_handle,
        temporary["examples.jsonl"].open("w", encoding="utf-8") as index_handle,
    ):
        for output_index, (partition, metadata) in enumerate(resolved):
            if partition not in arrays:
                arrays[partition] = (
                    np.memmap(partition / "input_ids.bin", mode="r", dtype="<u4"),
                    np.memmap(partition / "labels.bin", mode="r", dtype="<i4"),
                )
            inputs, labels = arrays[partition]
            offset = int(metadata["offset"])
            num_tokens = int(metadata["num_tokens"])
            source_inputs = np.asarray(inputs[offset : offset + num_tokens], dtype="<u4")
            source_labels = np.asarray(labels[offset : offset + num_tokens], dtype="<i4")
            if len(source_inputs) != num_tokens or len(source_labels) != num_tokens:
                raise ValueError(f"out-of-range source segment in {partition}: {metadata}")
            source_inputs.tofile(input_handle)
            source_labels.tofile(label_handle)

            rewritten = dict(metadata)
            rewritten["example_id"] = f"{selected.stem}-{output_index:06d}"
            rewritten["offset"] = total_tokens
            index_handle.write(
                json.dumps(rewritten, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            total_tokens += num_tokens
            prompt_tokens += int(metadata["prompt_tokens"])
            supervised_tokens += int(metadata["supervised_tokens"])
            sources_count[str(metadata.get("source", "unknown"))] += 1
            capabilities[str(metadata.get("capability", "unknown"))] += 1
            visible_labels = source_labels[source_labels != -100]
            for token, token_id in SPECIAL_TOKEN_IDS.items():
                special_inputs[token] += int(np.count_nonzero(source_inputs == token_id))
                special_labels[token] += int(np.count_nonzero(visible_labels == token_id))

    final_paths = {
        name: target / name for name in ("input_ids.bin", "labels.bin", "examples.jsonl")
    }
    for name, destination in final_paths.items():
        temporary[name].replace(destination)
    expected_markers = {token: len(resolved) for token in SPECIAL_TOKEN_IDS}
    actual_markers = {token: int(special_labels[token]) for token in SPECIAL_TOKEN_IDS}
    if actual_markers != expected_markers:
        raise ValueError(f"filtered shard lost supervised reasoning markers: {actual_markers}")
    metadata = {
        "format": SHARD_FORMAT_V2,
        "assistant_supervision": FINAL_ASSISTANT_SUPERVISION,
        "history_assistant_turns": MASKED_ASSISTANT_HISTORY,
        "chat_template_id": "complexity-chat-v3-32004",
        "source_sha256": sha256(selected),
        "examples": len(resolved),
        "num_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "supervised_tokens": supervised_tokens,
        "eos_token_id": 0,
        "vocab_size": 32_004,
        "sequence_length_cap": 2_048,
        "truncation_policy": "fail_closed_no_truncation",
        "input_dtype": "uint32-le",
        "label_dtype": "int32-le",
        "sources": dict(sorted(sources_count.items())),
        "capabilities": dict(sorted(capabilities.items())),
        "special_token_ids": SPECIAL_TOKEN_IDS,
        "special_token_input_counts": {
            token: int(special_inputs[token]) for token in SPECIAL_TOKEN_IDS
        },
        "special_token_label_counts": actual_markers,
        "files": {name: sha256(path) for name, path in final_paths.items()},
    }
    (target / "sft.idx.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def build_release(
    selected_root: Path,
    reasoning_root: Path,
    general_root: Path,
    output: Path,
    *,
    source_dataset: str,
    source_revision: str,
) -> dict[str, Any]:
    reasoning_tokenized = reasoning_root / TOKENIZED_SUBDIR
    general_tokenized = general_root / TOKENIZED_SUBDIR
    partitions = {
        "train": materialize_partition(
            selected_root / "train.jsonl",
            [(reasoning_root / "train.jsonl", reasoning_tokenized / "train")],
            output / "train",
        ),
        "eval": materialize_partition(
            selected_root / "eval.jsonl",
            [
                (general_root / "eval.jsonl", general_tokenized / "eval"),
                (reasoning_root / "eval.jsonl", reasoning_tokenized / "eval"),
            ],
            output / "eval",
        ),
    }
    shutil.copy2(reasoning_tokenized / "chat_template.json", output / "chat_template.json")
    tokenizer_output = output / "tokenizer"
    tokenizer_output.mkdir(parents=True, exist_ok=True)
    for filename in TOKENIZER_FILES:
        shutil.copy2(reasoning_tokenized / "tokenizer" / filename, tokenizer_output / filename)
    raw_manifest = json.loads((selected_root / "manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 2,
        "format": SHARD_FORMAT_V2,
        "quality_status": "passed",
        "release_quality": {
            "ready": True,
            "raw_quality_gate": "passed",
            "token_truncation": False,
            "tokenization": "filtered_from_audited_32004_shards",
        },
        "source_dataset": source_dataset,
        "source_revision": source_revision,
        "source_manifest_sha256": sha256(selected_root / "manifest.json"),
        "source_train_sha256": raw_manifest["train_sha256"],
        "source_eval_sha256": raw_manifest["eval_sha256"],
        "actual_unique_formatted_tokens": int(partitions["train"]["num_tokens"]),
        "tokenizer_vocab_size": 32_004,
        "special_token_ids": SPECIAL_TOKEN_IDS,
        "tokenizer_sha256": sha256(tokenizer_output / "tokenizer.json"),
        "chat_template_id": "complexity-chat-v3-32004",
        "assistant_supervision": FINAL_ASSISTANT_SUPERVISION,
        "history_assistant_turns": MASKED_ASSISTANT_HISTORY,
        "sequence_length_cap": 2_048,
        "partitions": partitions,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-root", type=Path, required=True)
    parser.add_argument("--reasoning-root", type=Path, required=True)
    parser.add_argument("--general-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-dataset",
        default="AETHORIA-AI/TR-HASH-MoE-200M-Unified-SFT-v3-32004-500K",
    )
    parser.add_argument("--source-revision", default="local")
    args = parser.parse_args()
    manifest = build_release(
        args.selected_root,
        args.reasoning_root,
        args.general_root,
        args.output,
        source_dataset=args.source_dataset,
        source_revision=args.source_revision,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
