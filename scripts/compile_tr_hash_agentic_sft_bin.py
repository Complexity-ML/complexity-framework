#!/usr/bin/env python3
"""Compile canonical Agentic SFT JSONL into reusable native token shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from complexity.inference.chat_template import (
    NATIVE_AGENTIC_FULL_TRAJECTORY_PROJECTION,
    agentic_chat_template,
)
from complexity.tokenizer import Tokenizer
from complexity.training.sft_shard import LEGACY_ALL_ASSISTANT_SUPERVISION
from scripts.sft_500m_32k_tr import SFTBinDataset, encode_sft_example

FORMAT = "complexity-sft-token-shard-v2"
TOKENIZER_REPOSITORY = "AETHORIA-AI/TR-HASH-Tokenizer-32K-Agentic"
TOKENIZER_REVISION = "2fcbc2c5359ded0244ca14531f1b3806eebac55e"
EXPECTED_TOKEN_IDS = {
    "<|begin|>": 0,
    "<|end|>": 1,
    "<|pad|>": 2,
    "<|unk|>": 3,
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


def encode_agentic_trajectory(
    tokenizer: Tokenizer,
    record: dict[str, Any],
    seq_len: int,
) -> dict[str, torch.Tensor]:
    """Encode one native trajectory while supervising every assistant action."""

    trajectory = record.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError("trajectory must be a non-empty list")
    if trajectory[-1].get("role") != "assistant":
        raise ValueError("trajectory must end with an assistant action")

    token_ids: list[int] = []
    supervised: list[bool] = []

    def append(text: str, *, train: bool) -> None:
        ids = tokenizer.encode(text, add_special_tokens=False)
        token_ids.extend(ids)
        supervised.extend([train] * len(ids))

    assistant_actions = 0
    for index, message in enumerate(trajectory):
        if not isinstance(message, dict):
            raise ValueError(f"trajectory message {index} is not an object")
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if not content:
            raise ValueError(f"trajectory message {index} has empty content")
        if role == "system":
            append(f"<|system|>{content}<|end_of_turn|>", train=False)
        elif role == "user":
            append(f"<|user|>{content}<|end_of_turn|>", train=False)
        elif role == "tool":
            append(
                f"<|tool_result_start|>{content}<|tool_result_end|><|end_of_turn|>",
                train=False,
            )
        elif role == "assistant":
            if not content.endswith("<|end_of_turn|>"):
                raise ValueError(f"assistant action {index} lacks end_of_turn")
            if "<|assistant|>" in content:
                raise ValueError(f"assistant action {index} contains an assistant prefix")
            append("<|assistant|>", train=False)
            append(content, train=True)
            assistant_actions += 1
        else:
            raise ValueError(f"unsupported trajectory role: {role}")

    if assistant_actions < 1:
        raise ValueError("trajectory contains no assistant action")
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        token_ids.append(int(eos_id))
        supervised.append(True)
    if len(token_ids) < 2:
        raise ValueError("trajectory contains fewer than two tokens")
    if len(token_ids) > seq_len + 1:
        raise ValueError(
            f"trajectory exceeds context without truncation: {len(token_ids)} > {seq_len + 1}"
        )

    input_ids = token_ids[:-1]
    labels = [
        token_ids[index + 1] if supervised[index + 1] else -100 for index in range(len(input_ids))
    ]
    if not any(label != -100 for label in labels):
        raise ValueError("trajectory contains no supervised assistant tokens")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_id if eos_id is not None else 0
    padding = seq_len - len(input_ids)
    input_ids.extend([int(pad_id)] * padding)
    labels.extend([-100] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compile_partition(
    source: Path,
    target: Path,
    *,
    tokenizer: Tokenizer,
    seq_len: int,
    chat_template: dict[str, Any],
) -> dict[str, Any]:
    target.mkdir(parents=True, exist_ok=True)
    inputs_path = target / "input_ids.bin"
    labels_path = target / "labels.bin"
    examples_path = target / "examples.jsonl"
    counts: Counter[str] = Counter()
    offset = 0
    contains_trajectories = False

    with (
        source.open(encoding="utf-8") as rows,
        inputs_path.open("wb") as input_file,
        labels_path.open("wb") as label_file,
        examples_path.open("w", encoding="utf-8") as example_file,
    ):
        for row_index, line in enumerate(rows):
            record = json.loads(line)
            if "trajectory" in record:
                encoded = encode_agentic_trajectory(tokenizer, record, seq_len)
                contains_trajectories = True
            else:
                encoded = encode_sft_example(
                    tokenizer,
                    record,
                    seq_len,
                    min_completion_tokens=1,
                    chat_template=chat_template,
                )
            input_ids = encoded["input_ids"].numpy().astype("<u4", copy=False)
            labels = encoded["labels"].numpy().astype("<i4", copy=False)
            supervised_positions = np.flatnonzero(labels != -100)
            if not len(supervised_positions):
                raise ValueError(f"row {row_index} has no supervised tokens")
            num_tokens = int(supervised_positions[-1]) + 1
            expected = int(record.get("token_count", num_tokens + 1)) - 1
            if num_tokens != expected:
                raise ValueError(
                    f"row {row_index} changed length during compilation: "
                    f"expected={expected}, encoded={num_tokens}"
                )
            input_ids = input_ids[:num_tokens]
            labels = labels[:num_tokens]
            if input_ids.max(initial=0) >= tokenizer.vocab_size:
                raise ValueError(f"row {row_index} contains an out-of-vocabulary ID")
            input_file.write(input_ids.tobytes())
            label_file.write(labels.tobytes())
            supervised = int(np.count_nonzero(labels != -100))
            category = str(record.get("category", "unknown"))
            metadata = {
                "example_id": str(record.get("source_id", f"row-{row_index}")),
                "task": category,
                "source": str(record.get("source_dataset", "unknown")),
                "offset": offset,
                "num_tokens": num_tokens,
                "supervised_tokens": supervised,
            }
            example_file.write(
                json.dumps(metadata, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            offset += num_tokens
            counts[category] += 1
            counts["examples"] += 1
            counts["supervised_tokens"] += supervised
            if counts["examples"] % 10_000 == 0:
                print(
                    f"{source.name}: {counts['examples']:,} examples, {offset:,} tokens",
                    flush=True,
                )

    index = {
        "format": FORMAT,
        "assistant_supervision": (
            LEGACY_ALL_ASSISTANT_SUPERVISION if contains_trajectories else "final_assistant_only"
        ),
        "history_assistant_turns": ("supervised" if contains_trajectories else "masked_context"),
        "chat_template_id": chat_template["id"],
        "examples": counts["examples"],
        "num_tokens": offset,
        "supervised_tokens": counts["supervised_tokens"],
        "categories": {
            key: value
            for key, value in sorted(counts.items())
            if key not in {"examples", "supervised_tokens"}
        },
        "seq_len": seq_len,
        "vocab_size": tokenizer.vocab_size,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "tokenizer": {
            "repository": TOKENIZER_REPOSITORY,
            "revision": TOKENIZER_REVISION,
        },
        "source_jsonl": source.name,
        "source_jsonl_sha256": sha256_file(source),
        "files": {
            "input_ids.bin": sha256_file(inputs_path),
            "labels.bin": sha256_file(labels_path),
            "examples.jsonl": sha256_file(examples_path),
        },
        "no_truncation": True,
        "contains_full_trajectories": contains_trajectories,
    }
    (target / "sft.idx.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=2048)
    args = parser.parse_args()

    tokenizer = Tokenizer.load(str(args.tokenizer))
    if tokenizer.vocab_size != 32_000:
        raise ValueError(f"expected vocab 32000, got {tokenizer.vocab_size}")
    observed_token_ids = {
        token: tokenizer._tokenizer.token_to_id(token) for token in EXPECTED_TOKEN_IDS
    }
    if observed_token_ids != EXPECTED_TOKEN_IDS:
        raise ValueError(
            f"tokenizer does not match the pinned Agentic marker IDs: {observed_token_ids}"
        )
    dataset_manifest_path = args.dataset_dir / "dataset_info.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if dataset_manifest.get("quality_status") != "passed":
        raise ValueError("dataset manifest has not passed its quality gates")
    for split, filename in (("train", "train.jsonl"), ("validation", "eval.jsonl")):
        source = args.dataset_dir / filename
        expected_sha = dataset_manifest["artifacts"][split]["sha256"]
        observed_sha = sha256_file(source)
        if observed_sha != expected_sha:
            raise ValueError(f"{filename} checksum mismatch: {observed_sha} != {expected_sha}")
    eos_token = tokenizer._tokenizer.id_to_token(tokenizer.eos_token_id)
    chat_template = agentic_chat_template(eos_token=eos_token)
    args.output.mkdir(parents=True, exist_ok=True)
    partitions = {
        "train": compile_partition(
            args.dataset_dir / "train.jsonl",
            args.output / "train",
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            chat_template=chat_template,
        ),
        "eval": compile_partition(
            args.dataset_dir / "eval.jsonl",
            args.output / "eval",
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            chat_template=chat_template,
        ),
    }
    if any(partition["contains_full_trajectories"] for partition in partitions.values()):
        chat_template["training_projection"] = NATIVE_AGENTIC_FULL_TRAJECTORY_PROJECTION
    (args.output / "chat_template.json").write_text(
        json.dumps(chat_template, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "format": "tr-hash-agentic-sft-tokenized-v1",
        "quality_status": "passed",
        "release_quality": {"ready": True},
        "tokenizer": {
            "repository": TOKENIZER_REPOSITORY,
            "revision": TOKENIZER_REVISION,
        },
        "seq_len": args.seq_len,
        "partitions": partitions,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Exercise the exact reader used by the training process before publishing.
    train = SFTBinDataset(args.output, args.seq_len, 1729, 0, 1, repeat=False)
    evaluation = SFTBinDataset(args.output / "eval", args.seq_len, 1729, 0, 1, repeat=False)
    if (
        len(train.examples) != partitions["train"]["examples"]
        or len(evaluation.examples) != partitions["eval"]["examples"]
    ):
        raise RuntimeError("compiled shard count mismatch")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
