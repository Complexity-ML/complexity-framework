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

from complexity.inference.chat_template import (
    agentic_chat_template,
)
from complexity.tokenizer import Tokenizer
from scripts.sft_500m_32k_tr import SFTBinDataset, encode_sft_example

FORMAT = "complexity-sft-token-shard-v2"
TOKENIZER_REPOSITORY = "AETHORIA-AI/TR-HASH-Tokenizer-32K-Agentic"
TOKENIZER_REVISION = "2fcbc2c5359ded0244ca14531f1b3806eebac55e"


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

    with (
        source.open(encoding="utf-8") as rows,
        inputs_path.open("wb") as input_file,
        labels_path.open("wb") as label_file,
        examples_path.open("w", encoding="utf-8") as example_file,
    ):
        for row_index, line in enumerate(rows):
            record = json.loads(line)
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
        "assistant_supervision": "final_assistant_only",
        "history_assistant_turns": "masked_context",
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
    eos_token = tokenizer._tokenizer.id_to_token(tokenizer.eos_token_id)
    chat_template = agentic_chat_template(eos_token=eos_token)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "chat_template.json").write_text(
        json.dumps(chat_template, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

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
