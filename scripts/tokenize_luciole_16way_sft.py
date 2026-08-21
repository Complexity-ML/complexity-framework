#!/usr/bin/env python3
"""Materialize the Luciole 16-way JSONL release as reusable SFT token shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from complexity.inference.chat_template import default_chat_template
from complexity.tokenizer import Tokenizer
from complexity.training.sft_shard import (
    FINAL_ASSISTANT_SUPERVISION,
    MASKED_ASSISTANT_HISTORY,
    SHARD_FORMAT_V2,
)
from scripts.sft_500m_32k_tr import encode_sft_example

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


def materialize_partition(
    source: Path,
    target: Path,
    *,
    tokenizer: Tokenizer,
    seq_len: int,
    min_completion_tokens: int,
) -> dict[str, Any]:
    target.mkdir(parents=True, exist_ok=True)
    input_path = target / "input_ids.bin"
    label_path = target / "labels.bin"
    examples_path = target / "examples.jsonl"
    input_partial = input_path.with_suffix(".bin.partial")
    label_partial = label_path.with_suffix(".bin.partial")
    examples_partial = examples_path.with_suffix(".jsonl.partial")

    examples = 0
    num_tokens = 0
    supervised_tokens = 0
    with (
        source.open(encoding="utf-8") as source_handle,
        input_partial.open("wb") as input_handle,
        label_partial.open("wb") as label_handle,
        examples_partial.open("w", encoding="utf-8") as examples_handle,
    ):
        for line_number, line in enumerate(tqdm(source_handle, desc=source.stem), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            encoded = encode_sft_example(
                tokenizer,
                record,
                seq_len,
                min_completion_tokens,
                default_chat_template(),
            )
            input_ids = encoded["input_ids"].numpy()
            labels = encoded["labels"].numpy()
            active = np.flatnonzero(labels != -100)
            if not active.size:
                raise ValueError(f"{source}:{line_number} has no supervised assistant tokens")
            length = int(active[-1]) + 1
            visible_inputs = np.asarray(input_ids[:length], dtype="<u4")
            visible_labels = np.asarray(labels[:length], dtype="<i4")
            visible_inputs.tofile(input_handle)
            visible_labels.tofile(label_handle)
            supervised = int(np.count_nonzero(visible_labels != -100))
            example_id = f"{source.stem}-{examples:06d}"
            source_name = str(record.get("source", "unknown"))
            examples_handle.write(
                json.dumps(
                    {
                        "example_id": example_id,
                        "task": source_name,
                        "source": source_name,
                        "offset": num_tokens,
                        "num_tokens": length,
                        "supervised_tokens": supervised,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            examples += 1
            num_tokens += length
            supervised_tokens += supervised

    input_partial.replace(input_path)
    label_partial.replace(label_path)
    examples_partial.replace(examples_path)
    metadata = {
        "format": SHARD_FORMAT_V2,
        "assistant_supervision": FINAL_ASSISTANT_SUPERVISION,
        "history_assistant_turns": MASKED_ASSISTANT_HISTORY,
        "chat_template_id": default_chat_template()["id"],
        "examples": examples,
        "num_tokens": num_tokens,
        "supervised_tokens": supervised_tokens,
        "eos_token_id": int(tokenizer.eos_token_id),
        "sequence_length_cap": seq_len,
        "input_dtype": "uint32-le",
        "label_dtype": "int32-le",
        "files": {
            "input_ids.bin": sha256(input_path),
            "labels.bin": sha256(label_path),
            "examples.jsonl": sha256(examples_path),
        },
    }
    (target / "sft.idx.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--min-completion-tokens", type=int, default=32)
    parser.add_argument("--source-revision", default="main")
    parser.add_argument("--upload-repo", default=None)
    args = parser.parse_args()

    tokenizer = Tokenizer.load(str(args.tokenizer))
    args.output.mkdir(parents=True, exist_ok=True)
    partition_metadata = {
        split: materialize_partition(
            args.source / filename,
            args.output / split,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            min_completion_tokens=args.min_completion_tokens,
        )
        for split, filename in (("train", "train.jsonl"), ("eval", "eval.jsonl"))
    }

    template = default_chat_template()
    (args.output / "chat_template.json").write_text(
        json.dumps(template, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tokenizer_output = args.output / "tokenizer"
    tokenizer_output.mkdir(exist_ok=True)
    copied_tokenizer_files = []
    for name in TOKENIZER_FILES:
        source_file = args.tokenizer / name
        if source_file.is_file():
            shutil.copy2(source_file, tokenizer_output / name)
            copied_tokenizer_files.append(name)

    source_manifest = json.loads((args.source / "manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "format": SHARD_FORMAT_V2,
        "quality_status": "passed",
        "release_quality": {"ready": True},
        "source_dataset": "AETHORIA-AI/luciole-16way-sft-209k",
        "source_revision": args.source_revision,
        "source_manifest_sha256": sha256(args.source / "manifest.json"),
        "source_train_sha256": source_manifest["train_sha256"],
        "source_eval_sha256": source_manifest["eval_sha256"],
        "tokenizer_files": copied_tokenizer_files,
        "tokenizer_sha256": sha256(args.tokenizer / "tokenizer.json"),
        "chat_template_id": template["id"],
        "assistant_supervision": FINAL_ASSISTANT_SUPERVISION,
        "history_assistant_turns": MASKED_ASSISTANT_HISTORY,
        "sequence_length_cap": args.seq_len,
        "partitions": partition_metadata,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.upload_repo:
        from huggingface_hub import HfApi

        HfApi().upload_folder(
            repo_id=args.upload_repo,
            repo_type="dataset",
            folder_path=args.output,
            path_in_repo="tokenized/tr-hash-32k-v1",
            commit_message="Add reusable TR-HASH 32k Luciole SFT token shards",
            ignore_patterns=["*.partial"],
        )
        print(f"Uploaded tokenized view to {args.upload_repo}", flush=True)

    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
