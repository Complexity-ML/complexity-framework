#!/usr/bin/env python3
"""Project GSM8K train into native Agentic JSONL and token shards.

The GSM8K test split is read only to prove split separation. Its questions and
answers are never written into the training artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from complexity.inference.chat_template import agentic_chat_template, render_inference_prompt
from complexity.tokenizer import Tokenizer

DATASET_REPOSITORY = "openai/gsm8k"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
TOKENIZER_REPOSITORY = "AETHORIA-AI/TR-HASH-Tokenizer-32K-Agentic"
TOKENIZER_REVISION = "2fcbc2c5359ded0244ca14531f1b3806eebac55e"
TARGET_REPOSITORY = "AETHORIA-AI/GSM8K-TR-HASH-Agentic-32K"
EXPECTED_TRAIN = 7_473
EXPECTED_TEST = 1_319
FORMAT = "complexity-sft-token-shard-v2"
ANNOTATION = re.compile(r"\s*<<[^<>\n]+>>\s*")
FINAL_MARKER = re.compile(r"(?:^|\n)####\s*([^\n]+)\s*$")
WHITESPACE = re.compile(r"\s+")
AGENTIC_MARKERS = {
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_question(value: str) -> str:
    return WHITESPACE.sub(" ", value.casefold()).strip()


def question_digest(value: str) -> str:
    return hashlib.sha256(normalized_question(value).encode()).hexdigest()


def validate_tokenizer(tokenizer: Tokenizer) -> None:
    if tokenizer.vocab_size != 32_000:
        raise ValueError(f"expected Agentic vocab 32000, got {tokenizer.vocab_size}")
    if tokenizer.bos_token_id != 0 or tokenizer.eos_token_id != 1 or tokenizer.pad_token_id != 2:
        raise ValueError(
            "Agentic tokenizer requires bos=0, eos=1 and pad=2; got "
            f"{tokenizer.bos_token_id}, {tokenizer.eos_token_id}, {tokenizer.pad_token_id}"
        )
    for marker, expected in AGENTIC_MARKERS.items():
        encoded = tokenizer.encode(marker, add_special_tokens=False)
        if encoded != [expected]:
            raise ValueError(
                f"Agentic marker mismatch for {marker}: {encoded}, expected {[expected]}"
            )


def clean_answer(answer: str) -> str:
    match = FINAL_MARKER.search(answer)
    if match is None:
        raise ValueError("GSM8K answer has no final #### marker")
    final = match.group(1).strip()
    reasoning = FINAL_MARKER.sub("", answer)
    reasoning = ANNOTATION.sub(" ", reasoning).strip()
    reasoning = "\n".join(
        WHITESPACE.sub(" ", line).strip() for line in reasoning.splitlines() if line.strip()
    )
    if not reasoning:
        raise ValueError("GSM8K answer has no reasoning before final marker")
    if "<<" in reasoning or ">>" in reasoning or "####" in reasoning:
        raise ValueError("GSM8K verifier annotation survived normalization")
    return f"{reasoning}\nFinal answer: {final}"


def encode_record(
    tokenizer: Tokenizer,
    question: str,
    answer: str,
    *,
    sequence_length: int,
) -> tuple[str, str, np.ndarray, np.ndarray, int]:
    eos_token = tokenizer._tokenizer.id_to_token(tokenizer.eos_token_id)
    template = agentic_chat_template(eos_token=eos_token)
    prompt = render_inference_prompt(question, template)
    completion = clean_answer(answer)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    full = [*prompt_ids, *completion_ids, tokenizer.eos_token_id]
    if len(full) > sequence_length + 1:
        raise ValueError(f"GSM8K example exceeds {sequence_length + 1} tokens: {len(full)}")
    inputs = np.asarray(full[:-1], dtype="<u4")
    labels = np.asarray(full[1:], dtype="<i4")
    labels[: max(0, len(prompt_ids) - 1)] = -100
    supervised = int(np.count_nonzero(labels != -100))
    if supervised != len(completion_ids) + 1:
        raise RuntimeError("assistant-only label count is inconsistent")
    return prompt, completion, inputs, labels, supervised


def load_rows(path: Path) -> list[dict[str, str]]:
    rows = pq.read_table(path, columns=["question", "answer"]).to_pylist()
    return [{"question": str(row["question"]), "answer": str(row["answer"])} for row in rows]


def write_artifact(
    *,
    train_parquet: Path,
    test_parquet: Path,
    tokenizer_path: Path,
    output: Path,
    sequence_length: int,
    expected_train: int = EXPECTED_TRAIN,
    expected_test: int = EXPECTED_TEST,
) -> dict[str, Any]:
    tokenizer = Tokenizer.load(str(tokenizer_path))
    validate_tokenizer(tokenizer)
    train_rows = load_rows(train_parquet)
    test_rows = load_rows(test_parquet)
    if len(train_rows) != expected_train or len(test_rows) != expected_test:
        raise ValueError(f"unexpected GSM8K sizes: train={len(train_rows)}, test={len(test_rows)}")
    train_questions = {question_digest(row["question"]) for row in train_rows}
    test_questions = {question_digest(row["question"]) for row in test_rows}
    overlap = train_questions & test_questions
    if overlap:
        raise ValueError(f"GSM8K train/test exact-question overlap: {len(overlap)}")
    if len(train_questions) != len(train_rows):
        raise ValueError("GSM8K train contains duplicate normalized questions")

    train_dir = output / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output / "train.jsonl"
    inputs_path = train_dir / "input_ids.bin"
    labels_path = train_dir / "labels.bin"
    examples_path = train_dir / "examples.jsonl"
    offset = 0
    supervised_tokens = 0
    total_full_tokens = 0
    with (
        jsonl_path.open("w", encoding="utf-8") as jsonl_file,
        inputs_path.open("wb") as inputs_file,
        labels_path.open("wb") as labels_file,
        examples_path.open("w", encoding="utf-8") as examples_file,
    ):
        for index, row in enumerate(train_rows):
            prompt, completion, inputs, labels, supervised = encode_record(
                tokenizer,
                row["question"],
                row["answer"],
                sequence_length=sequence_length,
            )
            source_id = f"gsm8k-train-{index:05d}"
            record = {
                "source_id": source_id,
                "source_dataset": DATASET_REPOSITORY,
                "source_revision": DATASET_REVISION,
                "source_split": "train",
                "category": "supervised_math_reasoning",
                "prompt": prompt,
                "completion": completion,
                "token_count": len(inputs) + 1,
            }
            jsonl_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            inputs_file.write(inputs.tobytes())
            labels_file.write(labels.tobytes())
            examples_file.write(
                json.dumps(
                    {
                        "example_id": source_id,
                        "task": "supervised_math_reasoning",
                        "source": DATASET_REPOSITORY,
                        "source_split": "train",
                        "offset": offset,
                        "num_tokens": len(inputs),
                        "supervised_tokens": supervised,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            offset += len(inputs)
            supervised_tokens += supervised
            total_full_tokens += len(inputs) + 1

    eos_token = tokenizer._tokenizer.id_to_token(tokenizer.eos_token_id)
    template = agentic_chat_template(eos_token=eos_token)
    (output / "chat_template.json").write_text(
        json.dumps(template, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    index = {
        "format": FORMAT,
        "assistant_supervision": "final_assistant_only",
        "history_assistant_turns": "masked_context",
        "chat_template_id": template["id"],
        "examples": len(train_rows),
        "num_tokens": offset,
        "supervised_tokens": supervised_tokens,
        "categories": {"supervised_math_reasoning": len(train_rows)},
        "seq_len": sequence_length,
        "vocab_size": tokenizer.vocab_size,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "source_split": "train",
        "test_split_present": False,
        "tokenizer": {
            "repository": TOKENIZER_REPOSITORY,
            "revision": TOKENIZER_REVISION,
        },
        "files": {
            "input_ids.bin": sha256_file(inputs_path),
            "labels.bin": sha256_file(labels_path),
            "examples.jsonl": sha256_file(examples_path),
        },
        "no_truncation": True,
    }
    (train_dir / "sft.idx.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    info = {
        "schema_version": 1,
        "name": "gsm8k-tr-hash-agentic-32k",
        "target_repository": TARGET_REPOSITORY,
        "publication_enabled": False,
        "source": {
            "repository": DATASET_REPOSITORY,
            "config": "main",
            "revision": DATASET_REVISION,
            "train_parquet_sha256": sha256_file(train_parquet),
            "test_parquet_sha256": sha256_file(test_parquet),
        },
        "splits": {
            "train": {"examples": len(train_rows), "full_tokens": total_full_tokens},
            "test": {
                "examples": len(test_rows),
                "usage": "evaluation_only",
                "payload_in_artifact": False,
            },
        },
        "train_test_normalized_question_overlap": 0,
        "tokenizer": index["tokenizer"],
        "chat_template": template["id"],
        "assistant_only_loss": True,
        "no_truncation": True,
        "training_index_sha256": sha256_file(train_dir / "sft.idx.json"),
        "training_jsonl_sha256": sha256_file(jsonl_path),
    }
    (output / "dataset_info.json").write_text(
        json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "source_assets.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "reuse_policy": "pass_existing_paths_before_downloading",
                "assets": [
                    {
                        "environment_variable": "GSM8K_TRAIN_PARQUET",
                        "repository": DATASET_REPOSITORY,
                        "revision": DATASET_REVISION,
                        "path_in_repository": "main/train-00000-of-00001.parquet",
                        "sha256": info["source"]["train_parquet_sha256"],
                    },
                    {
                        "environment_variable": "GSM8K_TEST_PARQUET",
                        "repository": DATASET_REPOSITORY,
                        "revision": DATASET_REVISION,
                        "path_in_repository": "main/test-00000-of-00001.parquet",
                        "sha256": info["source"]["test_parquet_sha256"],
                        "usage": "evaluation_only",
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / ".gitattributes").write_text(
        "*.bin filter=lfs diff=lfs merge=lfs -text\n*.jsonl filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "---\nlicense: mit\ntask_categories:\n- text-generation\n---\n\n"
        "# GSM8K TR-HASH Agentic 32K\n\n"
        "Unpublished experimental projection of the pinned GSM8K **train** split for "
        "TR-HASH Agentic SFT. The GSM8K test split is excluded from every training "
        "file and reserved for supervised-checkpoint evaluation.\n\n"
        f"- Source: `{DATASET_REPOSITORY}` @ `{DATASET_REVISION}`\n"
        f"- Tokenizer: `{TOKENIZER_REPOSITORY}` @ `{TOKENIZER_REVISION}`\n"
        f"- Train examples: {len(train_rows):,}\n"
        "- Training epochs are a model-run choice; this artifact contains one copy "
        "of each source-train example.\n",
        encoding="utf-8",
    )
    artifact_files = (
        ".gitattributes",
        "README.md",
        "chat_template.json",
        "dataset_info.json",
        "source_assets.json",
        "train.jsonl",
        "train/examples.jsonl",
        "train/input_ids.bin",
        "train/labels.bin",
        "train/sft.idx.json",
    )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "tr-hash-agentic-gsm8k-sft-dataset-v1",
                "quality_status": "passed",
                "publication_ready": False,
                "publication_blocker": "model training and epoch 1/2/3 test evaluation pending",
                "target_repository": TARGET_REPOSITORY,
                "files": {name: sha256_file(output / name) for name in artifact_files},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--test-parquet", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=2048)
    args = parser.parse_args()
    info = write_artifact(
        train_parquet=args.train_parquet,
        test_parquet=args.test_parquet,
        tokenizer_path=args.tokenizer,
        output=args.output,
        sequence_length=args.sequence_length,
    )
    print(json.dumps(info, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
