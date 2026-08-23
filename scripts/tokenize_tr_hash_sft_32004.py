#!/usr/bin/env python3
"""Encode audited SFT text into reusable TR-HASH 32,004-token shards.

The source compiler already rejects rows above the model context. This encoder
checks that invariant again and fails instead of truncating either the prompt
or the supervised completion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from complexity.inference.chat_template import (
    align_chat_template_eos,
    huggingface_chat_template,
    reasoning_chat_template_32004,
)
from complexity.tokenizer import Tokenizer
from complexity.training.sft_shard import (
    FINAL_ASSISTANT_SUPERVISION,
    MASKED_ASSISTANT_HISTORY,
    SHARD_FORMAT_V2,
)

try:
    from scripts.sft_500m_32k_tr import format_record
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from sft_500m_32k_tr import format_record


TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
)
REASONING_SPECIAL_TOKEN_IDS = {
    "<|think_start|>": 32_000,
    "<|think_end|>": 32_001,
    "<|final_start|>": 32_002,
    "<|final_end|>": 32_003,
}


def validate_reasoning_tokenizer(tokenizer: Tokenizer) -> None:
    if len(tokenizer) != 32_004:
        raise ValueError(f"expected canonical TR-HASH vocab 32,004, got {len(tokenizer):,}")
    for token, expected_id in REASONING_SPECIAL_TOKEN_IDS.items():
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if encoded != [expected_id]:
            raise ValueError(f"{token} encoded as {encoded}, expected [{expected_id}]")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_aligned_chat_template(tokenizer: Tokenizer) -> dict[str, Any]:
    eos_text = tokenizer.decode([tokenizer.eos_token_id], skip_special_tokens=False)
    if tokenizer.encode(eos_text, add_special_tokens=False) != [tokenizer.eos_token_id]:
        raise ValueError("tokenizer EOS text does not round-trip to its EOS token ID")
    return align_chat_template_eos(reasoning_chat_template_32004(), eos_token=eos_text)


def _raw_token_ids(
    tokenizer: Tokenizer,
    record: dict[str, Any],
    chat_template: dict[str, Any],
) -> tuple[list[int], list[int]]:
    prompt, completion = format_record(record, chat_template)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        completion_ids.append(tokenizer.eos_token_id)
    return prompt_ids, completion_ids


def encode_complete_example(
    tokenizer: Tokenizer,
    record: dict[str, Any],
    *,
    sequence_length: int,
    chat_template: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    chat_template = chat_template or tokenizer_aligned_chat_template(tokenizer)
    prompt_ids, completion_ids = _raw_token_ids(tokenizer, record, chat_template)
    prompt_tokens = len(prompt_ids)
    completion_tokens = len(completion_ids)
    if prompt_tokens + completion_tokens > sequence_length + 1:
        raise ValueError(
            "example would be truncated: "
            f"prompt={prompt_tokens} completion={completion_tokens} "
            f"capacity={sequence_length + 1}"
        )
    full = prompt_ids + completion_ids
    if len(full) < 2:
        raise ValueError("example has fewer than two encoded tokens")
    visible_inputs = np.asarray(full[:-1], dtype="<u4")
    visible_labels = np.asarray(full[1:], dtype="<i4")
    prompt_targets = max(0, min(len(visible_labels), prompt_tokens - 1))
    visible_labels[:prompt_targets] = -100
    if not np.any(visible_labels != -100):
        raise ValueError("example has no supervised assistant tokens")
    supervised = int(np.count_nonzero(visible_labels != -100))
    if supervised != completion_tokens:
        raise ValueError(
            "supervised completion changed during encoding: "
            f"expected={completion_tokens} actual={supervised}"
        )
    if visible_inputs.size and int(visible_inputs.max()) >= len(tokenizer):
        raise ValueError("encoded token id exceeds tokenizer vocabulary")
    return visible_inputs, visible_labels, prompt_tokens, completion_tokens


def materialize_partition(
    source: Path,
    target: Path,
    *,
    tokenizer: Tokenizer,
    sequence_length: int,
    chat_template: dict[str, Any] | None = None,
    resume: bool = False,
    checkpoint_every: int = 10_000,
) -> dict[str, Any]:
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "input_ids.bin": target / "input_ids.bin",
        "labels.bin": target / "labels.bin",
        "examples.jsonl": target / "examples.jsonl",
    }
    partials = {name: path.with_suffix(path.suffix + ".partial") for name, path in paths.items()}
    state_path = target / ".tokenize_state.json"
    metadata_path = target / "sft.idx.json"
    source_sha256 = sha256(source)
    if resume and metadata_path.is_file() and all(path.is_file() for path in paths.values()):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("source_sha256") != source_sha256:
            raise ValueError(f"completed tokenized partition source changed: {source}")
        for name, expected in metadata["files"].items():
            if sha256(paths[name]) != expected:
                raise ValueError(f"completed tokenized partition hash mismatch: {paths[name]}")
        return metadata

    examples = total_tokens = supervised_tokens = prompt_tokens = 0
    sources: Counter[str] = Counter()
    capabilities: Counter[str] = Counter()
    special_input_counts: Counter[str] = Counter()
    special_label_counts: Counter[str] = Counter()
    source_lines_consumed = 0
    output_mode = "wb"
    chat_template = chat_template or tokenizer_aligned_chat_template(tokenizer)

    if resume and state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("source_sha256") != source_sha256:
            raise ValueError(f"tokenization resume source changed: {source}")
        if int(state.get("sequence_length", 0)) != sequence_length:
            raise ValueError("tokenization resume sequence length changed")
        for name, partial in partials.items():
            if not partial.is_file():
                raise FileNotFoundError(f"missing tokenization partial: {partial}")
            with partial.open("r+b") as handle:
                handle.truncate(int(state["partial_bytes"][name]))
        examples = int(state["examples"])
        total_tokens = int(state["total_tokens"])
        supervised_tokens = int(state["supervised_tokens"])
        prompt_tokens = int(state["prompt_tokens"])
        source_lines_consumed = int(state["source_lines_consumed"])
        sources.update(state.get("sources", {}))
        capabilities.update(state.get("capabilities", {}))
        special_input_counts.update(state.get("special_input_counts", {}))
        special_label_counts.update(state.get("special_label_counts", {}))
        output_mode = "ab"
        print(
            f"[tokenize resume] {source.name}: lines={source_lines_consumed:,} "
            f"examples={examples:,} tokens={total_tokens:,}",
            flush=True,
        )

    def checkpoint(input_handle, label_handle, index_handle, line_number: int) -> None:
        for handle in (input_handle, label_handle, index_handle):
            handle.flush()
            os.fsync(handle.fileno())
        payload = {
            "schema_version": 1,
            "source_sha256": source_sha256,
            "sequence_length": sequence_length,
            "source_lines_consumed": line_number,
            "examples": examples,
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "supervised_tokens": supervised_tokens,
            "sources": dict(sources),
            "capabilities": dict(capabilities),
            "special_input_counts": dict(special_input_counts),
            "special_label_counts": dict(special_label_counts),
            "partial_bytes": {
                "input_ids.bin": input_handle.tell(),
                "labels.bin": label_handle.tell(),
                "examples.jsonl": index_handle.tell(),
            },
        }
        temporary = state_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(state_path)

    with (
        source.open(encoding="utf-8") as source_handle,
        partials["input_ids.bin"].open(output_mode) as input_handle,
        partials["labels.bin"].open(output_mode) as label_handle,
        partials["examples.jsonl"].open(output_mode) as index_handle,
    ):
        iterator = enumerate(source_handle, start=1)
        for _ in range(source_lines_consumed):
            next(iterator, None)
        last_line_number = source_lines_consumed
        for line_number, line in tqdm(
            iterator,
            desc=source.stem,
            initial=source_lines_consumed,
        ):
            last_line_number = line_number
            if not line.strip():
                if line_number % checkpoint_every == 0:
                    checkpoint(input_handle, label_handle, index_handle, line_number)
                continue
            record = json.loads(line)
            inputs, labels, prompt_count, completion_count = encode_complete_example(
                tokenizer,
                record,
                sequence_length=sequence_length,
                chat_template=chat_template,
            )
            inputs.tofile(input_handle)
            labels.tofile(label_handle)
            source_name = str(record.get("source", "unknown"))
            capability = str(record.get("capability", "unknown"))
            index_handle.write(
                (
                    json.dumps(
                        {
                            "example_id": f"{source.stem}-{examples:06d}",
                            "task": capability,
                            "source": source_name,
                            "capability": capability,
                            "offset": total_tokens,
                            "num_tokens": int(inputs.size),
                            "prompt_tokens": prompt_count,
                            "supervised_tokens": completion_count,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
            )
            examples += 1
            total_tokens += int(inputs.size)
            prompt_tokens += prompt_count
            supervised_tokens += completion_count
            sources[source_name] += 1
            capabilities[capability] += 1
            for token, token_id in REASONING_SPECIAL_TOKEN_IDS.items():
                special_input_counts[token] += int(np.count_nonzero(inputs == token_id))
                visible = labels[labels != -100]
                special_label_counts[token] += int(np.count_nonzero(visible == token_id))
            if line_number % checkpoint_every == 0:
                checkpoint(input_handle, label_handle, index_handle, line_number)

        checkpoint(input_handle, label_handle, index_handle, last_line_number)

    for name, path in paths.items():
        partials[name].replace(path)
    metadata = {
        "format": SHARD_FORMAT_V2,
        "assistant_supervision": FINAL_ASSISTANT_SUPERVISION,
        "history_assistant_turns": MASKED_ASSISTANT_HISTORY,
        "chat_template_id": chat_template["id"],
        "chat_template_eos_token": chat_template["eos_token"],
        "source_sha256": source_sha256,
        "examples": examples,
        "num_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "supervised_tokens": supervised_tokens,
        "eos_token_id": int(tokenizer.eos_token_id),
        "vocab_size": len(tokenizer),
        "sequence_length_cap": sequence_length,
        "truncation_policy": "fail_closed_no_truncation",
        "input_dtype": "uint32-le",
        "label_dtype": "int32-le",
        "sources": dict(sorted(sources.items())),
        "capabilities": dict(sorted(capabilities.items())),
        "special_token_ids": REASONING_SPECIAL_TOKEN_IDS,
        "special_token_input_counts": {
            token: int(special_input_counts[token]) for token in REASONING_SPECIAL_TOKEN_IDS
        },
        "special_token_label_counts": {
            token: int(special_label_counts[token]) for token in REASONING_SPECIAL_TOKEN_IDS
        },
        "files": {name: sha256(path) for name, path in paths.items()},
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    state_path.unlink(missing_ok=True)
    return metadata


def materialize_release(
    source: Path,
    tokenizer_path: Path,
    output: Path,
    *,
    sequence_length: int = 2048,
    source_dataset: str = "AETHORIA-AI/TR-HASH-MoE-200M-SFT-v3-32004-300K",
    source_revision: str = "main",
    resume: bool = False,
) -> dict[str, Any]:
    raw_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if int(raw_manifest["sequence_length"]) != sequence_length:
        raise ValueError("raw dataset and token-shard context lengths differ")
    tokenizer = Tokenizer.load(str(tokenizer_path))
    validate_reasoning_tokenizer(tokenizer)
    chat_template = tokenizer_aligned_chat_template(tokenizer)
    output.mkdir(parents=True, exist_ok=True)
    partitions = {
        split: materialize_partition(
            source / filename,
            output / split,
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            chat_template=chat_template,
            resume=resume,
        )
        # ``scripts.sft_tr`` recognizes the matched held-out partition as
        # ``eval`` (or ``diagnostic`` for newer dual-eval packages).
        for split, filename in (("train", "train.jsonl"), ("eval", "eval.jsonl"))
    }
    expected_train_tokens = raw_manifest.get("actual_unique_formatted_tokens")
    if expected_train_tokens is not None:
        expected_train_tokens = int(expected_train_tokens)
        actual_train_tokens = int(partitions["train"]["num_tokens"])
        if actual_train_tokens != expected_train_tokens:
            raise ValueError(
                "raw selection and materialized training token counts differ: "
                f"raw={expected_train_tokens} materialized={actual_train_tokens}"
            )
    (output / "chat_template.json").write_text(
        json.dumps(chat_template, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tokenizer_output = output / "tokenizer"
    tokenizer_output.mkdir(exist_ok=True)
    for name in TOKENIZER_FILES:
        shutil.copy2(tokenizer_path / name, tokenizer_output / name)
    (tokenizer_output / "chat_template.jinja").write_text(
        huggingface_chat_template(chat_template), encoding="utf-8"
    )
    manifest = {
        "schema_version": 2,
        "format": SHARD_FORMAT_V2,
        "quality_status": "passed",
        "release_quality": {
            "ready": True,
            "raw_quality_gate": "passed",
            "token_truncation": False,
        },
        "source_dataset": source_dataset,
        "source_revision": source_revision,
        "source_manifest_sha256": sha256(source / "manifest.json"),
        "source_train_sha256": raw_manifest["train_sha256"],
        "source_eval_sha256": raw_manifest["eval_sha256"],
        "nominal_target_unique_formatted_tokens": raw_manifest.get(
            "nominal_target_unique_formatted_tokens"
        ),
        "actual_unique_formatted_tokens": int(partitions["train"]["num_tokens"]),
        "token_quota_overshoot": raw_manifest.get("token_quota_overshoot"),
        "tokenizer_vocab_size": len(tokenizer),
        "special_token_ids": REASONING_SPECIAL_TOKEN_IDS,
        "tokenizer_sha256": sha256(tokenizer_path / "tokenizer.json"),
        "chat_template_id": chat_template["id"],
        "chat_template_eos_token": chat_template["eos_token"],
        "assistant_supervision": FINAL_ASSISTANT_SUPERVISION,
        "history_assistant_turns": MASKED_ASSISTANT_HISTORY,
        "sequence_length_cap": sequence_length,
        "partitions": partitions,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument(
        "--source-dataset",
        default="AETHORIA-AI/TR-HASH-MoE-200M-SFT-v3-32004-300K",
    )
    parser.add_argument("--source-revision", default="main")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume each partition from its last atomic 10K-line checkpoint.",
    )
    args = parser.parse_args()
    manifest = materialize_release(
        args.source,
        args.tokenizer,
        args.output,
        sequence_length=args.seq_len,
        source_dataset=args.source_dataset,
        source_revision=args.source_revision,
        resume=args.resume,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
