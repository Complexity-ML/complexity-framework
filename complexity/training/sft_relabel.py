"""Relabel existing SFT token shards without rebuilding their content."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SHARD_FORMAT_V2 = "complexity-sft-token-shard-v2"
ALL_ASSISTANT_PROJECTION = "naturalize_card_hand_supervise_all_assistant_turns"


def _marker_offsets(tokens: np.ndarray, marker: np.ndarray) -> list[int]:
    if marker.size == 0 or tokens.size < marker.size:
        return []
    candidates = np.flatnonzero(tokens[: tokens.size - marker.size + 1] == marker[0])
    return [
        int(start)
        for start in candidates
        if np.array_equal(tokens[start : start + marker.size], marker)
    ]


def relabel_example(
    input_ids: np.ndarray,
    labels: np.ndarray,
    *,
    assistant_marker: Iterable[int],
    eos_token_id: int,
) -> tuple[np.ndarray, int]:
    """Activate causal labels for every complete assistant span in one example.

    Existing labels are retained for the final response because its terminal
    EOS is a target and therefore is not necessarily present in ``input_ids``.
    Earlier assistant responses are recovered from their marker through the
    EOS token already present in the unchanged token stream.
    """

    tokens = np.asarray(input_ids, dtype=np.int64)
    old = np.asarray(labels, dtype=np.int64)
    if tokens.ndim != 1 or old.ndim != 1 or tokens.shape != old.shape:
        raise ValueError("input_ids and labels must be aligned one-dimensional arrays")
    marker = np.asarray(tuple(assistant_marker), dtype=np.int64)
    starts = _marker_offsets(tokens, marker)
    if not starts:
        raise ValueError("example contains no Assistant marker")

    relabeled = old.copy()
    allowed = np.zeros(tokens.size, dtype=bool)
    for marker_start in starts:
        label_start = marker_start + marker.size - 1
        content_start = marker_start + marker.size
        eos_locations = np.flatnonzero(tokens[content_start:] == int(eos_token_id))
        if eos_locations.size:
            eos_index = content_start + int(eos_locations[0])
            allowed[label_start:eos_index] = True
            relabeled[label_start:eos_index] = tokens[label_start + 1 : eos_index + 1]
        else:
            # The final EOS may live only in labels because the causal input is
            # shifted left by one token. Preserve that already-correct target.
            allowed[label_start:] = True

    active = relabeled != -100
    if not np.any(active):
        raise ValueError("example contains no supervised assistant token")
    if np.any(active & ~allowed):
        raise ValueError("active label found outside an assistant response")
    comparable = active[:-1]
    if np.any(relabeled[:-1][comparable] != tokens[1:][comparable]):
        raise ValueError("active labels are not causally aligned with input_ids")
    if active[-1] and int(relabeled[-1]) != int(eos_token_id):
        raise ValueError("final active label must be the assistant EOS token")
    return relabeled.astype(np.dtype("<i4"), copy=False), int(active.sum())


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _partition_paths(root: Path) -> list[Path]:
    if (root / "sft.idx.json").is_file():
        return [root]
    return sorted(
        path for path in root.iterdir() if (path / "sft.idx.json").is_file()
    )


def _relabel_partition(
    source: Path,
    target: Path,
    *,
    assistant_marker: list[int],
    eos_token_id: int,
    skip_content_verification: bool,
) -> dict[str, int]:
    metadata = json.loads((source / "sft.idx.json").read_text(encoding="utf-8"))
    inputs_path = source / "input_ids.bin"
    labels_path = source / "labels.bin"
    input_ids = np.memmap(inputs_path, mode="r", dtype=np.dtype("<u4"))
    old_labels = np.memmap(labels_path, mode="r", dtype=np.dtype("<i4"))
    if input_ids.shape != old_labels.shape:
        raise ValueError(f"unaligned shard arrays in {source}")

    examples = [
        json.loads(line)
        for line in (source / "examples.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    target.mkdir(parents=True, exist_ok=True)
    _link_or_copy(inputs_path, target / "input_ids.bin")
    new_labels = np.memmap(
        target / "labels.bin",
        mode="w+",
        dtype=np.dtype("<i4"),
        shape=old_labels.shape,
    )

    total_supervised = 0
    changed_labels = 0
    with (target / "examples.jsonl").open("w", encoding="utf-8") as output_index:
        for example in examples:
            start = int(example["offset"])
            length = int(example["num_tokens"])
            end = start + length
            if start < 0 or length <= 0 or end > input_ids.size:
                raise ValueError(f"invalid example bounds in {source}: {example}")
            updated, supervised = relabel_example(
                input_ids[start:end],
                old_labels[start:end],
                assistant_marker=assistant_marker,
                eos_token_id=eos_token_id,
            )
            new_labels[start:end] = updated
            changed_labels += int(np.count_nonzero(updated != old_labels[start:end]))
            total_supervised += supervised
            example["supervised_tokens"] = supervised
            output_index.write(json.dumps(example, ensure_ascii=False) + "\n")
    new_labels.flush()

    for path in source.iterdir():
        if path.is_file() and path.name not in {
            "input_ids.bin",
            "labels.bin",
            "examples.jsonl",
            "sft.idx.json",
        }:
            _link_or_copy(path, target / path.name)

    if not skip_content_verification:
        source_hash = _sha256(inputs_path)
        target_hash = _sha256(target / "input_ids.bin")
        if source_hash != target_hash:
            raise ValueError(f"reused input_ids changed in {target}")
    metadata.update(
        {
            "format": SHARD_FORMAT_V2,
            "parent_format": metadata.get("format"),
            "assistant_supervision": "all_assistant_turns",
            "content_reused": True,
            "content_verification": (
                "skipped_unchanged" if skip_content_verification else "sha256"
            ),
            "alignment_verification": "causal_labels_and_assistant_spans",
            "supervised_tokens": total_supervised,
        }
    )
    (target / "sft.idx.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "examples": len(examples),
        "tokens": int(input_ids.size),
        "supervised_tokens": total_supervised,
        "changed_labels": changed_labels,
    }


def relabel_dataset(
    source_root: str | Path,
    output_root: str | Path,
    *,
    tokenizer: Any,
    skip_content_verification: bool = False,
) -> dict[str, dict[str, int]]:
    """Create a new all-assistant-label dataset while reusing source content."""

    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    partitions = _partition_paths(source_root)
    if not partitions:
        raise FileNotFoundError(f"no SFT partitions found under {source_root}")

    marker = tokenizer.encode("Assistant:\n", add_special_tokens=False)
    eos_token_id = tokenizer.eos_token_id
    if not marker or eos_token_id is None:
        raise ValueError("tokenizer must encode Assistant marker and define EOS")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    try:
        for path in source_root.iterdir():
            if path.is_file() and path.name != "chat_template.json":
                _link_or_copy(path, temporary / path.name)
        template_path = source_root / "chat_template.json"
        if template_path.is_file():
            template = json.loads(template_path.read_text(encoding="utf-8"))
            template["training_projection"] = ALL_ASSISTANT_PROJECTION
            (temporary / "chat_template.json").write_text(
                json.dumps(template, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        results = {}
        for partition in partitions:
            relative = partition.relative_to(source_root)
            name = relative.as_posix() or "."
            target = temporary if name == "." else temporary / relative
            results[name] = _relabel_partition(
                partition,
                target,
                assistant_marker=list(marker),
                eos_token_id=int(eos_token_id),
                skip_content_verification=skip_content_verification,
            )
        temporary.rename(output_root)
        return results
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
