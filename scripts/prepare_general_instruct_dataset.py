#!/usr/bin/env python3
"""Prepare a general web-aligned SFT JSONL for the local SFT runner.

Output format:
    {"prompt": "User:\\n...\\n\\nAssistant:\\n", "completion": "..."}

This intentionally favors general instruction, QA, rewriting and chat data over
code/math-heavy corpora so it fits a FineWeb-pretrained base better.
"""

from __future__ import annotations

import argparse
import json
import random
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset


def text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def write_record(handle, prompt: str, completion: str) -> bool:
    prompt = prompt.strip()
    completion = completion.strip()
    if not prompt or not completion:
        return False
    record = {
        "prompt": f"User:\n{prompt}\n\nAssistant:\n",
        "completion": completion,
    }
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return True


def iter_openorca(max_samples: int | None, seed: int) -> Iterable[tuple[str, str]]:
    ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
    # Streaming datasets do not support deterministic shuffle by index, but this
    # small buffer avoids taking only the very first contiguous slice.
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    if max_samples is not None:
        ds = islice(ds, max_samples)
    for row in ds:
        question = text(row.get("question"))
        response = text(row.get("response"))
        system = text(row.get("system_prompt"))
        prompt = question if not system else f"{system}\n\n{question}"
        yield prompt, response


def iter_dolly(max_samples: int | None, seed: int) -> Iterable[tuple[str, str]]:
    ds = load_dataset("databricks/databricks-dolly-15k", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    if max_samples is not None:
        ds = islice(ds, max_samples)
    for row in ds:
        instruction = text(row.get("instruction"))
        context = text(row.get("context"))
        response = text(row.get("response"))
        prompt = instruction if not context else f"{instruction}\n\n{context}"
        yield prompt, response


def iter_ultrachat(max_samples: int | None, seed: int) -> Iterable[tuple[str, str]]:
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    if max_samples is not None:
        ds = islice(ds, max_samples)
    for row in ds:
        messages = row.get("messages") or []
        if not isinstance(messages, list) or len(messages) < 2:
            continue
        assistant_idx = None
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "assistant":
                assistant_idx = idx
                break
        if assistant_idx is None:
            continue
        user_parts: list[str] = []
        for msg in messages[:assistant_idx]:
            role = text(msg.get("role")).lower()
            content = text(msg.get("content"))
            if not content:
                continue
            if role in {"user", "human"}:
                user_parts.append(content)
            elif role == "system":
                user_parts.append(content)
        completion = text(messages[assistant_idx].get("content"))
        yield "\n\n".join(user_parts), completion


def keep(prompt: str, completion: str, max_prompt_chars: int, max_completion_chars: int) -> bool:
    if len(prompt) < 3 or len(completion) < 3:
        return False
    if len(prompt) > max_prompt_chars or len(completion) > max_completion_chars:
        return False
    bad_markers = ("<image", "![", "http://", "https://")
    joined = f"{prompt}\n{completion}".lower()
    return not any(marker in joined for marker in bad_markers)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/sft/general_instruct.jsonl")
    parser.add_argument("--openorca", type=int, default=80_000)
    parser.add_argument("--ultrachat", type=int, default=60_000)
    parser.add_argument("--dolly", type=int, default=15_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-prompt-chars", type=int, default=5000)
    parser.add_argument("--max-completion-chars", type=int, default=5000)
    args = parser.parse_args()

    sources = [
        ("openorca", iter_openorca(args.openorca, args.seed)),
        ("ultrachat", iter_ultrachat(args.ultrachat, args.seed + 1)),
        ("dolly", iter_dolly(args.dolly, args.seed + 2)),
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = {name: 0 for name, _ in sources}
    skipped = {name: 0 for name, _ in sources}

    # Interleave sources so the file itself is mixed even before the SFT
    # IterableDataset shuffles it each epoch.
    rng = random.Random(args.seed)
    pools = [(name, iter(iterator)) for name, iterator in sources]
    with out.open("w", encoding="utf-8") as handle:
        while pools:
            idx = rng.randrange(len(pools))
            name, iterator = pools[idx]
            try:
                prompt, completion = next(iterator)
            except StopIteration:
                pools.pop(idx)
                continue
            if not keep(prompt, completion, args.max_prompt_chars, args.max_completion_chars):
                skipped[name] += 1
                continue
            if write_record(handle, prompt, completion):
                counts[name] += 1
            else:
                skipped[name] += 1

    print(f"Wrote {sum(counts.values()):,} records to {out}")
    for name in sorted(counts):
        print(f"{name}: kept={counts[name]:,} skipped={skipped[name]:,}")


if __name__ == "__main__":
    main()
