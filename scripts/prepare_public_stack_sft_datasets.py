#!/usr/bin/env python3
"""Normalize public SFT datasets into file-backed stack-SFT JSONL files.

This script does not generate synthetic calculator/chat examples. It only
rewrites public dataset rows into the local {"prompt", "completion"} schema.
"""

from __future__ import annotations

import argparse
import json
import random
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset


ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
}


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def dump_record(handle, prompt: str, completion: str) -> bool:
    prompt = clean(prompt)
    completion = clean(completion)
    if len(prompt) < 3 or len(completion) < 2:
        return False
    handle.write(json.dumps({"prompt": prompt.rstrip() + "\n", "completion": completion}, ensure_ascii=False) + "\n")
    return True


def chat_prompt(instruction: str, extra_input: str = "") -> str:
    user = clean(instruction)
    extra_input = clean(extra_input)
    if extra_input:
        user = f"{user}\n\n{extra_input}"
    return f"User:\n{user}\n\nAssistant:\n"


def load_streaming(name: str, split: str, seed: int, buffer_size: int):
    ds = load_dataset(name, split=split, streaming=True)
    return ds.shuffle(seed=seed, buffer_size=buffer_size)


def write_code_instruct(out: Path, dataset: str, split: str, max_records: int, seed: int) -> int:
    ds = load_streaming(dataset, split, seed, 10_000)
    kept = 0
    with out.open("w", encoding="utf-8") as handle:
        for row in ds:
            instruction = clean(row.get("instruction") or row.get("prompt") or row.get("question"))
            extra_input = clean(row.get("input") or row.get("context"))
            completion = clean(row.get("output") or row.get("completion") or row.get("answer") or row.get("response"))
            if dump_record(handle, chat_prompt(instruction, extra_input), completion):
                kept += 1
            if kept >= max_records:
                break
    return kept


def parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    value = value.strip()
    if not value:
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def extract_messages(row: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    tools = clean(row.get("tools") or row.get("available_tools") or row.get("functions"))
    raw = row.get("messages") or row.get("conversations") or row.get("conversation")
    raw = parse_jsonish(raw)
    if not isinstance(raw, list):
        return [], tools

    messages: list[dict[str, Any]] = []
    for msg in raw:
        if not isinstance(msg, dict):
            continue
        role = clean(msg.get("role") or msg.get("from")).lower()
        role = ROLE_MAP.get(role, role or "user")
        content = clean(msg.get("content") or msg.get("value"))
        tool_calls = msg.get("tool_calls") or msg.get("function_call")
        messages.append({"role": role, "content": content, "tool_calls": tool_calls})
    return messages, tools


def tool_call_to_tag(call: Any) -> str:
    call = parse_jsonish(call)
    if isinstance(call, list):
        return "\n".join(tool_call_to_tag(item) for item in call if item)
    if not isinstance(call, dict):
        return clean(call)
    fn = call.get("function") if isinstance(call.get("function"), dict) else call
    name = fn.get("name") or call.get("name")
    arguments = fn.get("arguments") or call.get("arguments") or {}
    arguments = parse_jsonish(arguments)
    payload = {"name": name, "arguments": arguments if isinstance(arguments, dict) else {"value": arguments}}
    return f"<tool_call>{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}</tool_call>"


def conversation_to_record(row: dict[str, Any]) -> tuple[str, str] | None:
    messages, tools = extract_messages(row)
    if not messages:
        return None

    assistant_idx = None
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx]["role"] == "assistant":
            assistant_idx = idx
            break
    if assistant_idx is None:
        return None

    prompt_parts: list[str] = []
    if tools:
        prompt_parts.append(f"System:\nAvailable tools:\n{tools}")
    for msg in messages[:assistant_idx]:
        role = msg["role"].title()
        content = msg["content"]
        if content:
            prompt_parts.append(f"{role}:\n{content}")
    if not prompt_parts:
        return None

    assistant = messages[assistant_idx]
    completion = assistant["content"] or tool_call_to_tag(assistant.get("tool_calls"))
    if not completion:
        return None
    return "\n\n".join(prompt_parts) + "\n\nAssistant:\n", completion


def iter_function_call_records(dataset: str, split: str, seed: int, max_scan: int) -> Iterable[tuple[str, str]]:
    ds = load_streaming(dataset, split, seed, 10_000)
    for row in islice(ds, max_scan):
        record = conversation_to_record(row)
        if record:
            yield record


def write_function_calling(
    out: Path,
    dataset: str,
    split: str,
    max_records: int,
    seed: int,
    max_scan: int,
) -> int:
    kept = 0
    with out.open("w", encoding="utf-8") as handle:
        for prompt, completion in iter_function_call_records(dataset, split, seed, max_scan):
            if "<tool_call>" not in completion:
                continue
            if dump_record(handle, prompt, completion):
                kept += 1
            if kept >= max_records:
                break
    return kept


def write_no_tool_copy(general_path: Path, out: Path, max_records: int, seed: int) -> int:
    rng = random.Random(seed)
    lines = [line for line in general_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rng.shuffle(lines)
    selected = lines[:max_records]
    out.write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/sft")
    parser.add_argument("--code-dataset", default="HuggingFaceH4/CodeAlpaca_20K")
    parser.add_argument("--code-split", default="train")
    parser.add_argument("--code-records", type=int, default=20_000)
    parser.add_argument("--tool-dataset", default="NousResearch/hermes-function-calling-v1")
    parser.add_argument("--tool-split", default="train")
    parser.add_argument("--tool-records", type=int, default=40_000)
    parser.add_argument("--tool-max-scan", type=int, default=120_000)
    parser.add_argument("--general", default="data/sft/general_instruct.jsonl")
    parser.add_argument("--no-tool-records", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    code_count = write_code_instruct(
        out_dir / "code_instruct.jsonl",
        args.code_dataset,
        args.code_split,
        args.code_records,
        args.seed,
    )
    print(f"Wrote {code_count:,} code records to {out_dir / 'code_instruct.jsonl'}")

    tool_count = write_function_calling(
        out_dir / "agentic_tool_traces.jsonl",
        args.tool_dataset,
        args.tool_split,
        args.tool_records,
        args.seed + 1,
        args.tool_max_scan,
    )
    print(f"Wrote {tool_count:,} tool-call records to {out_dir / 'agentic_tool_traces.jsonl'}")
    if tool_count == 0:
        raise RuntimeError(f"{args.tool_dataset} produced no tool-call records")

    general_path = Path(args.general)
    if general_path.exists():
        no_tool_count = write_no_tool_copy(general_path, out_dir / "no_tool_instruct.jsonl", args.no_tool_records, args.seed + 2)
        print(f"Wrote {no_tool_count:,} no-tool records to {out_dir / 'no_tool_instruct.jsonl'}")


if __name__ == "__main__":
    main()
