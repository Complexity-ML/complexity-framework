"""Local native Agentic SFT projection, selection and shard compilation.

Quotas and source revisions belong to caller-owned recipes. This module does
not download datasets, insert reasoning traces, publish artifacts or launch
training. Labels already predict the next token, including the final EOS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .sft_shard import (
    FINAL_ASSISTANT_SUPERVISION,
    LEGACY_ALL_ASSISTANT_SUPERVISION,
    MASKED_ASSISTANT_HISTORY,
    SHARD_FORMAT_V2,
)


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def clean_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def stable_hash(value: Any) -> str:
    text = value if isinstance(value, str) else compact_json(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_hash(value: str) -> str:
    return stable_hash(" ".join(value.lower().split()))


def direct_completion(text: str) -> str:
    return f"<|final_start|>{text}<|final_end|><|end_of_turn|>"


def tool_completion(call: Mapping[str, Any]) -> str:
    return "<|tool_call_start|>" + compact_json(call) + "<|tool_call_end|><|end_of_turn|>"


def normalize_tool(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, dict):
        return None
    function = raw.get("function") if isinstance(raw.get("function"), dict) else raw
    name = clean_text(function.get("name"))
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not name or not isinstance(arguments, dict):
        return None
    return {"name": name, "arguments": arguments}


def normalize_tools(raw_tools: Any) -> list[dict[str, Any]] | None:
    if not isinstance(raw_tools, list) or not raw_tools:
        return None
    tools = []
    for raw in raw_tools:
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return None
        if not isinstance(raw, dict):
            return None
        tools.append(raw)
    return tools


def offered_tool_names(tools: list[dict[str, Any]]) -> set[str]:
    names = set()
    for tool in tools:
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = clean_text(function.get("name"))
        if name:
            names.add(name)
    return names


def tools_prefix(tools: list[dict[str, Any]]) -> str:
    return "<|system|>Available tools:\n" + compact_json(tools) + "<|end_of_turn|>"


def render_history(messages: list[dict[str, Any]]) -> str | None:
    pieces = []
    for message in messages:
        if not isinstance(message, dict):
            return None
        role, content = clean_text(message.get("role")).lower(), clean_text(message.get("content"))
        if role in {"system", "user"} and content:
            pieces.append(f"<|{role}|>{content}<|end_of_turn|>")
        elif role == "assistant" and content:
            pieces.append("<|assistant|>" + direct_completion(content))
        elif role == "assistant" and message.get("tool_calls"):
            calls = message["tool_calls"]
            if not isinstance(calls, list) or len(calls) != 1:
                return None
            call = normalize_tool(calls[0])
            if call is None:
                return None
            pieces.append("<|assistant|>" + tool_completion(call))
        elif role == "tool" and content:
            pieces.append(f"<|tool_result_start|>{content}<|tool_result_end|><|end_of_turn|>")
        else:
            return None
    return "".join(pieces)


def parse_when2call_response(content: str) -> tuple[str, str | dict[str, Any]] | None:
    """Normalize one legacy call; malformed call markers never become answers."""
    content = clean_text(content)
    prefix, suffix = "<TOOLCALL>", "</TOOLCALL>"
    if prefix in content or suffix in content:
        if not content.startswith(prefix) or not content.endswith(suffix):
            return None
        try:
            calls = json.loads(content[len(prefix) : -len(suffix)])
        except json.JSONDecodeError:
            return None
        if not isinstance(calls, list) or len(calls) != 1:
            return None
        call = normalize_tool(calls[0])
        return None if call is None else ("tool_call", call)
    return ("no_call", content) if content else None


def mini_projected_record(
    row: dict[str, Any], *, source_dataset: str = "tool-calls-mini"
) -> dict[str, Any] | None:
    messages, tools = row.get("messages"), normalize_tools(row.get("tools"))
    if not isinstance(messages, list) or tools is None:
        return None
    indices = [
        i
        for i, msg in enumerate(messages)
        if isinstance(msg, dict) and clean_text(msg.get("role")).lower() == "assistant"
    ]
    calls = [i for i in indices if messages[i].get("tool_calls")]
    finals = [i for i in indices if clean_text(messages[i].get("content"))]
    if not calls and not finals:
        return None
    row_hash = stable_hash(row)
    choose_call = bool(calls) and (int(row_hash[:8], 16) % 2 == 0 or not finals)
    index = calls[0] if choose_call else finals[-1]
    history = render_history(messages[:index])
    if not history:
        return None
    if choose_call:
        raw = messages[index]["tool_calls"]
        if not isinstance(raw, list) or len(raw) != 1:
            return None
        call = normalize_tool(raw[0])
        if call is None or call["name"] not in offered_tool_names(tools):
            return None
        completion, category = tool_completion(call), "tool_call"
    else:
        completion = direct_completion(clean_text(messages[index].get("content")))
        category = (
            "tool_final"
            if any(msg.get("role") == "tool" for msg in messages[:index])
            else "no_call"
        )
    return {
        "prompt": tools_prefix(tools) + history + "<|assistant|>",
        "completion": completion,
        "source_dataset": source_dataset,
        "source_subset": "default",
        "source_id": f"{source_dataset}:{row_hash}",
        "category": category,
        "split_origin": "stable-hash",
    }


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _encoded(tokenizer: Any, text: str) -> list[int]:
    # Native Tokenizer's wrapper may have an existing HF postprocessor. Pass
    # the flag to the underlying engine as well: this encoder owns the EOS.
    engine = getattr(tokenizer, "_tokenizer", tokenizer)
    encoded = engine.encode(text, add_special_tokens=False)
    return list(encoded.ids if hasattr(encoded, "ids") else encoded)


def _finish_encoding(tokenizer: Any, ids: list[int], supervised: list[bool], seq_len: int):
    import torch

    _positive_int(seq_len, "seq_len")
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("SFT encoding requires an explicit EOS token ID")
    ids, supervised = [*ids, int(eos)], [*supervised, True]
    if len(ids) < 2:
        raise ValueError("example contains fewer than two tokens")
    if len(ids) > seq_len + 1:
        raise ValueError(f"example exceeds context without truncation: {len(ids)} > {seq_len + 1}")
    inputs = ids[:-1]
    labels = [ids[i + 1] if supervised[i + 1] else -100 for i in range(len(inputs))]
    padding = seq_len - len(inputs)
    pad = tokenizer.pad_token_id
    inputs.extend([int(eos if pad is None else pad)] * padding)
    labels.extend([-100] * padding)
    return {
        "input_ids": torch.tensor(inputs, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def encode_agentic_trajectory(
    tokenizer: Any,
    record: Mapping[str, Any],
    seq_len: int,
    *,
    assistant_supervision: str = LEGACY_ALL_ASSISTANT_SUPERVISION,
):
    """Mask context/tool results; supervise explicit assistant actions and EOS.

    The default preserves historical all-assistant labels. Callers compiling a
    final-assistant-only dataset must select that policy explicitly.
    """
    _positive_int(seq_len, "seq_len")
    trajectory = record.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError("trajectory must be a non-empty list")
    if not isinstance(trajectory[-1], dict) or trajectory[-1].get("role") != "assistant":
        raise ValueError("trajectory must end with an assistant action")
    if assistant_supervision not in {LEGACY_ALL_ASSISTANT_SUPERVISION, FINAL_ASSISTANT_SUPERVISION}:
        raise ValueError("unsupported assistant supervision")
    ids, supervised = [], []
    for index, message in enumerate(trajectory):
        if not isinstance(message, dict):
            raise ValueError(f"trajectory message {index} is not an object")
        role, content = clean_text(message.get("role")).lower(), clean_text(message.get("content"))
        if not content:
            raise ValueError(f"trajectory message {index} has empty content")
        train = False
        if role in {"system", "user"}:
            text = f"<|{role}|>{content}<|end_of_turn|>"
        elif role == "tool":
            text = f"<|tool_result_start|>{content}<|tool_result_end|><|end_of_turn|>"
        elif role == "assistant":
            if not content.endswith("<|end_of_turn|>") or "<|assistant|>" in content:
                raise ValueError(
                    f"assistant action {index} requires a body with end_of_turn, without a prefix"
                )
            if any(
                marker in content
                for marker in (
                    "<|system|>",
                    "<|user|>",
                    "<|tool_result_start|>",
                    "<|tool_result_end|>",
                )
            ):
                raise ValueError(f"assistant action {index} contains context-role markers")
            prefix = _encoded(tokenizer, "<|assistant|>")
            ids.extend(prefix)
            supervised.extend([False] * len(prefix))
            text = content
            train = (
                assistant_supervision == LEGACY_ALL_ASSISTANT_SUPERVISION
                or index == len(trajectory) - 1
            )
        else:
            raise ValueError(f"unsupported trajectory role: {role}")
        encoded = _encoded(tokenizer, text)
        ids.extend(encoded)
        supervised.extend([train] * len(encoded))
    return _finish_encoding(tokenizer, ids, supervised, seq_len)


def encode_native_sft_record(tokenizer: Any, record: Mapping[str, Any], seq_len: int):
    """Encode literal native prompt/completion text without truncation."""
    _positive_int(seq_len, "seq_len")
    prompt, completion = clean_text(record.get("prompt")), clean_text(record.get("completion"))
    if not prompt.endswith("<|assistant|>") or not completion.endswith("<|end_of_turn|>"):
        raise ValueError("native SFT requires assistant-prefixed prompt and complete target turn")
    prompt_ids, completion_ids = _encoded(tokenizer, prompt), _encoded(tokenizer, completion)
    if not prompt_ids or not completion_ids:
        raise ValueError("native SFT requires nonempty prompt and completion tokens")
    return _finish_encoding(
        tokenizer,
        prompt_ids + completion_ids,
        [False] * len(prompt_ids) + [True] * len(completion_ids),
        seq_len,
    )


def load_sft_recipe(path: str | Path) -> dict[str, Any]:
    """Read explicit quotas; no global production-size or source defaults."""
    recipe = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(recipe, dict) or recipe.get("format") != "complexity-sft-preparation-v1":
        raise ValueError("expected complexity-sft-preparation-v1 recipe")
    quotas = recipe.get("quotas")
    if not isinstance(quotas, dict) or set(quotas) != {"train", "validation"}:
        raise ValueError("recipe requires train and validation quotas")
    for split, sources in quotas.items():
        if not isinstance(sources, dict) or not sources:
            raise ValueError(f"{split} quotas must be a nonempty source mapping")
        for name, quota in sources.items():
            if not isinstance(name, str) or not name:
                raise ValueError("source names must be nonempty strings")
            _positive_int(quota, f"{split}/{name} quota")
    return recipe


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_directory(pending: Path, output: Path) -> None:
    """Reserve a new output before moving staged files; never replace old data."""
    output.mkdir()
    try:
        for child in pending.iterdir():
            os.replace(child, output / child.name)
    except BaseException:
        import shutil

        shutil.rmtree(output)
        raise


def prepare_sft_dataset(
    sources: Mapping[str, Mapping[str, Iterable[Mapping[str, Any]]]],
    recipe: Mapping[str, Any],
    tokenizer: Any,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Select projected local records in source order with exact explicit quotas.

    Inputs are split/source iterables of native prompt/completion records. Source
    projection is separate. Validation is selected first; prompt and source-ID
    deduplication spans both splits. Exhaustion fails, never repeats examples.
    """
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    quotas = recipe.get("quotas", {})
    if recipe.get("format") != "complexity-sft-preparation-v1":
        raise ValueError("expected complexity-sft-preparation-v1 recipe")
    if set(quotas) != {"train", "validation"} or set(sources) != set(quotas):
        raise ValueError("train and validation sources/quotas are required")
    max_length = _positive_int(recipe.get("sequence_length", 2048), "sequence_length")
    prompts, ids_seen, selected, rejected = set(), set(), {}, Counter()
    for split in ("validation", "train"):
        if not isinstance(quotas[split], Mapping) or not quotas[split]:
            raise ValueError(f"{split} quotas must be nonempty")
        if set(sources[split]) != set(quotas[split]):
            raise ValueError(f"{split} sources and quotas differ")
        records = []
        for source_name, quota in quotas[split].items():
            _positive_int(quota, f"{split}/{source_name} quota")
            accepted = 0
            for row in sources[split][source_name]:
                if not isinstance(row, Mapping):
                    rejected["malformed"] += 1
                    continue
                record = dict(row)
                prompt, completion = (
                    clean_text(record.get("prompt")),
                    clean_text(record.get("completion")),
                )
                source_id = (
                    clean_text(record.get("source_id")) or f"{source_name}:{stable_hash(record)}"
                )
                key = normalized_hash(prompt)
                if key in prompts or source_id in ids_seen:
                    rejected["duplicate"] += 1
                    continue
                if any(
                    marker in completion
                    for marker in ("<|think_start|>", "<|think_end|>", "<TOOLCALL>", "</TOOLCALL>")
                ):
                    rejected["unsupported_target_markers"] += 1
                    continue
                try:
                    _validate_target(completion)
                    encoded = encode_native_sft_record(tokenizer, record, max_length)
                except ValueError:
                    rejected["invalid_or_overlength"] += 1
                    continue
                length = int((encoded["labels"] != -100).nonzero()[-1]) + 1
                record.update(
                    source_id=source_id,
                    source_dataset=record.get("source_dataset", source_name),
                    token_count=length + 1,
                )
                records.append(record)
                prompts.add(key)
                ids_seen.add(source_id)
                accepted += 1
                if accepted == quota:
                    break
            if accepted != quota:
                raise ValueError(
                    f"{split}/{source_name} exhausted: {accepted}/{quota} accepted; rejected={dict(rejected)}"
                )
        selected[split] = records
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".complexity-sft-", dir=output.parent) as directory:
        pending = Path(directory)
        artifacts = {}
        for split, filename in (("train", "train.jsonl"), ("validation", "eval.jsonl")):
            path = pending / filename
            with path.open("w", encoding="utf-8") as handle:
                for row in selected[split]:
                    handle.write(compact_json(row) + "\n")
            artifacts[split] = {
                "file": filename,
                "examples": len(selected[split]),
                "sha256": _hash_file(path),
            }
        manifest = {
            "format": "complexity-prepared-sft-v1",
            "recipe": dict(recipe),
            "artifacts": artifacts,
            "rejected": dict(rejected),
            "no_truncation": True,
            "reasoning_inserted": False,
            "quality_status": "local_contracts_checked",
            "release_eligible": False,
        }
        (pending / "dataset_info.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        _publish_directory(pending, output)
    return manifest


def _validate_target(completion: str) -> None:
    end = "<|end_of_turn|>"
    if completion.count(end) != 1 or not completion.endswith(end):
        raise ValueError("target requires exactly one terminal end_of_turn")
    body = completion[: -len(end)]
    if body.startswith("<|tool_call_start|>") and body.endswith("<|tool_call_end|>"):
        raw = json.loads(body[len("<|tool_call_start|>") : -len("<|tool_call_end|>")])
        if not isinstance(raw, dict) or normalize_tool(raw) != raw:
            raise ValueError("target requires one canonical structured tool call")
    elif not (
        body.startswith("<|final_start|>")
        and body.endswith("<|final_end|>")
        and body.count("<|final_start|>") == body.count("<|final_end|>") == 1
    ):
        raise ValueError("target requires complete native final-answer markers")


def compile_sft_dataset(
    dataset_dir: str | Path,
    tokenizer_dir: str | Path,
    output_dir: str | Path,
    *,
    sequence_length: int = 2048,
    assistant_supervision: str,
) -> dict[str, Any]:
    """Compile local train/eval JSONL atomically with an explicit label policy.

    Final-only shards are readable by the native SFT workflow. Historical
    all-assistant shards remain a different objective and are rejected there.
    """
    import numpy as np

    from complexity.export.assets import copy_tokenizer_files
    from complexity.inference.chat_template import (
        NATIVE_AGENTIC_FULL_TRAJECTORY_PROJECTION,
        agentic_chat_template,
    )
    from complexity.tokenizer import Tokenizer
    from complexity.tokenizer.config import get_special_tokens

    if assistant_supervision not in {FINAL_ASSISTANT_SUPERVISION, LEGACY_ALL_ASSISTANT_SUPERVISION}:
        raise ValueError("unsupported assistant supervision")
    _positive_int(sequence_length, "sequence_length")
    root, token_root, output = Path(dataset_dir), Path(tokenizer_dir), Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    tokenizer = Tokenizer.load(str(token_root))
    for expected_id, marker in enumerate(get_special_tokens("tr_hash_agentic_reasoning")):
        if tokenizer._tokenizer.token_to_id(marker) != expected_id:
            raise ValueError(f"tokenizer does not preserve native Agentic marker IDs: {marker}")
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer requires EOS")
    eos_token = tokenizer._tokenizer.id_to_token(tokenizer.eos_token_id)
    template = agentic_chat_template(eos_token=eos_token)
    info_path = root / "dataset_info.json"
    source_manifest = json.loads(info_path.read_text()) if info_path.is_file() else None
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".complexity-shards-", dir=output.parent) as directory:
        pending, partitions = Path(directory), {}
        contains_trajectories = False
        for split, filename in (("train", "train.jsonl"), ("eval", "eval.jsonl")):
            source = root / filename
            source_sha = _hash_file(source)
            if source_manifest is not None:
                declared = (
                    source_manifest.get("artifacts", {})
                    .get("train" if split == "train" else "validation", {})
                    .get("sha256")
                )
                if declared != source_sha:
                    raise ValueError(f"source checksum mismatch: {filename}")
            target = pending / split
            target.mkdir()
            counts, offset = Counter(), 0
            with (
                source.open(encoding="utf-8") as rows,
                (target / "input_ids.bin").open("wb") as inputs,
                (target / "labels.bin").open("wb") as labels,
                (target / "examples.jsonl").open("w", encoding="utf-8") as examples,
            ):
                for row_number, line in enumerate(rows, 1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError(f"{filename}:{row_number}: expected an object")
                    contains_trajectories = contains_trajectories or "trajectory" in record
                    encoded = (
                        encode_agentic_trajectory(
                            tokenizer,
                            record,
                            sequence_length,
                            assistant_supervision=assistant_supervision,
                        )
                        if "trajectory" in record
                        else encode_native_sft_record(tokenizer, record, sequence_length)
                    )
                    length = int((encoded["labels"] != -100).nonzero()[-1]) + 1
                    if "token_count" in record and record["token_count"] != length + 1:
                        raise ValueError(f"{filename}:{row_number}: token_count mismatch")
                    token_ids = encoded["input_ids"][:length].numpy().astype("<u4", copy=False)
                    targets = encoded["labels"][:length].numpy().astype("<i4", copy=False)
                    if token_ids.max() >= tokenizer.vocab_size or np.any(
                        (targets != -100) & ((targets < 0) | (targets >= tokenizer.vocab_size))
                    ):
                        raise ValueError(f"{filename}:{row_number}: token ID outside vocabulary")
                    supervised = int(np.count_nonzero(targets != -100))
                    inputs.write(token_ids.tobytes())
                    labels.write(targets.tobytes())
                    examples.write(
                        compact_json(
                            {
                                "example_id": record.get("source_id", f"row-{row_number}"),
                                "task": record.get("category", "unknown"),
                                "source": record.get("source_dataset", "local"),
                                "offset": offset,
                                "num_tokens": length,
                                "supervised_tokens": supervised,
                            }
                        )
                        + "\n"
                    )
                    offset += length
                    counts["examples"] += 1
                    counts["supervised_tokens"] += supervised
            if not counts["examples"]:
                raise ValueError(f"{filename} contains no examples")
            if _hash_file(source) != source_sha:
                raise ValueError(f"source changed during compilation: {filename}")
            index = {
                "format": SHARD_FORMAT_V2,
                "assistant_supervision": assistant_supervision,
                "history_assistant_turns": MASKED_ASSISTANT_HISTORY
                if assistant_supervision == FINAL_ASSISTANT_SUPERVISION
                else "supervised",
                "chat_template_id": template["id"],
                "chat_template_eos_token": eos_token,
                "examples": counts["examples"],
                "num_tokens": offset,
                "supervised_tokens": counts["supervised_tokens"],
                "seq_len": sequence_length,
                "vocab_size": tokenizer.vocab_size,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "input_dtype": "uint32-le",
                "label_dtype": "int32-le",
                "truncation_policy": "fail_closed_no_truncation",
                "label_alignment": "already_next_token",
                "source_jsonl_sha256": source_sha,
                "files": {
                    name: _hash_file(target / name)
                    for name in ("input_ids.bin", "labels.bin", "examples.jsonl")
                },
            }
            (target / "sft.idx.json").write_text(json.dumps(index, indent=2) + "\n")
            partitions[split] = index
        copy_tokenizer_files(token_root, pending / "tokenizer")
        tokenizer_sha = _hash_file(pending / "tokenizer/tokenizer.json")
        if contains_trajectories:
            template["training_projection"] = NATIVE_AGENTIC_FULL_TRAJECTORY_PROJECTION
        (pending / "chat_template.json").write_text(json.dumps(template, indent=2) + "\n")
        manifest = {
            "format": SHARD_FORMAT_V2,
            "partitions": partitions,
            "tokenizer_sha256": tokenizer_sha,
            "release_eligible": False,
        }
        (pending / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        _publish_directory(pending, output)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="local train.jsonl/eval.jsonl directory")
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("output", type=Path, help="new output directory")
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument(
        "--assistant-supervision",
        choices=(FINAL_ASSISTANT_SUPERVISION, LEGACY_ALL_ASSISTANT_SUPERVISION),
        required=True,
    )
    args = parser.parse_args(argv)
    result = compile_sft_dataset(
        args.dataset,
        args.tokenizer,
        args.output,
        sequence_length=args.sequence_length,
        assistant_supervision=args.assistant_supervision,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
