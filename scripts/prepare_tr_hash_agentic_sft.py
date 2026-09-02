#!/usr/bin/env python3
"""Build the canonical balanced TR-HASH Agentic SFT dataset.

The dataset teaches ordinary instruction following first, then tool-use
selection and tool-result continuation.  It deliberately does not synthesize
private reasoning traces and rejects records that exceed the model context.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

TOKENIZER_REPOSITORY = "AETHORIA-AI/TR-HASH-Tokenizer-32K-Agentic"
TOKENIZER_REVISION = "2fcbc2c5359ded0244ca14531f1b3806eebac55e"

SOURCES = {
    "smol_smoltalk": {
        "repository": "HuggingFaceTB/smol-smoltalk",
        "revision": "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc",
        "license": "Apache-2.0",
    },
    "when2call": {
        "repository": "nvidia/When2Call",
        "revision": "0582f7749df63a96fdc3070932e83e72396ace53",
        "license": "CC-BY-4.0",
    },
    "tool_calls_mini": {
        "repository": "qgallouedec/tool-calls-mini",
        "revision": "bd4039441db9074ac4e5621c6ebdd77726e878ef",
        "license": "Apache-2.0",
    },
}

SPECIAL_TOKEN_IDS = {
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

TRAIN_QUOTAS = {
    "smol_smoltalk": 90_000,
    "when2call_pref": 7_900,
    "when2call_sft": 1_676,
    "tool_calls_mini": 424,
}
EVAL_QUOTAS = {
    "smol_smoltalk": 4_500,
    "when2call_test": 475,
    "tool_calls_mini": 25,
}


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def clean_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def stable_hash(value: Any) -> str:
    encoded = compact_json(value) if not isinstance(value, str) else value
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def normalized_hash(value: str) -> str:
    return stable_hash(" ".join(value.lower().split()))


def direct_completion(text: str) -> str:
    return f"<|final_start|>{text}<|final_end|><|end_of_turn|>"


def tool_completion(call: dict[str, Any]) -> str:
    return (
        "<|tool_call_start|>"
        + compact_json(call)
        + "<|tool_call_end|><|end_of_turn|>"
    )


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
    if not name:
        return None
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    return {"name": name, "arguments": arguments}


def normalize_tools(raw_tools: Any) -> list[dict[str, Any]] | None:
    if not isinstance(raw_tools, list) or not raw_tools:
        return None
    tools: list[dict[str, Any]] = []
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
    names: set[str] = set()
    for tool in tools:
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = clean_text(function.get("name"))
        if name:
            names.add(name)
    return names


def tools_prefix(tools: list[dict[str, Any]]) -> str:
    return (
        "<|system|>Available tools:\n"
        + compact_json(tools)
        + "<|end_of_turn|>"
    )


def render_history(messages: list[dict[str, Any]]) -> str | None:
    pieces: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            return None
        role = clean_text(message.get("role")).lower()
        content = clean_text(message.get("content"))
        if role in {"system", "user"} and content:
            pieces.append(f"<|{role}|>{content}<|end_of_turn|>")
        elif role == "assistant" and content:
            pieces.append("<|assistant|>" + direct_completion(content))
        elif role == "assistant" and message.get("tool_calls"):
            raw_calls = message.get("tool_calls")
            if not isinstance(raw_calls, list) or len(raw_calls) != 1:
                return None
            call = normalize_tool(raw_calls[0])
            if call is None:
                return None
            pieces.append("<|assistant|>" + tool_completion(call))
        elif role == "tool" and content:
            pieces.append(
                f"<|tool_result_start|>{content}<|tool_result_end|><|end_of_turn|>"
            )
        else:
            return None
    return "".join(pieces)


def smoltalk_record(row: dict[str, Any], split: str) -> dict[str, Any] | None:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None
    assistant_indices = [
        i
        for i, message in enumerate(messages)
        if isinstance(message, dict)
        and clean_text(message.get("role")).lower() == "assistant"
        and clean_text(message.get("content"))
    ]
    if not assistant_indices:
        return None
    index = assistant_indices[-1]
    prompt = render_history(messages[:index])
    if not prompt:
        return None
    source_id = stable_hash({"messages": messages, "source": row.get("source")})
    return {
        "prompt": prompt + "<|assistant|>",
        "completion": direct_completion(clean_text(messages[index].get("content"))),
        "source_dataset": SOURCES["smol_smoltalk"]["repository"],
        "source_subset": clean_text(row.get("source")) or "unknown",
        "source_id": f"smol-smoltalk:{source_id}",
        "category": "direct",
        "split_origin": split,
    }


def parse_when2call_response(content: str) -> tuple[str, str | dict[str, Any]] | None:
    content = content.strip()
    prefix, suffix = "<TOOLCALL>", "</TOOLCALL>"
    if content.startswith(prefix) and content.endswith(suffix):
        try:
            calls = json.loads(content[len(prefix) : -len(suffix)])
        except json.JSONDecodeError:
            return None
        if not isinstance(calls, list) or len(calls) != 1:
            return None
        call = normalize_tool(calls[0])
        return None if call is None else ("tool_call", call)
    return ("no_call", content) if content else None


def when2call_record(
    row: dict[str, Any], *, response: str, subset: str, split: str, identifier: str | None = None
) -> dict[str, Any] | None:
    tools = normalize_tools(row.get("tools"))
    messages = row.get("messages")
    parsed = parse_when2call_response(response)
    if tools is None or not isinstance(messages, list) or parsed is None:
        return None
    history = render_history(messages)
    if not history:
        return None
    category, value = parsed
    if category == "tool_call":
        assert isinstance(value, dict)
        if value["name"] not in offered_tool_names(tools):
            return None
        completion = tool_completion(value)
    else:
        assert isinstance(value, str)
        completion = direct_completion(value)
    raw_id = identifier or stable_hash({"tools": tools, "messages": messages, "response": response})
    return {
        "prompt": tools_prefix(tools) + history + "<|assistant|>",
        "completion": completion,
        "source_dataset": SOURCES["when2call"]["repository"],
        "source_subset": subset,
        "source_id": f"when2call:{raw_id}",
        "category": category,
        "split_origin": split,
    }


def when2call_test_record(row: dict[str, Any]) -> dict[str, Any] | None:
    answer_type = clean_text(row.get("correct_answer"))
    answers = row.get("answers")
    question = clean_text(row.get("question"))
    if not isinstance(answers, dict) or answer_type not in answers or not question:
        return None
    response = clean_text(answers[answer_type])
    synthetic = {"tools": row.get("tools"), "messages": [{"role": "user", "content": question}]}
    record = when2call_record(
        synthetic,
        response=(f"<TOOLCALL>[{response}]</TOOLCALL>" if answer_type == "tool_call" else response),
        subset=f"test-mcq-{answer_type}",
        split="validation",
        identifier=clean_text(row.get("uuid")) or stable_hash(row),
    )
    if record is not None and answer_type != "tool_call":
        record["category"] = answer_type
    return record


def mini_projected_record(row: dict[str, Any]) -> dict[str, Any] | None:
    messages = row.get("messages")
    tools = normalize_tools(row.get("tools"))
    if not isinstance(messages, list) or tools is None:
        return None
    assistant_indices = [
        i
        for i, message in enumerate(messages)
        if isinstance(message, dict) and clean_text(message.get("role")).lower() == "assistant"
    ]
    if not assistant_indices:
        return None
    row_hash = stable_hash(row)
    tool_indices = [i for i in assistant_indices if messages[i].get("tool_calls")]
    final_indices = [i for i in assistant_indices if clean_text(messages[i].get("content"))]
    # Alternate between learning to call a tool and learning to consume its result.
    choose_call = bool(tool_indices) and (int(row_hash[:8], 16) % 2 == 0 or not final_indices)
    index = tool_indices[0] if choose_call else final_indices[-1]
    history = render_history(messages[:index])
    if not history:
        return None
    message = messages[index]
    if choose_call:
        raw_calls = message.get("tool_calls")
        if not isinstance(raw_calls, list) or len(raw_calls) != 1:
            return None
        call = normalize_tool(raw_calls[0])
        if call is None or call["name"] not in offered_tool_names(tools):
            return None
        completion = tool_completion(call)
        category = "tool_call"
    else:
        content = clean_text(message.get("content"))
        if not content:
            return None
        completion = direct_completion(content)
        category = "tool_final" if any(
            clean_text(item.get("role")).lower() == "tool" for item in messages[:index]
        ) else "no_call"
    return {
        "prompt": tools_prefix(tools) + history + "<|assistant|>",
        "completion": completion,
        "source_dataset": SOURCES["tool_calls_mini"]["repository"],
        "source_subset": "default",
        "source_id": f"tool-calls-mini:{row_hash}",
        "category": category,
        "split_origin": "stable-hash",
    }


def load_piqa_hashes(probe: Path | None) -> set[str]:
    if probe is None:
        return set()
    hashes: set[str] = set()
    for line in (probe / "dev.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        hashes.update(normalized_hash(clean_text(row.get(key))) for key in ("goal", "sol1", "sol2"))
    return hashes


def contaminated(record: dict[str, Any], piqa_hashes: set[str]) -> bool:
    if not piqa_hashes:
        return False
    text = record["prompt"] + record["completion"]
    for marker in SPECIAL_TOKEN_IDS:
        text = text.replace(marker, "\n")
    return any(normalized_hash(piece) in piqa_hashes for piece in text.splitlines() if piece.strip())


def validate_tokenizer(tokenizer: Tokenizer) -> None:
    if tokenizer.get_vocab_size() != 32_000:
        raise ValueError(f"expected vocab 32000, got {tokenizer.get_vocab_size()}")
    for token, expected in SPECIAL_TOKEN_IDS.items():
        actual = tokenizer.token_to_id(token)
        if actual != expected:
            raise ValueError(f"{token}: expected id {expected}, got {actual}")


class Collector:
    def __init__(self, tokenizer: Tokenizer, max_length: int, piqa_hashes: set[str]) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.piqa_hashes = piqa_hashes
        self.seen_prompts: set[str] = set()
        self.rejected: Counter[str] = Counter()

    def add(self, target: list[dict[str, Any]], record: dict[str, Any] | None) -> bool:
        if record is None:
            self.rejected["malformed"] += 1
            return False
        prompt_hash = normalized_hash(record["prompt"])
        if prompt_hash in self.seen_prompts:
            self.rejected["duplicate_prompt"] += 1
            return False
        if contaminated(record, self.piqa_hashes):
            self.rejected["piqa_exact"] += 1
            return False
        token_count = len(self.tokenizer.encode(record["prompt"]).ids) + len(
            self.tokenizer.encode(record["completion"]).ids
        ) + 1
        if token_count > self.max_length + 1:
            self.rejected["overlength"] += 1
            return False
        record["token_count"] = token_count
        self.seen_prompts.add(prompt_hash)
        target.append(record)
        return True


def iter_stream(repository: str, *, split: str, revision: str, config: str | None = None, seed: int) -> Iterable[dict[str, Any]]:
    dataset = load_dataset(repository, config, split=split, revision=revision, streaming=True)
    yield from dataset.shuffle(seed=seed, buffer_size=10_000)


def collect_smoltalk(
    collector: Collector, target: list[dict[str, Any]], *, split: str, quota: int, seed: int
) -> None:
    source = SOURCES["smol_smoltalk"]
    scanned = 0
    for row in iter_stream(
        source["repository"], split=split, revision=source["revision"], seed=seed
    ):
        scanned += 1
        collector.add(target, smoltalk_record(row, split))
        if len(target) >= quota:
            break
        if scanned % 10_000 == 0:
            print(f"smol-smoltalk/{split}: {len(target):,}/{quota:,} accepted", flush=True)
    if len(target) != quota:
        raise RuntimeError(f"smol-smoltalk/{split}: only {len(target):,}/{quota:,}")


def collect_when2call_train(collector: Collector, target: list[dict[str, Any]], seed: int) -> None:
    source = SOURCES["when2call"]
    category_quota = {"tool_call": 2_900, "no_call": 5_000}
    counts: Counter[str] = Counter()
    for row in iter_stream(
        source["repository"], config="train_pref", split="train", revision=source["revision"], seed=seed
    ):
        chosen = row.get("chosen_response")
        response = clean_text(chosen.get("content")) if isinstance(chosen, dict) else ""
        parsed = parse_when2call_response(response)
        if parsed is None or counts[parsed[0]] >= category_quota[parsed[0]]:
            continue
        before = len(target)
        collector.add(
            target,
            when2call_record(row, response=response, subset="train_pref-chosen", split="train"),
        )
        if len(target) > before:
            counts[parsed[0]] += 1
        if all(counts[key] >= value for key, value in category_quota.items()):
            break
    if counts != Counter(category_quota):
        raise RuntimeError(f"When2Call train_pref quota not reached: {dict(counts)}")

    added = 0
    for row in iter_stream(
        source["repository"], config="train_sft", split="train", revision=source["revision"], seed=seed + 1
    ):
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            continue
        response_message = messages[-1]
        if not isinstance(response_message, dict) or clean_text(response_message.get("role")) != "assistant":
            continue
        projected = dict(row)
        projected["messages"] = messages[:-1]
        before = len(target)
        collector.add(
            target,
            when2call_record(
                projected,
                response=clean_text(response_message.get("content")),
                subset="train_sft-no-call",
                split="train",
            ),
        )
        added += len(target) - before
        if added >= TRAIN_QUOTAS["when2call_sft"]:
            break
    if added != TRAIN_QUOTAS["when2call_sft"]:
        raise RuntimeError(f"When2Call train_sft quota not reached: {added}")


def collect_when2call_eval(collector: Collector, target: list[dict[str, Any]], seed: int) -> None:
    source = SOURCES["when2call"]
    quotas = {"tool_call": 175, "request_for_info": 150, "cannot_answer": 150}
    counts: Counter[str] = Counter()
    for row in iter_stream(
        source["repository"], config="test", split="mcq", revision=source["revision"], seed=seed
    ):
        answer_type = clean_text(row.get("correct_answer"))
        if answer_type not in quotas or counts[answer_type] >= quotas[answer_type]:
            continue
        before = len(target)
        collector.add(target, when2call_test_record(row))
        if len(target) > before:
            counts[answer_type] += 1
        if all(counts[key] >= value for key, value in quotas.items()):
            break
    if counts != Counter(quotas):
        raise RuntimeError(f"When2Call test quota not reached: {dict(counts)}")


def collect_mini(
    collector: Collector, train: list[dict[str, Any]], validation: list[dict[str, Any]], seed: int
) -> None:
    source = SOURCES["tool_calls_mini"]
    candidates: list[tuple[str, dict[str, Any]]] = []
    for row in iter_stream(
        source["repository"], split="train", revision=source["revision"], seed=seed
    ):
        record = mini_projected_record(row)
        if record is not None:
            candidates.append((stable_hash(record["source_id"]), record))
    candidates.sort(key=lambda item: item[0])
    required = EVAL_QUOTAS["tool_calls_mini"] + TRAIN_QUOTAS["tool_calls_mini"]
    if len(candidates) < required:
        raise RuntimeError(f"tool-calls-mini: only {len(candidates)}/{required} valid conversations")
    eval_candidates = candidates[: EVAL_QUOTAS["tool_calls_mini"]]
    train_candidates = candidates[EVAL_QUOTAS["tool_calls_mini"] : required]
    for _, record in eval_candidates:
        record["split_origin"] = "stable-hash-validation"
        collector.add(validation, record)
    for _, record in train_candidates:
        record["split_origin"] = "stable-hash-train"
        collector.add(train, record)
    if len(eval_candidates) != EVAL_QUOTAS["tool_calls_mini"] or len(train_candidates) != TRAIN_QUOTAS["tool_calls_mini"]:
        raise RuntimeError("tool-calls-mini split quota mismatch")


def validate_records(records: dict[str, list[dict[str, Any]]], tokenizer: Tokenizer) -> dict[str, Any]:
    prompt_hashes: dict[str, str] = {}
    source_splits: dict[str, str] = {}
    report: dict[str, Any] = {}
    for split, items in records.items():
        categories = Counter()
        sources = Counter()
        token_counts: list[int] = []
        for record in items:
            prompt = record["prompt"]
            completion = record["completion"]
            if not prompt.endswith("<|assistant|>"):
                raise ValueError(f"{split}: prompt does not end with assistant marker")
            if "<|think_start|>" in completion or "<|think_end|>" in completion:
                raise ValueError(f"{split}: reasoning marker found in target")
            if "<TOOLCALL>" in completion:
                raise ValueError(f"{split}: legacy tool-call marker found")
            if completion.count("<|end_of_turn|>") != 1:
                raise ValueError(f"{split}: invalid target turn boundary")
            if "<|tool_call_start|>" in completion:
                payload = completion.split("<|tool_call_start|>", 1)[1].split("<|tool_call_end|>", 1)[0]
                call = json.loads(payload)
                if not isinstance(call, dict) or not call.get("name") or not isinstance(call.get("arguments"), dict):
                    raise ValueError(f"{split}: invalid tool-call JSON")
            else:
                if completion.count("<|final_start|>") != 1 or completion.count("<|final_end|>") != 1:
                    raise ValueError(f"{split}: invalid final-answer markers")
            prompt_hash = normalized_hash(prompt)
            if prompt_hash in prompt_hashes:
                raise ValueError(f"duplicate prompt across {prompt_hashes[prompt_hash]} and {split}")
            prompt_hashes[prompt_hash] = split
            source_id = record["source_id"]
            previous_split = source_splits.get(source_id)
            if previous_split is not None and previous_split != split:
                raise ValueError(f"source leakage for {source_id}")
            source_splits[source_id] = split
            computed = len(tokenizer.encode(prompt).ids) + len(tokenizer.encode(completion).ids) + 1
            if computed != record["token_count"] or computed > 2_049:
                raise ValueError(f"{split}: invalid token count {computed}")
            token_counts.append(computed)
            categories[record["category"]] += 1
            sources[record["source_dataset"]] += 1
        ordered = sorted(token_counts)
        report[split] = {
            "examples": len(items),
            "tokens": sum(token_counts),
            "categories": dict(sorted(categories.items())),
            "sources": dict(sorted(sources.items())),
            "token_length": {
                "min": ordered[0],
                "p50": ordered[len(ordered) // 2],
                "p95": ordered[math.floor((len(ordered) - 1) * 0.95)],
                "max": ordered[-1],
            },
        }
    return report


def write_dataset(
    out_dir: Path,
    records: dict[str, list[dict[str, Any]]],
    report: dict[str, Any],
    collector: Collector,
    seed: int,
    piqa_enabled: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for split, items in records.items():
        rng.shuffle(items)
        filename = "train.jsonl" if split == "train" else "eval.jsonl"
        with (out_dir / filename).open("w", encoding="utf-8") as handle:
            for record in items:
                handle.write(compact_json(record) + "\n")

    manifest = {
        "format": "tr-hash-agentic-sft-balanced-v1",
        "repository": "AETHORIA-AI/TR-HASH-Agentic-SFT-32K-21K",
        "tokenizer": {"repository": TOKENIZER_REPOSITORY, "revision": TOKENIZER_REVISION},
        "sources": SOURCES,
        "max_length": 2048,
        "seed": seed,
        "splits": report,
        "rejected": dict(sorted(collector.rejected.items())),
        "piqa_exact_exclusion_enabled": piqa_enabled,
        "notes": [
            "The canonical repository is replaced in place; no v2 namespace is used.",
            "No synthetic chain-of-thought or think markers are present in targets.",
            "Train and validation use official upstream splits except tool-calls-mini, which uses a stable conversation hash.",
        ],
    }
    (out_dir / "dataset_info.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / ".gitattributes").write_text("*.jsonl filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")
    readme = """---
license: other
language:
- en
task_categories:
- text-generation
tags:
- agentic
- function-calling
- tool-use
- tr-hash
- supervised-finetuning
size_categories:
- 100K<n<1M
pretty_name: TR-HASH Agentic SFT 32K
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: eval.jsonl
---

# TR-HASH Agentic SFT 32K

Balanced instruction and tool-use SFT data for
[`AETHORIA-AI/TR-HASH-Tokenizer-32K-Agentic`](https://huggingface.co/AETHORIA-AI/TR-HASH-Tokenizer-32K-Agentic).
The canonical repository name is retained, while its contents replace the former
tool-heavy 21K laboratory corpus.

## Composition

| Split | General instruction | Tool-aware | Total |
|---|---:|---:|---:|
| Train | 90,000 | 10,000 | 100,000 |
| Validation | 4,500 | 500 | 5,000 |

The 10% tool-aware training slice contains tool calls, no-call decisions with
tools present, and final answers after tool results. The remaining 90% teaches
ordinary instruction following without encouraging tool hallucination.

Every row contains `prompt`, `completion`, source lineage, category and token
count. Direct answers use `<|final_start|>...<|final_end|>`. Tool calls use
native `<|tool_call_start|>...<|tool_call_end|>` markers. No target contains a
synthetic chain-of-thought or `<|think_start|>` marker.

## Sources and licenses

- [HuggingFaceTB/smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) — Apache-2.0.
- [nvidia/When2Call](https://huggingface.co/datasets/nvidia/When2Call) — CC BY 4.0.
- [qgallouedec/tool-calls-mini](https://huggingface.co/datasets/qgallouedec/tool-calls-mini) — Apache-2.0.

This is a mixed-source compilation. Each upstream item remains governed by its
source license; users must preserve the applicable attribution and notices.
Pinned source revisions, exact counts, token statistics and rejection reasons
are recorded in `dataset_info.json`.

## Quality controls

- official upstream train/test splits are used when available;
- the small tool-trajectory source is split by stable conversation hash;
- exact duplicate prompts and cross-split source leakage are rejected;
- examples over 2,048 tokens are rejected rather than truncated;
- tool-call JSON and native marker balance are validated;
- optional exact PIQA probe exclusion is supported by the builder.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--piqa-probe", type=Path)
    args = parser.parse_args()

    tokenizer_path = args.tokenizer
    if tokenizer_path is None:
        tokenizer_path = Path(
            snapshot_download(
                TOKENIZER_REPOSITORY,
                revision=TOKENIZER_REVISION,
                allow_patterns=["tokenizer.json"],
            )
        )
    tokenizer_file = tokenizer_path / "tokenizer.json" if tokenizer_path.is_dir() else tokenizer_path
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    validate_tokenizer(tokenizer)
    piqa_hashes = load_piqa_hashes(args.piqa_probe)
    collector = Collector(tokenizer, args.max_length, piqa_hashes)

    train_general: list[dict[str, Any]] = []
    eval_general: list[dict[str, Any]] = []
    collect_smoltalk(
        collector,
        train_general,
        split="train",
        quota=TRAIN_QUOTAS["smol_smoltalk"],
        seed=args.seed,
    )
    collect_smoltalk(
        collector,
        eval_general,
        split="test",
        quota=EVAL_QUOTAS["smol_smoltalk"],
        seed=args.seed + 1,
    )

    train_agentic: list[dict[str, Any]] = []
    eval_agentic: list[dict[str, Any]] = []
    collect_when2call_train(collector, train_agentic, args.seed + 2)
    collect_when2call_eval(collector, eval_agentic, args.seed + 3)
    collect_mini(collector, train_agentic, eval_agentic, args.seed + 4)

    records = {
        "train": train_general + train_agentic,
        "validation": eval_general + eval_agentic,
    }
    if len(records["train"]) != 100_000 or len(records["validation"]) != 5_000:
        raise RuntimeError(
            f"unexpected totals: train={len(records['train'])}, validation={len(records['validation'])}"
        )
    report = validate_records(records, tokenizer)
    write_dataset(args.out_dir, records, report, collector, args.seed, bool(piqa_hashes))
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"dataset ready: {args.out_dir}")


if __name__ == "__main__":
    main()
