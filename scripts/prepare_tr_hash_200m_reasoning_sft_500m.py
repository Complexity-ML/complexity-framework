#!/usr/bin/env python3
"""Build the audited 500M-token TR-HASH reasoning SFT mixture.

The release target is expressed in *visible training tokens*, not rows or
repeated epoch exposure.  Sources are streamed, revision pinned, normalized,
quality gated, deduplicated globally and checked against protected benchmark
questions before a row is admitted.  No example may be truncated.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import unicodedata
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset

from complexity.inference.chat_template import align_chat_template_eos, default_chat_template
from complexity.tokenizer import Tokenizer

try:
    from scripts.prepare_tr_hash_200m_clean_sft import (
        _encoded_length,
        assistant_turn_examples,
        conversation_digest,
        normalize_source_row,
        rejection_reasons,
        sha256,
        strip_training_artifacts,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from prepare_tr_hash_200m_clean_sft import (
        _encoded_length,
        assistant_turn_examples,
        conversation_digest,
        normalize_source_row,
        rejection_reasons,
        sha256,
        strip_training_artifacts,
    )


SUPPORTED_ADAPTERS = {
    "messages",
    "instruction_response",
    "openr1_math_verified",
    "numina_math_15",
}


@dataclass(frozen=True)
class BenchmarkGuard:
    exact: frozenset[str]
    sixteen_token_prefixes: frozenset[tuple[str, ...]]


def make_benchmark_guard(prompts: set[str]) -> BenchmarkGuard:
    prefixes = {tuple(prompt.split()[:16]) for prompt in prompts if len(prompt.split()) >= 16}
    return BenchmarkGuard(frozenset(prompts), frozenset(prefixes))


def normalize_for_dedup(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+", text, flags=re.UNICODE))


def normalized_conversation_digest(messages: list[dict[str, str]]) -> str:
    payload = "\n".join(
        f"{message['role']}:{normalize_for_dedup(message['content'])}" for message in messages
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def visible_training_tokens(
    messages: list[dict[str, str]],
    tokenizer: Tokenizer,
    chat_template: dict[str, Any],
) -> int:
    # The materialized shard stores full[:-1] as input_ids.  Match that exact
    # count so the JSONL selection audit and binary-shard audit agree.
    return _encoded_length(messages, tokenizer, chat_template) - 1


def _passes_field_filters(row: dict[str, Any], source: dict[str, Any]) -> bool:
    for field, allowed in source.get("include_values", {}).items():
        if row.get(field) not in allowed:
            return False
    for field, forbidden in source.get("exclude_values", {}).items():
        if row.get(field) in forbidden:
            return False
    return True


def _numina_math_15(row: dict[str, Any]) -> list[dict[str, str]] | None:
    if str(row.get("problem_is_valid", "")).casefold() != "yes":
        return None
    if str(row.get("solution_is_valid", "")).casefold() != "yes":
        return None
    problem = str(row.get("problem") or "").strip()
    solution = str(row.get("solution") or "").strip()
    if not problem or not solution:
        return None
    return [
        {"role": "user", "content": problem},
        {"role": "assistant", "content": solution},
    ]


def normalize_reasoning_row(
    row: dict[str, Any], source: dict[str, Any]
) -> list[dict[str, str]] | None:
    if not _passes_field_filters(row, source):
        return None
    adapter = str(source["adapter"])
    if adapter == "numina_math_15":
        return _numina_math_15(row)
    return normalize_source_row(row, adapter=adapter)


def _iter_source_rows(
    source: dict[str, Any], *, seed: int, shuffle_buffer: int
) -> Iterable[dict[str, Any]]:
    if "local_jsonl" in source:
        path = Path(os.path.expandvars(str(source["local_jsonl"])))
        if not path.is_file():
            raise FileNotFoundError(f"missing local replay JSONL: {path}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    stream = load_dataset(
        source["dataset"],
        source.get("config"),
        split=source["split"],
        revision=source["revision"],
        streaming=True,
    ).shuffle(seed=seed, buffer_size=shuffle_buffer)
    yield from stream


def _benchmark_text(row: dict[str, Any], field: str) -> str:
    value = row.get(field, "")
    return str(value or "").strip()


def load_protected_benchmark_prompts(recipe: dict[str, Any]) -> BenchmarkGuard:
    protected: set[str] = set()
    for benchmark in recipe.get("protected_benchmarks", []):
        if "archive_url" in benchmark:
            with urllib.request.urlopen(benchmark["archive_url"], timeout=120) as response:
                payload = response.read()
            actual_sha256 = hashlib.sha256(payload).hexdigest()
            if actual_sha256 != benchmark["archive_sha256"]:
                raise RuntimeError(f"{benchmark['name']} archive hash mismatch: {actual_sha256}")
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                for member in benchmark["members"]:
                    with archive.open(member) as handle:
                        for raw_line in handle:
                            row = json.loads(raw_line.decode("utf-8"))
                            text = normalize_for_dedup(
                                _benchmark_text(row, benchmark["text_field"])
                            )
                            if text:
                                protected.add(text)
            continue
        for split in benchmark["splits"]:
            dataset = load_dataset(
                benchmark["dataset"],
                benchmark.get("config"),
                split=split,
                revision=benchmark["revision"],
                streaming=True,
            )
            for row in dataset:
                text = normalize_for_dedup(_benchmark_text(row, benchmark["text_field"]))
                if text:
                    protected.add(text)
    return make_benchmark_guard(protected)


def benchmark_overlap(
    messages: list[dict[str, str]], protected_prompts: BenchmarkGuard | set[str]
) -> bool:
    guard = (
        protected_prompts
        if isinstance(protected_prompts, BenchmarkGuard)
        else make_benchmark_guard(protected_prompts)
    )
    user_text = normalize_for_dedup(
        "\n".join(message["content"] for message in messages if message["role"] == "user")
    )
    if user_text in guard.exact:
        return True
    # Catch a verbatim benchmark question embedded in a longer wrapper without
    # an O(rows * benchmark-size) substring scan.
    words = user_text.split()
    windows = {tuple(words[index : index + 16]) for index in range(len(words) - 15)}
    return not windows.isdisjoint(guard.sixteen_token_prefixes)


def load_recipe(path: Path) -> dict[str, Any]:
    recipe = json.loads(path.read_text(encoding="utf-8"))
    if int(recipe.get("target_unique_formatted_tokens", 0)) != 500_000_000:
        raise ValueError("reasoning SFT recipe must target exactly 500M nominal tokens")
    if int(recipe.get("sequence_length", 0)) != 2_048:
        raise ValueError("reasoning SFT recipe must use the released 2,048-token context")
    sources = recipe.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("reasoning SFT recipe must define sources")
    names = [str(source["name"]) for source in sources]
    if len(names) != len(set(names)):
        raise ValueError("reasoning SFT source names must be unique")
    unsupported = {
        str(source.get("adapter"))
        for source in sources
        if source.get("adapter") not in SUPPORTED_ADAPTERS
    }
    if unsupported:
        raise ValueError(f"unsupported reasoning adapters: {sorted(unsupported)}")
    target = sum(int(source["train_token_target"]) for source in sources)
    if target != 500_000_000:
        raise ValueError(f"source token quotas sum to {target}, expected 500000000")
    return recipe


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _restore_completed_rows(
    paths: tuple[Path, Path],
) -> tuple[set[str], set[str]]:
    seen_exact: set[str] = set()
    seen_normalized: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                messages = row.get("messages")
                if not isinstance(messages, list) or not messages:
                    raise RuntimeError(f"malformed resume row {path}:{line_number}")
                exact = conversation_digest(messages)
                normalized = normalized_conversation_digest(messages)
                if exact in seen_exact:
                    raise RuntimeError(f"duplicate exact resume row {path}:{line_number}")
                seen_exact.add(exact)
                seen_normalized.add(normalized)
    return seen_exact, seen_normalized


def _truncate_to_offset(path: Path, offset: int) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing partial file required for resume: {path}")
    with path.open("r+b") as handle:
        if offset < 0 or offset > handle.seek(0, os.SEEK_END):
            raise RuntimeError(f"invalid resume byte offset {offset} for {path}")
        handle.truncate(offset)


def _write_build_state(
    *,
    path: Path,
    recipe_sha256: str,
    train_handle,
    eval_handle,
    source_report: dict[str, Any],
) -> None:
    train_handle.flush()
    eval_handle.flush()
    os.fsync(train_handle.fileno())
    os.fsync(eval_handle.fileno())
    _atomic_write_json(
        path,
        {
            "schema_version": 1,
            "recipe_sha256": recipe_sha256,
            "completed_sources": list(source_report),
            "source_report": source_report,
            "train_byte_offset": train_handle.tell(),
            "eval_byte_offset": eval_handle.tell(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recipe",
        type=Path,
        default=Path("configs/tr_hash_200m_reasoning_sft_500m.json"),
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--max-scan-examples", type=int, default=2_000_000)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last atomically completed source in .build_state.json.",
    )
    parser.add_argument(
        "--smoke-token-target",
        type=int,
        default=0,
        help="Use this per-source token target and skip remote benchmark loading.",
    )
    args = parser.parse_args()

    recipe = load_recipe(args.recipe)
    recipe_sha256 = sha256(args.recipe)
    tokenizer = Tokenizer.load(str(args.tokenizer))
    if len(tokenizer) != 32_000:
        raise ValueError(f"expected the released 32K tokenizer, got {len(tokenizer)}")
    eos_text = tokenizer.decode([tokenizer.eos_token_id], skip_special_tokens=False)
    chat_template = align_chat_template_eos(default_chat_template(), eos_token=eos_text)
    protected = (
        make_benchmark_guard(set())
        if args.smoke_token_target > 0
        else load_protected_benchmark_prompts(recipe)
    )
    sequence_length = int(recipe["sequence_length"])
    policy = dict(recipe["quality_policy"])
    seed = int(recipe["seed"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_partial = args.output_dir / "train.jsonl.partial"
    eval_partial = args.output_dir / "eval.jsonl.partial"
    train_path = args.output_dir / "train.jsonl"
    eval_path = args.output_dir / "eval.jsonl"
    state_path = args.output_dir / ".build_state.json"
    source_report: dict[str, Any] = {}
    file_mode = "w"
    if args.resume and state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("recipe_sha256") != recipe_sha256:
            raise RuntimeError("resume recipe hash differs from the atomically saved build state")
        saved_source_report = dict(state.get("source_report", {}))
        completed_sources = list(state.get("completed_sources", []))
        if set(completed_sources) != set(saved_source_report):
            raise RuntimeError("resume state source set is inconsistent")
        # JSON is written with sorted keys for reproducible bytes, so restore
        # the semantic recipe order from the explicit completed_sources list.
        source_report = {
            source_name: saved_source_report[source_name] for source_name in completed_sources
        }
        expected_prefix = [str(item["name"]) for item in recipe["sources"]][
            : len(completed_sources)
        ]
        if completed_sources != expected_prefix:
            raise RuntimeError("resume state does not match the recipe source prefix")
        _truncate_to_offset(train_partial, int(state["train_byte_offset"]))
        _truncate_to_offset(eval_partial, int(state["eval_byte_offset"]))
        seen_exact, seen_normalized = _restore_completed_rows((train_partial, eval_partial))
        file_mode = "a"
        print(
            json.dumps(
                {
                    "resume": {
                        "completed_sources": completed_sources,
                        "train_examples": sum(
                            int(item["train_examples"]) for item in source_report.values()
                        ),
                    }
                },
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        seen_exact = set()
        seen_normalized = set()
        if args.resume:
            print("[resume] no completed-source state found; starting fresh", flush=True)

    with (
        train_partial.open(file_mode, encoding="utf-8") as train_handle,
        eval_partial.open(file_mode, encoding="utf-8") as eval_handle,
    ):
        for source_index, source in enumerate(recipe["sources"]):
            source_name = str(source["name"])
            if source_name in source_report:
                continue
            token_target = int(source["train_token_target"])
            if source.get("fill_to_total"):
                actual_so_far = sum(
                    int(item["actual_train_tokens"]) for item in source_report.values()
                )
                token_target = 500_000_000 - actual_so_far
                if token_target <= 0:
                    raise RuntimeError(
                        f"{source_name}: no positive fill quota remains after {actual_so_far} tokens"
                    )
            if args.smoke_token_target > 0:
                token_target = min(token_target, args.smoke_token_target)
            eval_target = int(source.get("eval_target_examples", 0))
            scanned = kept_train = kept_eval = train_tokens = 0
            rejected: Counter[str] = Counter()
            if source.get("trusted_tokenized_replay"):
                normalized_aliases = 0
                removed_benchmark_tokens = 0
                for row in _iter_source_rows(
                    source,
                    seed=seed + source_index,
                    shuffle_buffer=args.shuffle_buffer,
                ):
                    scanned += 1
                    messages = row.get("messages")
                    if not isinstance(messages, list) or not messages:
                        raise RuntimeError(f"{source_name}: malformed trusted replay row {scanned}")
                    if benchmark_overlap(messages, protected):
                        rejected["benchmark_overlap"] += 1
                        removed_benchmark_tokens += visible_training_tokens(
                            messages, tokenizer, chat_template
                        )
                        continue
                    exact = conversation_digest(messages)
                    normalized = normalized_conversation_digest(messages)
                    if exact in seen_exact:
                        raise RuntimeError(
                            f"{source_name}: exact duplicate in trusted replay at row {scanned}"
                        )
                    if normalized in seen_normalized:
                        normalized_aliases += 1
                    seen_exact.add(exact)
                    seen_normalized.add(normalized)
                    train_handle.write(
                        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
                    kept_train += 1
                expected_examples = int(source["expected_train_examples"])
                if kept_train + rejected["benchmark_overlap"] != expected_examples:
                    raise RuntimeError(
                        f"{source_name}: expected {expected_examples} replay rows, "
                        f"accounted for {kept_train + rejected['benchmark_overlap']}"
                    )
                # The replay token count comes from its already materialized,
                # SHA-audited uint32 shard. Re-encoding it would be redundant.
                train_tokens = token_target - removed_benchmark_tokens
                source_report[source_name] = {
                    "dataset": source.get("dataset"),
                    "local_jsonl": source.get("local_jsonl"),
                    "revision": source.get("revision"),
                    "license": source["license"],
                    "adapter": source["adapter"],
                    "capability": source["capability"],
                    "trusted_tokenized_replay": True,
                    "target_train_tokens": token_target,
                    "actual_train_tokens": train_tokens,
                    "train_examples": kept_train,
                    "eval_examples": 0,
                    "scanned": scanned,
                    "normalized_aliases_retained": normalized_aliases,
                    "removed_benchmark_tokens": removed_benchmark_tokens,
                    "rejected": dict(rejected),
                }
                print(
                    json.dumps({source_name: source_report[source_name]}, sort_keys=True),
                    flush=True,
                )
                _write_build_state(
                    path=state_path,
                    recipe_sha256=recipe_sha256,
                    train_handle=train_handle,
                    eval_handle=eval_handle,
                    source_report=source_report,
                )
                continue
            for row in _iter_source_rows(
                source,
                seed=seed + source_index,
                shuffle_buffer=args.shuffle_buffer,
            ):
                scanned += 1
                messages = normalize_reasoning_row(row, source)
                if messages is None:
                    rejected["invalid_or_filtered"] += 1
                    continue
                messages = strip_training_artifacts(messages)
                examples = assistant_turn_examples(
                    messages,
                    expand=bool(source.get("expand_assistant_turns", False)),
                    tokenizer=tokenizer,
                    sequence_length=sequence_length,
                    chat_template=chat_template,
                )
                for example in examples:
                    reasons = rejection_reasons(
                        example,
                        require_valid_python=bool(source.get("require_valid_python", False)),
                        policy={
                            **policy,
                            **(
                                {"min_assistant_chars": int(source["min_assistant_chars"])}
                                if "min_assistant_chars" in source
                                else {}
                            ),
                        },
                    )
                    if reasons:
                        rejected.update(reasons)
                        continue
                    if benchmark_overlap(example, protected):
                        rejected["benchmark_overlap"] += 1
                        continue
                    exact = conversation_digest(example)
                    normalized = normalized_conversation_digest(example)
                    if exact in seen_exact or normalized in seen_normalized:
                        rejected["duplicate"] += 1
                        continue
                    token_count = visible_training_tokens(example, tokenizer, chat_template)
                    record = (
                        json.dumps(
                            {
                                "messages": example,
                                "source": source_name,
                                "capability": source["capability"],
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    seen_exact.add(exact)
                    seen_normalized.add(normalized)
                    if kept_eval < eval_target:
                        eval_handle.write(record)
                        kept_eval += 1
                        continue
                    train_handle.write(record)
                    kept_train += 1
                    train_tokens += token_count
                    if train_tokens >= token_target:
                        break
                if train_tokens >= token_target:
                    break
                if scanned >= int(source.get("max_scan_examples", args.max_scan_examples)):
                    break
            if train_tokens < token_target or kept_eval != eval_target:
                raise RuntimeError(
                    f"{source_name}: token target={token_target}, actual={train_tokens}, "
                    f"eval={kept_eval}/{eval_target}, scanned={scanned}, "
                    f"rejected={dict(rejected)}"
                )
            source_report[source_name] = {
                "dataset": source.get("dataset"),
                "local_jsonl": source.get("local_jsonl"),
                "config": source.get("config"),
                "split": source.get("split"),
                "revision": source.get("revision"),
                "license": source["license"],
                "adapter": source["adapter"],
                "capability": source["capability"],
                "target_train_tokens": token_target,
                "actual_train_tokens": train_tokens,
                "train_examples": kept_train,
                "eval_examples": kept_eval,
                "scanned": scanned,
                "rejected": dict(sorted(rejected.items())),
            }
            print(json.dumps({source_name: source_report[source_name]}, sort_keys=True), flush=True)
            _write_build_state(
                path=state_path,
                recipe_sha256=recipe_sha256,
                train_handle=train_handle,
                eval_handle=eval_handle,
                source_report=source_report,
            )

    train_partial.replace(train_path)
    eval_partial.replace(eval_path)
    actual_tokens = sum(item["actual_train_tokens"] for item in source_report.values())
    manifest = {
        "schema_version": 1,
        "name": recipe["name"],
        "recipe_sha256": recipe_sha256,
        "seed": seed,
        "sequence_length": sequence_length,
        "nominal_target_unique_formatted_tokens": 500_000_000,
        "actual_unique_formatted_tokens": actual_tokens,
        "token_quota_overshoot": actual_tokens - 500_000_000,
        "token_count_definition": "sum of materialized visible input_ids before packing",
        "chat_template_id": chat_template["id"],
        "chat_template_eos_token": chat_template["eos_token"],
        "quality_policy": policy,
        "protected_benchmarks": recipe.get("protected_benchmarks", []),
        "protected_prompt_count": len(protected.exact),
        "sources": source_report,
        "train_examples": sum(item["train_examples"] for item in source_report.values()),
        "eval_examples": sum(item["eval_examples"] for item in source_report.values()),
        "train_sha256": sha256(train_path),
        "eval_sha256": sha256(eval_path),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    state_path.unlink(missing_ok=True)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
