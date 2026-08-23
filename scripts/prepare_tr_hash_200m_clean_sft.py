#!/usr/bin/env python3
"""Build the quality-gated TR-HASH MoE 200M SFT v2 JSONL release.

Unlike the historical 16-way recipe, this builder rejects examples that would
be truncated by the training sequence length.  Long conversations are expanded
into assistant-turn examples and trimmed only at complete turn boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset

from complexity.inference.chat_template import align_chat_template_eos, default_chat_template
from complexity.tokenizer import Tokenizer

try:
    from scripts.audit_luciole_16way_quality import EMPTY_THINK, audit_row
    from scripts.prepare_luciole_16way_sft import normalize_messages
    from scripts.sft_500m_32k_tr import format_record
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from audit_luciole_16way_quality import EMPTY_THINK, audit_row
    from prepare_luciole_16way_sft import normalize_messages
    from sft_500m_32k_tr import format_record


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_recipe(path: Path) -> dict[str, Any]:
    recipe = json.loads(path.read_text(encoding="utf-8"))
    if int(recipe.get("sequence_length", 0)) < 1024:
        raise ValueError("clean SFT recipe requires sequence_length >= 1024")
    sources = recipe.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("clean SFT recipe must define non-empty sources")
    names = [str(source["name"]) for source in sources]
    if len(names) != len(set(names)):
        raise ValueError("clean SFT source names must be unique")
    adapters = {"messages", "instruction_response", "openr1_math_verified"}
    unsupported = {
        str(source.get("adapter")) for source in sources
        if source.get("adapter") not in adapters
    }
    if unsupported:
        raise ValueError(f"unsupported clean SFT adapters: {sorted(unsupported)}")
    total = sum(int(source["train_target"]) for source in sources)
    maximum = int(recipe.get("max_train_examples", total))
    if total > maximum:
        raise ValueError(f"clean SFT target {total} exceeds maximum {maximum}")
    return recipe


def _clean_reasoning_markup(text: str) -> str:
    text = re.sub(r"</?(?:think|final|answer)>", "", text, flags=re.I)
    return EMPTY_THINK.sub("", text).strip()


def normalize_source_row(
    row: dict[str, Any],
    *,
    adapter: str,
) -> list[dict[str, str]] | None:
    if adapter == "messages":
        return normalize_messages(row, max_chars=200_000)
    if adapter == "instruction_response":
        instruction = str(row.get("instruction") or row.get("prompt") or "").strip()
        response = str(row.get("response") or row.get("output") or "").strip()
        if not instruction or not response:
            return None
        return [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
    if adapter == "openr1_math_verified":
        problem = str(row.get("problem") or "").strip()
        reference_solution = str(row.get("solution") or "").strip()
        generations = row.get("generations")
        math_verify = row.get("correctness_math_verify")
        llama_verify = row.get("correctness_llama")
        finish_reasons = row.get("finish_reasons")
        verified_candidates: list[str] = []
        if isinstance(generations, list):
            for index, generation in enumerate(generations):
                if not isinstance(generation, str) or not generation.strip():
                    continue
                math_ok = (
                    isinstance(math_verify, list)
                    and index < len(math_verify)
                    and bool(math_verify[index])
                )
                llama_ok = (
                    isinstance(llama_verify, list)
                    and index < len(llama_verify)
                    and bool(llama_verify[index])
                )
                complete = not isinstance(finish_reasons, list) or (
                    index < len(finish_reasons)
                    and str(finish_reasons[index]).lower() in {"stop", "eos_token"}
                )
                if complete and (math_ok or llama_ok):
                    verified_candidates.append(_clean_reasoning_markup(generation))
        if not problem or not verified_candidates:
            return None
        # The upstream reference solution is usually substantially cleaner and
        # shorter than sampled model traces.  Admit it only when this row also
        # carries at least one successful verifier vote; otherwise use the
        # shortest verified generation.  A label-only reference (for example
        # ``C`` or ``391``) is not accepted as a reasoning trace.
        candidates = list(verified_candidates)
        if len(reference_solution) >= 16:
            candidates.append(reference_solution)
        # Prefer the shortest verified trace. It is the most likely to fit the
        # 2,048-token model while still carrying a verifier-backed solution.
        answer = min(candidates, key=len)
        return [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": answer},
        ]
    raise ValueError(f"unsupported source adapter: {adapter}")


def strip_training_artifacts(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    for message in messages:
        content = EMPTY_THINK.sub("", message["content"]).strip()
        content = re.sub(r"\n{3,}", "\n\n", content)
        if content:
            cleaned.append({"role": message["role"], "content": content})
    return cleaned


def _encoded_length(
    messages: list[dict[str, str]],
    tokenizer: Tokenizer,
    chat_template: dict[str, Any] | None = None,
) -> int:
    prompt, completion = format_record({"messages": messages}, chat_template)
    return (
        len(tokenizer.encode(prompt, add_special_tokens=False))
        + len(tokenizer.encode(completion, add_special_tokens=False))
        + 1
    )


def _drop_oldest_complete_turn(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    systems = [message for message in messages if message["role"] == "system"]
    dialogue = [message for message in messages if message["role"] != "system"]
    if len(dialogue) <= 2:
        return messages
    next_user = next(
        (index for index in range(1, len(dialogue)) if dialogue[index]["role"] == "user"),
        None,
    )
    if next_user is None:
        return messages
    return systems + dialogue[next_user:]


def fit_complete_turns(
    messages: list[dict[str, str]],
    tokenizer: Tokenizer,
    sequence_length: int,
    chat_template: dict[str, Any] | None = None,
) -> list[dict[str, str]] | None:
    candidate = [dict(message) for message in messages]
    while _encoded_length(candidate, tokenizer, chat_template) > sequence_length:
        reduced = _drop_oldest_complete_turn(candidate)
        if reduced == candidate:
            return None
        candidate = reduced
    return candidate


def assistant_turn_examples(
    messages: list[dict[str, str]],
    *,
    expand: bool,
    tokenizer: Tokenizer,
    sequence_length: int,
    chat_template: dict[str, Any] | None = None,
) -> Iterable[list[dict[str, str]]]:
    assistant_indices = [
        index for index, message in enumerate(messages) if message["role"] == "assistant"
    ]
    if not expand and assistant_indices:
        assistant_indices = assistant_indices[-1:]
    for assistant_index in assistant_indices:
        prefix = messages[: assistant_index + 1]
        if not any(message["role"] == "user" for message in prefix[:-1]):
            continue
        fitted = fit_complete_turns(prefix, tokenizer, sequence_length, chat_template)
        if fitted is not None:
            yield fitted


def rejection_reasons(
    messages: list[dict[str, str]],
    *,
    require_valid_python: bool,
    policy: dict[str, Any],
) -> list[str]:
    audit = audit_row({"messages": messages})
    reasons: list[str] = []
    if audit.assistant_chars < int(policy["min_assistant_chars"]):
        reasons.append("assistant_too_short")
    if policy.get("reject_refusals") and audit.refusal:
        reasons.append("refusal")
    if policy.get("reject_repeated_lines") and audit.repeated_lines:
        reasons.append("repeated_lines")
    if policy.get("reject_template_artifacts") and audit.template_artifact:
        reasons.append("template_artifact")
    if require_valid_python and audit.python_valid is not True:
        reasons.append("invalid_or_missing_python")
    return reasons


def conversation_digest(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recipe",
        type=Path,
        default=Path("configs/tr_hash_200m_clean_sft_v2.json"),
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision", help="Override every pinned source revision")
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--max-scan-multiplier", type=int, default=30)
    parser.add_argument(
        "--smoke-train-per-source",
        type=int,
        default=0,
        help="Override every source target and keep one eval row for a cheap schema smoke test.",
    )
    args = parser.parse_args()

    recipe = load_recipe(args.recipe)
    tokenizer = Tokenizer.load(str(args.tokenizer))
    eos_text = tokenizer.decode([tokenizer.eos_token_id], skip_special_tokens=False)
    if tokenizer.encode(eos_text, add_special_tokens=False) != [tokenizer.eos_token_id]:
        raise ValueError("tokenizer EOS text does not round-trip to its EOS token ID")
    chat_template = align_chat_template_eos(
        default_chat_template(),
        eos_token=eos_text,
    )
    sequence_length = int(recipe["sequence_length"])
    quality_policy = dict(recipe["quality_policy"])
    seed = int(recipe["seed"])
    eval_fraction = float(recipe["eval_fraction"])
    min_eval = int(recipe["min_eval_per_source"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial_train = args.output_dir / "train.jsonl.partial"
    partial_eval = args.output_dir / "eval.jsonl.partial"
    train_path = args.output_dir / "train.jsonl"
    eval_path = args.output_dir / "eval.jsonl"
    seen: set[str] = set()
    source_report: dict[str, Any] = {}

    with partial_train.open("w", encoding="utf-8") as train_handle, partial_eval.open(
        "w", encoding="utf-8"
    ) as eval_handle:
        for source_index, source in enumerate(recipe["sources"]):
            source_name = str(source["name"])
            split = str(source["split"])
            revision = args.revision or str(source["revision"])
            train_target = int(source["train_target"])
            if args.smoke_train_per_source > 0:
                train_target = min(train_target, args.smoke_train_per_source)
                eval_target = 1
            else:
                eval_target = max(min_eval, round(train_target * eval_fraction))
            required = train_target + eval_target
            stream = load_dataset(
                source["dataset"],
                source.get("config"),
                split=split,
                revision=revision,
                streaming=True,
            ).shuffle(seed=seed + source_index, buffer_size=args.shuffle_buffer)
            scanned = kept_train = kept_eval = 0
            rejected: Counter[str] = Counter()
            for row in stream:
                scanned += 1
                messages = normalize_source_row(row, adapter=str(source["adapter"]))
                if messages is None:
                    rejected["invalid_messages"] += 1
                    continue
                messages = strip_training_artifacts(messages)
                examples = list(
                    assistant_turn_examples(
                        messages,
                        expand=bool(source.get("expand_assistant_turns", False)),
                        tokenizer=tokenizer,
                        sequence_length=sequence_length,
                        chat_template=chat_template,
                    )
                )
                if not examples:
                    rejected["does_not_fit_complete"] += 1
                for example in examples:
                    effective_policy = {
                        **quality_policy,
                        **(
                            {"min_assistant_chars": int(source["min_assistant_chars"])}
                            if "min_assistant_chars" in source
                            else {}
                        ),
                    }
                    reasons = rejection_reasons(
                        example,
                        require_valid_python=bool(source.get("require_valid_python", False)),
                        policy=effective_policy,
                    )
                    if reasons:
                        rejected.update(reasons)
                        continue
                    digest = conversation_digest(example)
                    if digest in seen:
                        rejected["duplicate"] += 1
                        continue
                    seen.add(digest)
                    record = json.dumps(
                        {
                            "messages": example,
                            "source": source_name,
                            "capability": source["capability"],
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ) + "\n"
                    if kept_eval < eval_target:
                        eval_handle.write(record)
                        kept_eval += 1
                    elif kept_train < train_target:
                        train_handle.write(record)
                        kept_train += 1
                    if kept_train == train_target and kept_eval == eval_target:
                        break
                if kept_train == train_target and kept_eval == eval_target:
                    break
                if scanned >= required * args.max_scan_multiplier:
                    break
            if kept_train != train_target or kept_eval != eval_target:
                raise RuntimeError(
                    f"{source_name}: requested train={train_target} eval={eval_target}, "
                    f"kept train={kept_train} eval={kept_eval}, scanned={scanned}, "
                    f"rejected={dict(rejected)}"
                )
            source_report[source_name] = {
                "dataset": source["dataset"],
                "config": source.get("config"),
                "split": split,
                "revision": revision,
                "license": source["license"],
                "adapter": source["adapter"],
                "capability": source["capability"],
                "train": kept_train,
                "eval": kept_eval,
                "scanned": scanned,
                "rejected": dict(sorted(rejected.items())),
            }
            print(
                json.dumps({source_name: source_report[source_name]}, sort_keys=True),
                flush=True,
            )

    partial_train.replace(train_path)
    partial_eval.replace(eval_path)
    manifest = {
        "schema_version": 2,
        "name": recipe["name"],
        "recipe_sha256": sha256(args.recipe),
        "seed": seed,
        "sequence_length": sequence_length,
        "max_train_examples": int(recipe["max_train_examples"]),
        "smoke_test": args.smoke_train_per_source > 0,
        "assistant_supervision": "one_complete_assistant_turn_per_example",
        "chat_template_id": chat_template["id"],
        "chat_template_eos_token": chat_template["eos_token"],
        "quality_policy": quality_policy,
        "excluded_sources": recipe["excluded_sources"],
        "sources": source_report,
        "train_examples": sum(item["train"] for item in source_report.values()),
        "eval_examples": sum(item["eval"] for item in source_report.values()),
        "train_sha256": sha256(train_path),
        "eval_sha256": sha256(eval_path),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
