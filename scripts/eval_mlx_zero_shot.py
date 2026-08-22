#!/usr/bin/env python3
"""Zero-shot multiple-choice evaluation for a TR-HASH MLX bundle.

Scores continuations by causal log-likelihood; it does not generate text and
does not apply the chat template. This matches the standard zero-shot setup
for ARC-Easy, HellaSwag, MMLU, and PIQA more closely than sampled chat
generation.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import mlx.core as mx
from datasets import DownloadManager, load_dataset
from mlx_lm.utils import load, load_model, load_tokenizer


@dataclass(frozen=True)
class Example:
    benchmark: str
    example_id: str
    context: str
    continuations: tuple[str, ...]
    answer: int
    subject: str | None = None


@dataclass(frozen=True)
class EncodedChoice:
    example_index: int
    choice_index: int
    ids: tuple[int, ...]
    completion_start: int
    completion_length: int


def _load_bundle(model_dir: Path):
    try:
        return load(model_dir.as_posix())
    except ValueError as error:
        if "mlp.token_to_expert" not in str(error):
            raise
        model, _config = load_model(model_dir, strict=False)
        tokenizer = load_tokenizer(model_dir)
        return model, tokenizer


def _limit(rows: Iterable[Example], maximum: int | None) -> list[Example]:
    if maximum is None:
        return list(rows)
    result: list[Example] = []
    for row in rows:
        if len(result) >= maximum:
            break
        result.append(row)
    return result


def load_hellaswag(maximum: int | None) -> list[Example]:
    dataset = load_dataset("Rowan/hellaswag", split="validation")
    return _limit(
        (
            Example(
                benchmark="hellaswag",
                example_id=str(row["ind"]),
                context=str(row["ctx"]),
                continuations=tuple(" " + str(value).lstrip() for value in row["endings"]),
                answer=int(row["label"]),
                subject=str(row.get("activity_label", "")) or None,
            )
            for row in dataset
        ),
        maximum,
    )


def load_arc_easy(maximum: int | None) -> list[Example]:
    """Load ARC-Easy using the lm-evaluation-harness zero-shot contract."""

    dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")

    def examples() -> Iterable[Example]:
        for index, row in enumerate(dataset):
            labels = [str(value) for value in row["choices"]["label"]]
            answer_key = str(row["answerKey"])
            try:
                answer = labels.index(answer_key)
            except ValueError as error:
                raise ValueError(
                    f"ARC-Easy answer key {answer_key!r} is absent from "
                    f"choice labels {labels!r} for example {row.get('id', index)!r}"
                ) from error
            yield Example(
                benchmark="arc_easy",
                example_id=str(row.get("id", index)),
                context=f"Question: {row['question']}\nAnswer:",
                continuations=tuple(
                    " " + str(value).lstrip() for value in row["choices"]["text"]
                ),
                answer=answer,
            )

    return _limit(examples(), maximum)


def load_mmlu(maximum: int | None) -> list[Example]:
    dataset = load_dataset("cais/mmlu", "all", split="test")
    letters = "ABCD"

    def examples() -> Iterable[Example]:
        for index, row in enumerate(dataset):
            choices = tuple(str(value) for value in row["choices"])
            context = f"Question: {row['question']}\n"
            context += "".join(
                f"{letter}. {choice}\n"
                for letter, choice in zip(letters, choices, strict=True)
            )
            context += "Answer:"
            yield Example(
                benchmark="mmlu",
                example_id=str(index),
                context=context,
                continuations=tuple(f" {letter}" for letter in letters),
                answer=int(row["answer"]),
                subject=str(row["subject"]),
            )

    return _limit(examples(), maximum)


def load_piqa(maximum: int | None) -> list[Example]:
    # datasets>=4 no longer executes the legacy ybisk/piqa loader. Read the
    # same official AI2 archive and validation files that loader references.
    archive = DownloadManager().download_and_extract(
        "https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/"
        "physicaliqa-train-dev.zip"
    )
    root = Path(archive) / "physicaliqa-train-dev"
    inputs = (root / "dev.jsonl").read_text().splitlines()
    labels = (root / "dev-labels.lst").read_text().splitlines()

    def examples() -> Iterable[Example]:
        for index, (encoded, label) in enumerate(zip(inputs, labels, strict=True)):
            row = json.loads(encoded)
            yield Example(
                benchmark="piqa",
                example_id=str(index),
                context=str(row["goal"]),
                continuations=(
                    " " + str(row["sol1"]).lstrip(),
                    " " + str(row["sol2"]).lstrip(),
                ),
                answer=int(label),
            )

    return _limit(examples(), maximum)


LOADERS = {
    "arc_easy": load_arc_easy,
    "hellaswag": load_hellaswag,
    "mmlu": load_mmlu,
    "piqa": load_piqa,
}


def _encode_choices(
    tokenizer,
    examples: list[Example],
    max_length: int,
) -> list[EncodedChoice]:
    choices: list[EncodedChoice] = []
    for example_index, example in enumerate(examples):
        context_ids = tokenizer.encode(example.context, add_special_tokens=False)
        for choice_index, continuation in enumerate(example.continuations):
            continuation_ids = tokenizer.encode(continuation, add_special_tokens=False)
            if not continuation_ids:
                raise ValueError(
                    f"empty continuation in {example.benchmark}:{example.example_id}"
                )
            ids = context_ids + continuation_ids
            if len(ids) > max_length:
                removed = len(ids) - max_length
                ids = ids[removed:]
                completion_start = len(context_ids) - removed
            else:
                completion_start = len(context_ids)
            if completion_start < 1:
                raise ValueError(
                    f"context truncated past scoring boundary in "
                    f"{example.benchmark}:{example.example_id}"
                )
            choices.append(
                EncodedChoice(
                    example_index=example_index,
                    choice_index=choice_index,
                    ids=tuple(ids),
                    completion_start=completion_start,
                    completion_length=len(ids) - completion_start,
                )
            )
    return choices


def score_examples(
    model,
    tokenizer,
    examples: list[Example],
    *,
    batch_size: int,
    max_length: int,
) -> list[list[dict[str, float]]]:
    encoded = _encode_choices(tokenizer, examples, max_length)
    # Similar lengths in one batch reduce padding and Metal memory traffic.
    encoded.sort(key=lambda row: len(row.ids))
    scores: list[list[dict[str, float] | None]] = [
        [None] * len(example.continuations) for example in examples
    ]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0

    started = time.monotonic()
    for offset in range(0, len(encoded), batch_size):
        batch = encoded[offset : offset + batch_size]
        width = max(len(row.ids) for row in batch)
        token_rows = [
            list(row.ids) + [pad_id] * (width - len(row.ids)) for row in batch
        ]
        tokens = mx.array(token_rows, dtype=mx.int32)
        logits = model(tokens)[:, :-1, :].astype(mx.float32)
        labels = tokens[:, 1:]
        selected = mx.take_along_axis(logits, labels[..., None], axis=-1).squeeze(-1)
        log_probs = selected - mx.logsumexp(logits, axis=-1)
        positions = mx.arange(1, width)[None, :]
        starts = mx.array([row.completion_start for row in batch])[:, None]
        lengths = mx.array([len(row.ids) for row in batch])[:, None]
        mask = (positions >= starts) & (positions < lengths)
        totals = mx.sum(mx.where(mask, log_probs, 0.0), axis=1)
        counts = mx.sum(mask, axis=1)
        normalized = totals / counts
        mx.eval(totals, normalized)

        for row, total, norm in zip(batch, totals.tolist(), normalized.tolist(), strict=True):
            scores[row.example_index][row.choice_index] = {
                "loglikelihood": float(total),
                "loglikelihood_normalized": float(norm),
            }
        completed = min(offset + len(batch), len(encoded))
        if completed == len(encoded) or completed % max(batch_size, 400) < batch_size:
            elapsed = max(time.monotonic() - started, 1e-9)
            print(
                f"  scored {completed:,}/{len(encoded):,} choices "
                f"({completed / elapsed:.1f} choices/s)",
                flush=True,
            )

    if any(value is None for row in scores for value in row):
        raise RuntimeError("one or more benchmark choices were not scored")
    return [[value for value in row if value is not None] for row in scores]


def summarize(examples: list[Example], scores: list[list[dict[str, float]]]) -> dict[str, Any]:
    raw_correct = 0
    normalized_correct = 0
    by_subject: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for example, choice_scores in zip(examples, scores, strict=True):
        raw_prediction = max(
            range(len(choice_scores)), key=lambda index: choice_scores[index]["loglikelihood"]
        )
        normalized_prediction = max(
            range(len(choice_scores)),
            key=lambda index: choice_scores[index]["loglikelihood_normalized"],
        )
        raw_hit = raw_prediction == example.answer
        normalized_hit = normalized_prediction == example.answer
        raw_correct += raw_hit
        normalized_correct += normalized_hit
        if example.subject:
            by_subject[example.subject].append((raw_hit, normalized_hit))

    total = len(examples)
    result: dict[str, Any] = {
        "examples": total,
        "correct": raw_correct,
        "acc": raw_correct / total,
        "correct_norm": normalized_correct,
        "acc_norm": normalized_correct / total,
    }
    if by_subject:
        result["subjects"] = {
            subject: {
                "examples": len(hits),
                "acc": sum(raw for raw, _norm in hits) / len(hits),
                "acc_norm": sum(norm for _raw, norm in hits) / len(hits),
            }
            for subject, hits in sorted(by_subject.items())
        }
        result["macro_acc"] = sum(
            item["acc"] for item in result["subjects"].values()
        ) / len(result["subjects"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument(
        "--tasks",
        default="hellaswag,mmlu,piqa",
        help="Comma-separated subset of arc_easy,hellaswag,mmlu,piqa.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    unknown = sorted(set(tasks) - LOADERS.keys())
    if unknown:
        raise ValueError(f"unknown benchmarks: {unknown}")
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")

    print(f"Loading final MLX bundle: {args.model_dir}", flush=True)
    model, tokenizer = _load_bundle(args.model_dir)
    model.eval()
    mx.eval(model.parameters())
    results: dict[str, Any] = {
        "model": str(args.model_dir.resolve()),
        "zero_shot": True,
        "scoring": "causal_choice_loglikelihood",
        "chat_template_applied": False,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "max_samples": args.max_samples,
        "benchmarks": {},
    }

    for task in tasks:
        print(f"Loading {task}...", flush=True)
        examples = LOADERS[task](args.max_samples)
        print(f"Evaluating {task}: {len(examples):,} examples", flush=True)
        started = time.monotonic()
        scores = score_examples(
            model,
            tokenizer,
            examples,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        result = summarize(examples, scores)
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        results["benchmarks"][task] = result
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        print(
            f"{task}: acc={result['acc']:.4f}, "
            f"acc_norm={result['acc_norm']:.4f}",
            flush=True,
        )

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
