#!/usr/bin/env python3
"""Evaluate a native TR-HASH checkpoint on full ARC by continuation likelihood."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from complexity.models import ComplexityModel
from complexity.tokenizer import Tokenizer
from complexity.utils.device import configure_torch_acceleration
from scripts.sft_500m_32k_tr import (
    checkpoint_config,
    load_checkpoint_state,
    load_model_state_compat,
)


@dataclass(frozen=True)
class ARCExample:
    task: str
    example_id: str
    context: str
    continuations: tuple[str, ...]
    answer: int


@dataclass(frozen=True)
class Choice:
    example_index: int
    choice_index: int
    ids: tuple[int, ...]
    completion_start: int


def load_arc(path: Path, task: str) -> list[ARCExample]:
    """Load the pinned lm-eval ARC documents prepared by prepare_arc_eval_samples."""

    examples = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            doc = row["doc"]
            labels = [str(label) for label in doc["choices"]["label"]]
            answer_key = str(doc["answerKey"])
            if answer_key not in labels:
                raise ValueError(
                    f"answer key {answer_key!r} absent from {labels!r} in {path}:{index}"
                )
            examples.append(
                ARCExample(
                    task=task,
                    example_id=str(doc.get("id", row.get("doc_id", index))),
                    context=f"Question: {doc['question']}\nAnswer:",
                    continuations=tuple(
                        " " + str(choice).lstrip() for choice in doc["choices"]["text"]
                    ),
                    answer=labels.index(answer_key),
                )
            )
    return examples


def encode_choices(
    tokenizer: Tokenizer, examples: list[ARCExample], max_length: int
) -> list[Choice]:
    choices = []
    for example_index, example in enumerate(examples):
        context_ids = tokenizer.encode(example.context, add_special_tokens=False)
        for choice_index, continuation in enumerate(example.continuations):
            continuation_ids = tokenizer.encode(continuation, add_special_tokens=False)
            if not continuation_ids:
                raise ValueError(f"empty continuation for {example.task}:{example.example_id}")
            ids = context_ids + continuation_ids
            removed = max(0, len(ids) - max_length)
            ids = ids[removed:]
            completion_start = len(context_ids) - removed
            if completion_start < 1:
                raise ValueError(
                    f"context truncated past boundary for {example.task}:{example.example_id}"
                )
            choices.append(Choice(example_index, choice_index, tuple(ids), completion_start))
    return sorted(choices, key=lambda choice: len(choice.ids))


def evaluate(
    model: ComplexityModel,
    tokenizer: Tokenizer,
    examples: list[ARCExample],
    *,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> tuple[int, int]:
    encoded = encode_choices(tokenizer, examples, max_length)
    scores: list[list[tuple[float, float] | None]] = [
        [None] * len(example.continuations) for example in examples
    ]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    started = time.monotonic()
    with torch.inference_mode():
        for offset in range(0, len(encoded), batch_size):
            batch = encoded[offset : offset + batch_size]
            width = max(len(choice.ids) for choice in batch)
            tokens = torch.full((len(batch), width), pad_id, dtype=torch.long, device=device)
            for row, choice in enumerate(batch):
                tokens[row, : len(choice.ids)] = torch.tensor(choice.ids, device=device)
            logits = model(tokens)["logits"][:, :-1]
            labels = tokens[:, 1:]
            selected = torch.gather(logits, -1, labels[..., None]).squeeze(-1).float()
            log_probs = selected - torch.logsumexp(logits.float(), dim=-1)
            positions = torch.arange(1, width, device=device)[None, :]
            starts = torch.tensor([choice.completion_start for choice in batch], device=device)[
                :, None
            ]
            lengths = torch.tensor([len(choice.ids) for choice in batch], device=device)[:, None]
            mask = (positions >= starts) & (positions < lengths)
            totals = (log_probs * mask).sum(dim=1)
            normalized = totals / mask.sum(dim=1)
            for choice, total, norm in zip(
                batch, totals.tolist(), normalized.tolist(), strict=True
            ):
                scores[choice.example_index][choice.choice_index] = (total, norm)
            completed = min(offset + len(batch), len(encoded))
            if completed == len(encoded) or completed % max(batch_size, 512) < batch_size:
                rate = completed / max(time.monotonic() - started, 1e-9)
                print(
                    f"scored {completed:,}/{len(encoded):,} {examples[0].task} choices "
                    f"({rate:.1f}/s)",
                    flush=True,
                )

    correct = 0
    correct_norm = 0
    for example, choice_scores in zip(examples, scores, strict=True):
        if any(score is None for score in choice_scores):
            raise RuntimeError("one or more ARC choices were not scored")
        resolved = [score for score in choice_scores if score is not None]
        correct += max(range(len(resolved)), key=lambda index: resolved[index][0]) == example.answer
        correct_norm += (
            max(range(len(resolved)), key=lambda index: resolved[index][1]) == example.answer
        )
    return correct, correct_norm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--arc-easy-samples", type=Path, required=True)
    parser.add_argument("--arc-challenge-samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="float16")
    args = parser.parse_args()

    configure_torch_acceleration(kernel_policy=True)
    device = torch.device("cuda")
    checkpoint_path, state = load_checkpoint_state(args.checkpoint, map_location="cpu")
    config = checkpoint_config(state)
    config.use_custom_kernels = True
    model = ComplexityModel(config)
    load_model_state_compat(model, state["model"])
    model.to(device=device, dtype=getattr(torch, args.dtype)).eval()
    tokenizer = Tokenizer.load(str(args.tokenizer))

    benchmark_paths = {
        "arc_easy": args.arc_easy_samples,
        "arc_challenge": args.arc_challenge_samples,
    }
    report = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_step": int(state.get("step", 0)),
        "zero_shot": True,
        "scoring": "causal_choice_loglikelihood",
        "chat_template_applied": False,
        "dtype": args.dtype,
        "custom_triton": True,
        "benchmarks": {},
    }
    total_examples = total_correct = total_correct_norm = 0
    for task, path in benchmark_paths.items():
        examples = load_arc(path, task)
        started = time.monotonic()
        correct, correct_norm = evaluate(
            model,
            tokenizer,
            examples,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        count = len(examples)
        report["benchmarks"][task] = {
            "examples": count,
            "correct": correct,
            "acc": correct / count,
            "correct_norm": correct_norm,
            "acc_norm": correct_norm / count,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        total_examples += count
        total_correct += correct
        total_correct_norm += correct_norm
    report["combined"] = {
        "examples": total_examples,
        "correct": total_correct,
        "acc": total_correct / total_examples,
        "correct_norm": total_correct_norm,
        "acc_norm": total_correct_norm / total_examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
