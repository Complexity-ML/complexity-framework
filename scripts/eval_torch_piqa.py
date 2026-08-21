#!/usr/bin/env python3
"""Evaluate a native TR-HASH PyTorch checkpoint on the full PIQA validation set."""

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
class Choice:
    example_index: int
    choice_index: int
    ids: tuple[int, ...]
    completion_start: int


def load_piqa(probe: Path) -> list[dict]:
    inputs = (probe / "dev.jsonl").read_text(encoding="utf-8").splitlines()
    labels = (probe / "dev-labels.lst").read_text(encoding="utf-8").splitlines()
    if len(inputs) != len(labels):
        raise ValueError("PIQA inputs and labels have different lengths")
    return [
        {
            "id": index,
            "goal": row["goal"],
            "solutions": (row["sol1"], row["sol2"]),
            "answer": int(label),
        }
        for index, (encoded, label) in enumerate(zip(inputs, labels, strict=True))
        for row in (json.loads(encoded),)
    ]


def encode_choices(tokenizer: Tokenizer, examples: list[dict], max_length: int) -> list[Choice]:
    choices = []
    for example_index, example in enumerate(examples):
        context_ids = tokenizer.encode(example["goal"], add_special_tokens=False)
        for choice_index, solution in enumerate(example["solutions"]):
            continuation_ids = tokenizer.encode(
                " " + str(solution).lstrip(), add_special_tokens=False
            )
            ids = context_ids + continuation_ids
            if len(ids) > max_length:
                removed = len(ids) - max_length
                ids = ids[removed:]
                completion_start = len(context_ids) - removed
            else:
                completion_start = len(context_ids)
            if completion_start < 1 or not continuation_ids:
                raise ValueError(f"invalid scoring boundary for PIQA example {example_index}")
            choices.append(
                Choice(example_index, choice_index, tuple(ids), completion_start)
            )
    return sorted(choices, key=lambda item: len(item.ids))


def evaluate(
    model: ComplexityModel,
    tokenizer: Tokenizer,
    examples: list[dict],
    *,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> tuple[int, int]:
    encoded = encode_choices(tokenizer, examples, max_length)
    scores: list[list[tuple[float, float] | None]] = [[None, None] for _ in examples]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    started = time.monotonic()
    with torch.inference_mode():
        for offset in range(0, len(encoded), batch_size):
            batch = encoded[offset : offset + batch_size]
            width = max(len(item.ids) for item in batch)
            tokens = torch.full(
                (len(batch), width), pad_id, dtype=torch.long, device=device
            )
            for row, item in enumerate(batch):
                tokens[row, : len(item.ids)] = torch.tensor(item.ids, device=device)
            logits = model(tokens)["logits"][:, :-1]
            labels = tokens[:, 1:]
            selected = torch.gather(logits, -1, labels[..., None]).squeeze(-1).float()
            normalizer = torch.logsumexp(logits.float(), dim=-1)
            log_probs = selected - normalizer
            positions = torch.arange(1, width, device=device)[None, :]
            starts = torch.tensor(
                [item.completion_start for item in batch], device=device
            )[:, None]
            lengths = torch.tensor([len(item.ids) for item in batch], device=device)[:, None]
            mask = (positions >= starts) & (positions < lengths)
            totals = (log_probs * mask).sum(dim=1)
            counts = mask.sum(dim=1)
            normalized = totals / counts
            for item, total, norm in zip(
                batch, totals.tolist(), normalized.tolist(), strict=True
            ):
                scores[item.example_index][item.choice_index] = (total, norm)
            completed = min(offset + len(batch), len(encoded))
            if completed == len(encoded) or completed % max(batch_size, 512) < batch_size:
                rate = completed / max(time.monotonic() - started, 1e-9)
                print(f"scored {completed:,}/{len(encoded):,} choices ({rate:.1f}/s)", flush=True)

    correct = 0
    correct_norm = 0
    for example, choice_scores in zip(examples, scores, strict=True):
        if any(score is None for score in choice_scores):
            raise RuntimeError("one or more PIQA choices were not scored")
        resolved = [score for score in choice_scores if score is not None]
        correct += max(range(2), key=lambda index: resolved[index][0]) == example["answer"]
        correct_norm += max(range(2), key=lambda index: resolved[index][1]) == example["answer"]
    return correct, correct_norm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    args = parser.parse_args()

    configure_torch_acceleration(kernel_policy=True)
    device = torch.device("cuda")
    checkpoint_path, state = load_checkpoint_state(args.checkpoint, map_location="cpu")
    config = checkpoint_config(state)
    config.use_custom_kernels = True
    model = ComplexityModel(config)
    load_model_state_compat(model, state["model"])
    dtype = getattr(torch, args.dtype)
    model.to(device=device, dtype=dtype).eval()
    tokenizer = Tokenizer.load(str(args.tokenizer))
    examples = load_piqa(args.probe)

    started = time.monotonic()
    correct, correct_norm = evaluate(
        model,
        tokenizer,
        examples,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    elapsed = time.monotonic() - started
    report = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_step": int(state.get("step", 0)),
        "sft_matched_eval_loss": state.get("sft_matched_eval_loss"),
        "zero_shot": True,
        "scoring": "causal_choice_loglikelihood",
        "chat_template_applied": False,
        "dtype": args.dtype,
        "custom_triton": True,
        "benchmarks": {
            "piqa": {
                "examples": len(examples),
                "correct": correct,
                "acc": correct / len(examples),
                "correct_norm": correct_norm,
                "acc_norm": correct_norm / len(examples),
                "elapsed_seconds": round(elapsed, 3),
            }
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
