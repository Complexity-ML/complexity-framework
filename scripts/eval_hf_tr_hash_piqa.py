#!/usr/bin/env python3
"""Evaluate a local TR-HASH HF-style bundle on full PIQA."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer

from integrations.transformers.tr_hash_moe.configuration_tr_hash_moe import TRHashConfig
from integrations.transformers.tr_hash_moe.modeling_tr_hash_moe import TRHashForCausalLM
from scripts.eval_torch_piqa import Choice, load_piqa


def encode_choices(tokenizer: Tokenizer, examples: list[dict], max_length: int) -> list[Choice]:
    choices = []
    for example_index, example in enumerate(examples):
        context_ids = tokenizer.encode(example["goal"], add_special_tokens=False).ids
        for choice_index, solution in enumerate(example["solutions"]):
            continuation_ids = tokenizer.encode(
                " " + str(solution).lstrip(), add_special_tokens=False
            ).ids
            ids = context_ids + continuation_ids
            if len(ids) > max_length:
                removed = len(ids) - max_length
                ids = ids[removed:]
                completion_start = len(context_ids) - removed
            else:
                completion_start = len(context_ids)
            if completion_start < 1 or not continuation_ids:
                raise ValueError(f"invalid scoring boundary for PIQA example {example_index}")
            choices.append(Choice(example_index, choice_index, tuple(ids), completion_start))
    return sorted(choices, key=lambda item: len(item.ids))


def load_model(bundle: Path, device: torch.device) -> TRHashForCausalLM:
    raw = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
    model = TRHashForCausalLM(TRHashConfig(**raw))
    model.load_state_dict(load_file(bundle / "model.safetensors"), strict=True)
    return model.to(device=device, dtype=torch.float32).eval()


def evaluate(
    model: TRHashForCausalLM,
    tokenizer: Tokenizer,
    examples: list[dict],
    *,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> tuple[int, int, float]:
    encoded = encode_choices(tokenizer, examples, max_length)
    scores: list[list[tuple[float, float] | None]] = [[None, None] for _ in examples]
    pad_id = tokenizer.token_to_id("<pad>") or tokenizer.token_to_id("</s>") or 0
    started = time.monotonic()
    with torch.inference_mode():
        for offset in range(0, len(encoded), batch_size):
            batch = encoded[offset : offset + batch_size]
            width = max(len(item.ids) for item in batch)
            tokens = torch.full((len(batch), width), pad_id, dtype=torch.long, device=device)
            for row, item in enumerate(batch):
                tokens[row, : len(item.ids)] = torch.tensor(item.ids, device=device)
            logits = model(input_ids=tokens).logits[:, :-1]
            labels = tokens[:, 1:]
            selected = torch.gather(logits, -1, labels[..., None]).squeeze(-1).float()
            log_probs = selected - torch.logsumexp(logits.float(), dim=-1)
            positions = torch.arange(1, width, device=device)[None, :]
            starts = torch.tensor([item.completion_start for item in batch], device=device)[:, None]
            lengths = torch.tensor([len(item.ids) for item in batch], device=device)[:, None]
            mask = (positions >= starts) & (positions < lengths)
            totals = (log_probs * mask).sum(dim=1)
            normalized = totals / mask.sum(dim=1)
            for item, total, norm in zip(batch, totals.tolist(), normalized.tolist(), strict=True):
                scores[item.example_index][item.choice_index] = (total, norm)
            completed = min(offset + len(batch), len(encoded))
            if completed == len(encoded) or completed % 512 < batch_size:
                rate = completed / max(time.monotonic() - started, 1e-9)
                print(
                    f"scored {completed:,}/{len(encoded):,} choices ({rate:.1f}/s)",
                    flush=True,
                )

    correct = 0
    correct_norm = 0
    for example, choice_scores in zip(examples, scores, strict=True):
        if any(score is None for score in choice_scores):
            raise RuntimeError("one or more PIQA choices were not scored")
        resolved = [score for score in choice_scores if score is not None]
        correct += max(range(2), key=lambda index: resolved[index][0]) == example["answer"]
        correct_norm += max(range(2), key=lambda index: resolved[index][1]) == example["answer"]
    return correct, correct_norm, time.monotonic() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.bundle, device)
    tokenizer = Tokenizer.from_file(str(args.bundle / "tokenizer.json"))
    examples = load_piqa(args.probe)
    correct, correct_norm, elapsed = evaluate(
        model,
        tokenizer,
        examples,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    report = {
        "checkpoint": str(args.bundle.resolve()),
        "zero_shot": True,
        "scoring": "causal_choice_loglikelihood",
        "chat_template_applied": False,
        "dtype": "float32",
        "device": args.device,
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
