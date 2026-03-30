"""
Evaluate Complexity-Deep models on standard benchmarks (ARC-Easy, HellaSwag).

Zero-shot evaluation using log-likelihood scoring — no SFT needed.
Loads models directly via the complexity framework.

Usage:
    python scripts/eval_benchmarks.py --checkpoint checkpoints/run2-iso-shared/final
    python scripts/eval_benchmarks.py --checkpoint checkpoints/run1-dense/final
    python scripts/eval_benchmarks.py --checkpoint checkpoints/run2-iso-shared/final --tasks arc_easy
    python scripts/eval_benchmarks.py --checkpoint checkpoints/run2-iso-shared/final --tasks hellaswag

Complexity-ML / INL -- 2026
"""

import argparse
import json
import math
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from safetensors.torch import load_file

from complexity.config import ModelConfig
from complexity.models import ComplexityModel


def load_model(checkpoint_dir, device="cuda"):
    """Load model from checkpoint directory."""
    config = ModelConfig.load(os.path.join(checkpoint_dir, "model_config.yaml"))
    model = ComplexityModel(config)
    state = load_file(os.path.join(checkpoint_dir, "model.safetensors"))
    model.load_state_dict(state, strict=False)
    model.eval()
    model.to(device)
    return model, config


def score_choices_batched(model, tokenizer, context, choices, device="cuda", max_len=2048):
    """Score all choices in a single batched forward pass.

    Pads all (context + choice) sequences to the same length and runs
    one forward pass instead of N separate ones.

    Returns list of length-normalized log-likelihoods.
    """
    ctx_ids = tokenizer.encode(context)

    all_ids = []
    comp_starts = []
    comp_lengths = []
    for choice in choices:
        comp_ids = tokenizer.encode(choice)
        seq = (ctx_ids + comp_ids)[:max_len]
        all_ids.append(seq)
        comp_starts.append(len(ctx_ids))
        comp_lengths.append(len(seq) - len(ctx_ids))

    # Pad to max length in batch
    max_seq = max(len(s) for s in all_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    padded = [s + [pad_id] * (max_seq - len(s)) for s in all_ids]
    lengths = [len(s) for s in all_ids]

    input_tensor = torch.tensor(padded, device=device)

    with torch.no_grad():
        out = model(input_tensor)
        h = out["last_hidden_state"] if isinstance(out, dict) else out
        logits = (h @ model.embed_tokens.weight.T).float()

    log_probs = F.log_softmax(logits, dim=-1)

    scores = []
    for b in range(len(choices)):
        start = comp_starts[b]
        end = lengths[b]
        n_comp = comp_lengths[b]
        if n_comp <= 0 or start >= end:
            scores.append(float("-inf"))
            continue
        total = 0.0
        for i in range(start, end):
            total += log_probs[b, i - 1, all_ids[b][i]].item()
        scores.append(total / n_comp)

    return scores


# ── ARC-Easy ──

def eval_arc_easy(model, tokenizer, device="cuda", split="test"):
    """Evaluate on ARC-Easy (zero-shot, log-likelihood, batched)."""
    dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split)

    correct = 0
    total = 0

    for item in tqdm(dataset, desc="ARC-Easy"):
        question = item["question"]
        choices = item["choices"]
        answer_key = item["answerKey"]

        labels = choices["label"]
        texts = choices["text"]
        answer_idx = labels.index(answer_key)

        context = f"Question: {question}\nAnswer:"
        completions = [f" {text}" for text in texts]

        scores = score_choices_batched(model, tokenizer, context, completions, device)

        pred = max(range(len(scores)), key=lambda i: scores[i])
        if pred == answer_idx:
            correct += 1
        total += 1

    acc = correct / total
    return {"acc": acc, "correct": correct, "total": total}


# ── HellaSwag ──

def eval_hellaswag(model, tokenizer, device="cuda", split="validation"):
    """Evaluate on HellaSwag (zero-shot, log-likelihood, batched)."""
    dataset = load_dataset("Rowan/hellaswag", split=split)

    correct = 0
    total = 0

    for item in tqdm(dataset, desc="HellaSwag"):
        context = item["ctx"]
        endings = item["endings"]
        label = int(item["label"])

        scores = score_choices_batched(model, tokenizer, context, endings, device)

        pred = max(range(len(scores)), key=lambda i: scores[i])
        if pred == label:
            correct += 1
        total += 1

    acc = correct / total
    return {"acc": acc, "correct": correct, "total": total}


# ── Main ──

BENCHMARKS = {
    "arc_easy": eval_arc_easy,
    "hellaswag": eval_hellaswag,
}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Complexity-Deep on benchmarks")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tasks", type=str, default="arc_easy,hellaswag")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tokenizer", type=str, default="./tokenizer")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint, args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.1f}M params")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
    print(f"Tokenizer: {len(tokenizer)} tokens")

    tasks = args.tasks.split(",")
    results = {}

    for task in tasks:
        if task not in BENCHMARKS:
            print(f"Unknown task: {task} (available: {list(BENCHMARKS.keys())})")
            continue

        print(f"\n{'='*50}")
        print(f"Evaluating: {task}")
        print(f"{'='*50}")

        result = BENCHMARKS[task](model, tokenizer, args.device)
        results[task] = result
        print(f"  Accuracy: {result['acc']:.4f} ({result['correct']}/{result['total']})")

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    for task, result in results.items():
        print(f"  {task:15s}: {result['acc']:.4f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
