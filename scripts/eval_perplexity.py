"""
Evaluate perplexity for ablation study checkpoints.

Compares all 4 runs on the same held-out test set (FineWeb-Edu).
Outputs a table + CSV for the paper.

Usage:
    python scripts/eval_perplexity.py
    python scripts/eval_perplexity.py --checkpoint-dir ./checkpoints/ablation-150m
    python scripts/eval_perplexity.py --run 2  # Eval only Run 2
    python scripts/eval_perplexity.py --num-batches 500  # More batches for precision

INL - 2025
"""

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from torch.utils.data import DataLoader, IterableDataset
import torch
import torch.nn as nn
import argparse
import os
import json
import math
import csv

from complexity.config import ModelConfig
from complexity.models import ComplexityModel


# ── Run metadata ──────────────────────────────────────────────────────────

RUN_INFO = {
    1: ("run1-dense",  "Dense SwiGLU (baseline)"),
    2: ("run2-full",   "Token-Routed + Mu + INL"),
    3: ("run3-no-mu",  "Token-Routed + INL (no Mu)"),
    4: ("run4-inl",    "INL integer-first (i64)"),
}


# ── Test dataset ──────────────────────────────────────────────────────────

class FineWebTestDataset(IterableDataset):
    """Held-out test split from FineWeb-Edu for perplexity evaluation."""

    def __init__(self, tokenizer, max_length=2048, skip_train=10000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.skip_train = skip_train
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            split="train",
            streaming=True,
        )

    def __iter__(self):
        buffer = []
        # Skip first N examples (used for training) to get held-out data
        for i, example in enumerate(self.dataset):
            if i < self.skip_train:
                continue
            text = example.get("text", "")
            if not text:
                continue
            tokens = self.tokenizer.encode(text)
            buffer.extend(tokens)

            while len(buffer) >= self.max_length + 1:
                chunk = buffer[:self.max_length + 1]
                buffer = buffer[self.max_length:]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}


# ── Perplexity computation ────────────────────────────────────────────────

@torch.no_grad()
def compute_perplexity(model, dataloader, device, num_batches=200):
    """Compute perplexity over num_batches batches."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, :shift_logits.size(1)].contiguous()

        # Compute per-token loss
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='sum',
        )

        total_loss += loss.item()
        total_tokens += shift_labels.numel()

        if (batch_idx + 1) % 50 == 0:
            ppl_so_far = math.exp(total_loss / total_tokens)
            print(f"  Batch {batch_idx+1}/{num_batches} | PPL so far: {ppl_so_far:.2f}")

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity, avg_loss


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate perplexity for ablation runs")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/ablation-150m")
    parser.add_argument("--tokenizer", type=str, default="./tokenizer")
    parser.add_argument("--run", type=str, default="all",
                        help="Run ID: 1, 2, 3, 4, or 'all'")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=200,
                        help="Number of test batches (200 × 4 × 2048 ≈ 1.6M tokens)")
    parser.add_argument("--output", type=str, default="./eval_results.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)

    # Test dataset
    print("Loading test dataset (skipping training data)...")
    test_dataset = FineWebTestDataset(tokenizer=tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0)

    # Which runs to evaluate
    if args.run == "all":
        run_ids = [1, 2, 3, 4]
    else:
        run_ids = [int(args.run)]

    # Results
    results = []

    for run_id in run_ids:
        dirname, description = RUN_INFO[run_id]
        model_path = os.path.join(args.checkpoint_dir, dirname, "final")

        if not os.path.exists(model_path):
            print(f"\nSkipping Run {run_id} ({description}): {model_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"  Run {run_id}: {description}")
        print(f"  Loading from: {model_path}")
        print(f"{'='*60}")

        # Load model
        model = ComplexityModel.from_pretrained(model_path, device=str(device))
        num_params = model.num_parameters()
        print(f"  Parameters: {num_params:,} ({num_params/1e6:.1f}M)")

        # Compute perplexity
        perplexity, avg_loss = compute_perplexity(
            model, test_dataloader, device, num_batches=args.num_batches,
        )

        print(f"\n  >> Perplexity: {perplexity:.2f}")
        print(f"  >> Avg Loss:   {avg_loss:.4f}")

        results.append({
            "run": run_id,
            "name": dirname,
            "description": description,
            "params_M": round(num_params / 1e6, 1),
            "perplexity": round(perplexity, 2),
            "avg_loss": round(avg_loss, 4),
        })

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Print summary table
    if results:
        print(f"\n{'='*60}")
        print(f"  RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"{'Run':<5} {'Description':<32} {'Params':<10} {'PPL':<10} {'Loss':<10}")
        print("-" * 67)
        for r in results:
            print(f"{r['run']:<5} {r['description']:<32} {r['params_M']:<10} {r['perplexity']:<10} {r['avg_loss']:<10}")

        # Save CSV
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {args.output}")

        # Save JSON too
        json_path = args.output.replace(".csv", ".json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {json_path}")


if __name__ == "__main__":
    main()
