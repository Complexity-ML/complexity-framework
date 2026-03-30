"""
Train Dense Baseline 1.5B — Llama-style for paper comparison.

Same dimensions as complexity-deep but WITHOUT Token-Routed MLP,
INL Dynamics, or Mu-Guidance. Uses the framework Trainer API.

Usage:
    pip install complexity-framework
    python scripts/train_dense_baseline.py
    python scripts/train_dense_baseline.py --resume checkpoints/dense-baseline/step_50000

INL - 2025
"""

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from torch.utils.data import DataLoader, IterableDataset
import torch
import argparse
import os

from complexity.config import get_preset
from complexity.models import ComplexityModel
from complexity.training import Trainer, TrainingConfig


class FineWebStreamingDataset(IterableDataset):
    """Streaming tokenized chunks from FineWeb-Edu."""

    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            split="train",
            streaming=True,
        )

    def __iter__(self):
        buffer = []
        for example in self.dataset:
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


def main():
    parser = argparse.ArgumentParser(description="Train Dense Baseline 1.5B")
    parser.add_argument("--tokenizer", type=str, default="./tokenizer")
    parser.add_argument("--max-steps", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=5000)
    parser.add_argument("--save-steps", type=int, default=50000)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/dense-baseline")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    # Tokenizer
    if not os.path.exists(args.tokenizer):
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)

    # Model — dense Llama 1.5B (same dims as complexity-deep, no MoE)
    config = get_preset("llama-1.5b")
    config.vocab_size = len(tokenizer)
    model = ComplexityModel(config)
    print(f"Dense Llama 1.5B: {model.num_parameters():,} params")

    # Data
    dataset = FineWebStreamingDataset(tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            num_workers=args.num_workers, pin_memory=True)

    # Trainer (handles FSDP, AMP, cosine LR, checkpointing, grad clip)
    train_config = TrainingConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        precision="bf16",
        save_steps=args.save_steps,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume,
    )

    trainer = Trainer(model=model, config=train_config, train_dataloader=dataloader)
    summary = trainer.train()
    print(f"Training complete: {summary}")


if __name__ == "__main__":
    main()
