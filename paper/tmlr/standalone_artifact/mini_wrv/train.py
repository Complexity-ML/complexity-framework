"""Standalone training/evaluation CLI for the review model."""

from __future__ import annotations

import argparse
import csv
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from .data import FineWebTokenStream, TiktokenO200k, load_fineweb_parquet
from .model import ModelConfig, TinyLanguageModel


def load_config(path: str) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict) or "model" not in data or "training" not in data:
        raise ValueError("config must contain model and training mappings")
    return data


def evaluate(
    model: TinyLanguageModel,
    loader: DataLoader[torch.Tensor],
    *,
    batches: int,
    device: torch.device,
    bf16: bool,
    chunk_tokens: int,
) -> float:
    model.eval()
    losses: list[float] = []
    iterator = iter(loader)
    context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and bf16 else nullcontext()
    with torch.inference_mode(), context:
        for _ in range(batches):
            tokens = next(iterator).to(device)
            losses.append(float(model.loss(tokens, chunk_tokens=chunk_tokens)))
    model.train()
    return sum(losses) / len(losses)


def run(config_path: str, parquet_path: str, output_dir: str) -> None:
    raw = load_config(config_path)
    model_config = ModelConfig(**raw["model"])
    training = raw["training"]
    seed = int(training["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16 = bool(training.get("bf16", True))
    tokenizer = TiktokenO200k()
    documents = load_fineweb_parquet(parquet_path)
    sequence_length = int(training["sequence_length"])
    batch_size = int(training["batch_size"])
    train_loader = DataLoader(
        FineWebTokenStream(documents, tokenizer, sequence_length, split="train"),
        batch_size=batch_size,
    )
    eval_documents = load_fineweb_parquet(parquet_path)
    eval_loader = DataLoader(
        FineWebTokenStream(eval_documents, tokenizer, sequence_length, split="eval"),
        batch_size=batch_size,
    )
    model = TinyLanguageModel(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    steps = int(training["steps"])
    warmup_steps = int(training.get("warmup_steps", 50))
    eval_every = int(training.get("eval_every", 250))
    eval_batches = int(training.get("eval_batches", 32))
    chunk_tokens = int(training.get("loss_chunk_tokens", 512))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.csv"
    context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and bf16 else nullcontext()
    started = time.perf_counter()
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("step", "train_loss", "eval_loss", "lr", "tok_s"))
        writer.writeheader()
        iterator = iter(train_loader)
        for step in range(1, steps + 1):
            tokens = next(iterator).to(device)
            optimizer.zero_grad(set_to_none=True)
            with context:
                loss = model.loss(tokens, chunk_tokens=chunk_tokens)
            loss.backward()
            optimizer.step()
            progress = step / steps
            warmup = min(1.0, step / max(1, warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = float(training["learning_rate"]) * warmup * cosine
            for group in optimizer.param_groups:
                group["lr"] = lr
            elapsed = time.perf_counter() - started
            tok_s = step * batch_size * sequence_length / elapsed
            eval_loss = ""
            if step % eval_every == 0 or step == steps:
                eval_loss = evaluate(
                    model,
                    eval_loader,
                    batches=eval_batches,
                    device=device,
                    bf16=bf16,
                    chunk_tokens=chunk_tokens,
                )
            writer.writerow(
                {"step": step, "train_loss": float(loss), "eval_loss": eval_loss, "lr": lr, "tok_s": tok_s}
            )
            handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fineweb-parquet", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.config, args.fineweb_parquet, args.output)


if __name__ == "__main__":
    main()
