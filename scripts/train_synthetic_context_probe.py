#!/usr/bin/env python3
"""Train matched tiny GQA/causal-conv models on synthetic context tasks."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from complexity.config import ModelConfig
from complexity.evaluation.associative_recall import (
    build_associative_recall_batch,
    build_induction_batch,
    score_target_logits,
)
from complexity.experiments.shared_associative_attention import SharedAssociativeAttention
from complexity.models import ComplexityModel


def make_config(
    architecture: str, intermediate_size: int, state_rank: int = 16
) -> ModelConfig:
    model_architecture = (
        "causal_fast_weight_conv"
        if architecture == "shared_associative_attention"
        else architecture
    )
    return ModelConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=intermediate_size,
        max_position_embeddings=256,
        attention_type=model_architecture,
        causal_conv_kernel_size=4,
        causal_conv_dilation_cycle=4,
        causal_state_rank=state_rank,
        mlp_type="swiglu",
        tie_word_embeddings=True,
        use_cache=True,
    )


def build_model(
    config: ModelConfig,
    architecture: str,
    contextual_mix_init: float,
) -> ComplexityModel:
    model = ComplexityModel(config)
    if architecture == "shared_associative_attention":
        shared_context = SharedAssociativeAttention(
            config.hidden_size,
            config.causal_state_rank,
            config.vocab_size,
            contextual_mix_init=contextual_mix_init,
        )
        for layer in model.layers:
            layer.self_attn.shared_context = shared_context
    return model


def count_parameters(
    config: ModelConfig, architecture: str, contextual_mix_init: float
) -> int:
    with torch.device("meta"):
        model = build_model(config, architecture, contextual_mix_init)
    return sum(parameter.numel() for parameter in model.parameters())


def choose_matched_widths(
    architectures: list[str], state_rank: int, contextual_mix_init: float
) -> dict[str, int]:
    gqa_width = 256
    target = count_parameters(make_config("gqa", gqa_width), "gqa", contextual_mix_init)
    widths = {"gqa": gqa_width}
    for architecture in architectures:
        if architecture == "gqa":
            continue
        widths[architecture] = min(
            range(128, 385),
            key=lambda width: abs(
                count_parameters(
                    make_config(architecture, width, state_rank),
                    architecture,
                    contextual_mix_init,
                )
                - target
            ),
        )
    return widths


def target_logits(model: ComplexityModel, input_ids: torch.Tensor) -> torch.Tensor:
    hidden = model(input_ids, return_logits=False)["last_hidden_state"][:, -1]
    weight = model.embed_tokens.weight if model.lm_head is None else model.lm_head.weight
    return F.linear(hidden, weight)


def make_batch(task: str, batch_size: int, distance: int, seed: int):
    builder = build_associative_recall_batch if task == "associative_recall" else build_induction_batch
    return builder(batch_size, distance, 256, seed)


def evaluate(
    model: ComplexityModel,
    task: str,
    distances: list[int],
    batch_size: int,
    batches: int,
    device: torch.device,
) -> list[dict[str, float | int]]:
    model.eval()
    rows = []
    with torch.inference_mode():
        for distance in distances:
            values = {name: [] for name in ("accuracy", "mean_rank", "mean_margin", "nll")}
            for index in range(batches):
                batch = make_batch(task, batch_size, distance, 100_000 + distance * 100 + index)
                metrics = asdict(
                    score_target_logits(
                        target_logits(model, batch.input_ids.to(device)).float(),
                        batch.target_ids.to(device),
                    )
                )
                for name, value in metrics.items():
                    values[name].append(value)
            rows.append(
                {
                    "distance": distance,
                    "examples": batch_size * batches,
                    **{name: sum(items) / len(items) for name, items in values.items()},
                }
            )
    model.train()
    return rows


def train_condition(
    architecture: str,
    task: str,
    width: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    config = make_config(architecture, width, args.state_rank)
    model = build_model(config, architecture, args.contextual_mix_init).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    train_distances = args.train_distances
    eval_distances = args.eval_distances
    history = []
    started = time.perf_counter()

    for step in range(1, args.steps + 1):
        distance = train_distances[(step - 1) % len(train_distances)]
        batch = make_batch(task, args.batch_size, distance, args.seed * 1_000_000 + step)
        logits = target_logits(model, batch.input_ids.to(device))
        loss = F.cross_entropy(logits.float(), batch.target_ids.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            evaluation = evaluate(
                model,
                task,
                eval_distances,
                args.eval_batch_size,
                args.eval_batches,
                device,
            )
            elapsed = time.perf_counter() - started
            history.append(
                {
                    "step": step,
                    "train_loss": float(loss.item()),
                    "elapsed_seconds": elapsed,
                    "evaluation": evaluation,
                }
            )
            within = next(row for row in evaluation if row["distance"] == 16)
            beyond = next(row for row in evaluation if row["distance"] == 64)
            print(
                f"{architecture}/{task} step={step} loss={loss.item():.4f} "
                f"acc@16={within['accuracy']:.3f} acc@64={beyond['accuracy']:.3f}",
                flush=True,
            )

    dilations = (
        [int(getattr(layer.self_attn, "dilation")) for layer in model.layers]
        if architecture.startswith("causal_")
        or architecture == "shared_associative_attention"
        else None
    )
    receptive_field = 1 + (config.causal_conv_kernel_size - 1) * sum(dilations) if dilations else None
    return {
        "architecture": architecture,
        "task": task,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "intermediate_size": width,
        "train_distances": train_distances,
        "eval_distances": eval_distances,
        "dilations": dilations,
        "receptive_field_tokens": receptive_field,
        "history": history,
    }


def write_csv(results: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "architecture",
                "task",
                "parameters",
                "step",
                "distance",
                "examples",
                "accuracy",
                "mean_rank",
                "mean_margin",
                "nll",
                "elapsed_seconds",
            ],
        )
        writer.writeheader()
        for result in results:
            history = result["history"]
            assert isinstance(history, list)
            for checkpoint in history:
                assert isinstance(checkpoint, dict)
                evaluation = checkpoint["evaluation"]
                assert isinstance(evaluation, list)
                for metric in evaluation:
                    assert isinstance(metric, dict)
                    writer.writerow(
                        {
                            "architecture": result["architecture"],
                            "task": result["task"],
                            "parameters": result["parameters"],
                            "step": checkpoint["step"],
                            "elapsed_seconds": checkpoint["elapsed_seconds"],
                            **metric,
                        }
                    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--state-rank", type=int, default=16)
    parser.add_argument("--contextual-mix-init", type=float, default=0.1)
    parser.add_argument(
        "--train-distances", nargs="+", type=int, default=[8, 16, 32]
    )
    parser.add_argument(
        "--eval-distances", nargs="+", type=int, default=[8, 16, 32, 64, 128]
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=["gqa", "causal_conv"],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    widths = choose_matched_widths(
        args.architectures, args.state_rank, args.contextual_mix_init
    )
    results = []
    for architecture in args.architectures:
        for task in ("associative_recall", "induction"):
            results.append(train_condition(architecture, task, widths[architecture], args, device))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "protocol": vars(args) | {"output_dir": str(args.output_dir), "device_name": str(device)},
        "matched_widths": widths,
        "results": results,
    }
    json_path = args.output_dir / "synthetic_context_probe.json"
    csv_path = args.output_dir / "synthetic_context_probe.csv"
    json_path.write_text(json.dumps(payload, indent=2))
    write_csv(results, csv_path)
    print(json_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
