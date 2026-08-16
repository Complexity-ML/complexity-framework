#!/usr/bin/env python3
"""Evaluate synthetic associative recall from a saved Complexity checkpoint."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from complexity.config import ModelConfig
from complexity.evaluation.associative_recall import (
    build_associative_recall_batch,
    build_induction_batch,
    score_target_logits,
)
from complexity.models import ComplexityModel
from complexity.utils.local_checkpoint import load_local_checkpoint


TASKS = {
    "associative": build_associative_recall_batch,
    "induction": build_induction_batch,
}


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tasks", default="associative,induction")
    parser.add_argument("--distances", default="128,256,512,1024,2048,4096,8192")
    parser.add_argument("--seeds", default="1,2,3,4")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint_dir, state = load_local_checkpoint(args.checkpoint, map_location="cpu")
    config = ModelConfig.from_dict(state["config"])
    model = ComplexityModel(config)
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()

    task_names = [item.strip() for item in args.tasks.split(",") if item.strip()]
    unknown = sorted(set(task_names) - set(TASKS))
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}")
    distances = parse_int_list(args.distances)
    seeds = parse_int_list(args.seeds)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if device.type == "cuda" else None
    rows = []
    with torch.inference_mode():
        for task_name in task_names:
            builder = TASKS[task_name]
            for distance in distances:
                for seed in seeds:
                    batch = builder(
                        batch_size=args.batch_size,
                        distance=distance,
                        vocab_size=config.vocab_size,
                        seed=seed,
                    )
                    input_ids = batch.input_ids.to(device)
                    targets = batch.target_ids.to(device)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=dtype,
                        enabled=dtype is not None,
                    ):
                        outputs = model(input_ids, return_logits=False)
                        last_hidden = outputs["last_hidden_state"][:, -1]
                        logits = last_hidden @ model.embed_tokens.weight.T
                    metrics = score_target_logits(logits.float(), targets)
                    rows.append(
                        {
                            "checkpoint": str(checkpoint_dir),
                            "step": int(state.get("step", -1)),
                            "attention_type": config.attention_type,
                            "task": task_name,
                            "distance": distance,
                            "sequence_length": input_ids.shape[1],
                            "seed": seed,
                            "batch_size": args.batch_size,
                            "accuracy": f"{metrics.accuracy:.8f}",
                            "mean_rank": f"{metrics.mean_rank:.4f}",
                            "mean_margin": f"{metrics.mean_margin:.6f}",
                            "nll": f"{metrics.nll:.6f}",
                        }
                    )
                    print(
                        f"{task_name} distance={distance} seed={seed} "
                        f"acc={metrics.accuracy:.3f} rank={metrics.mean_rank:.1f} "
                        f"nll={metrics.nll:.3f}"
                    )

    fieldnames = list(rows[0])
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
