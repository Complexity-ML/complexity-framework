#!/usr/bin/env python3
"""Run runtime-filtered SFT stages without creating derivative datasets."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from complexity.training.sft_curriculum import (
    CurriculumStage,
    load_curriculum,
    load_projected_metadata,
    select_stage_examples,
)


def load_example_index(sft_root: str | Path) -> list[dict[str, Any]]:
    root = Path(sft_root)
    index = root / "train" / "examples.jsonl"
    if not index.exists():
        index = root / "examples.jsonl"
    if not index.exists():
        raise FileNotFoundError(f"SFT example index not found under {root}")
    with index.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def selected_count(
    sft_root: str | Path,
    curriculum_path: str | Path,
    stage_name: str,
) -> int:
    root = Path(sft_root)
    curriculum = load_curriculum(curriculum_path)
    metadata = load_projected_metadata(root / "projected.parquet")
    return len(
        select_stage_examples(
            load_example_index(root),
            curriculum,
            stage_name,
            metadata,
        )
    )


def stage_plan(
    stage: CurriculumStage,
    *,
    examples: int,
    world_size: int,
) -> dict[str, int | float | str]:
    batch_size = stage.batch_size or 32
    seq_len = stage.seq_len or 384
    examples_per_rank = math.ceil(examples / world_size)
    steps_per_epoch = math.ceil(examples_per_rank / batch_size)
    return {
        "stage": stage.name,
        "examples": examples,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "epochs": stage.epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": steps_per_epoch * stage.epochs,
        "lr": stage.lr,
        "eval_steps": stage.eval_steps or steps_per_epoch,
        "save_steps": stage.save_steps or steps_per_epoch,
    }


def selected_checkpoint(stage_root: Path) -> Path:
    best_file = stage_root / "best.json"
    if best_file.exists():
        best = json.loads(best_file.read_text(encoding="utf-8"))
        checkpoint = Path(best["checkpoint"])
        if checkpoint.exists():
            return checkpoint
    latest = stage_root / "latest"
    if latest.exists():
        target = latest.read_text(encoding="utf-8").strip()
        checkpoint = stage_root / target
        if checkpoint.exists():
            return checkpoint
    candidates = sorted(stage_root.glob("step_*"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"no completed checkpoint under {stage_root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sft-bin", required=True)
    parser.add_argument("--curriculum-config", required=True)
    parser.add_argument("--through-stage", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--tokenizer", default="./tokenizer-o200k")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.world_size < 1:
        raise ValueError("--world-size must be positive")
    curriculum_path = Path(args.curriculum_config).resolve()
    curriculum = load_curriculum(curriculum_path)
    final_index = curriculum.stage_index(args.through_stage)
    stages = curriculum.stages[: final_index + 1]
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    plans: list[dict[str, int | float | str]] = []
    for stage in stages:
        count = selected_count(args.sft_bin, curriculum_path, stage.name)
        plans.append(stage_plan(stage, examples=count, world_size=args.world_size))
    print(json.dumps({"curriculum": str(curriculum_path), "stages": plans}, indent=2))
    if args.dry_run:
        return

    trainer = Path(__file__).with_name("sft_100m_o200k_tr_local.py")
    checkpoint = Path(args.checkpoint).resolve()
    state_path = output_root / "curriculum-state.json"
    completed: list[dict[str, Any]] = []

    for stage, plan in zip(stages, plans, strict=True):
        stage_root = output_root / stage.name
        stage_root.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(trainer),
            "--checkpoint",
            str(checkpoint),
            "--sft-bin",
            str(Path(args.sft_bin).resolve()),
            "--tokenizer",
            str(Path(args.tokenizer).resolve()),
            "--curriculum-config",
            str(curriculum_path),
            "--curriculum-stage",
            stage.name,
            "--steps",
            str(plan["total_steps"]),
            "--epochs",
            str(plan["epochs"]),
            "--batch-size",
            str(plan["batch_size"]),
            "--seq-len",
            str(plan["seq_len"]),
            "--lr",
            str(plan["lr"]),
            "--weight-decay",
            "0.0",
            "--bf16",
            "--no-freeze-token-io",
            "--eval-steps",
            str(plan["eval_steps"]),
            "--eval-batches",
            "0",
            "--eval-at-start",
            "--save-best",
            "--early-stopping-min-epochs",
            "1",
            "--early-stopping-patience",
            str(args.early_stopping_patience),
            "--early-stopping-min-delta",
            str(args.early_stopping_min_delta),
            "--save-steps",
            str(plan["save_steps"]),
            "--save-total-limit",
            "2",
            "--save-model-only",
            "--save-dir",
            str(stage_root),
            "--run-name",
            f"sft-curriculum-{stage.name}",
            "--seed",
            str(args.seed),
            "--use-custom-kernels",
            "auto",
        ]
        print(f"\n=== {stage.name} ===", flush=True)
        print(" ".join(command), flush=True)
        subprocess.run(command, check=True)
        checkpoint = selected_checkpoint(stage_root)
        completed.append(
            {
                "stage": stage.name,
                "plan": plan,
                "stage_config": asdict(stage),
                "selected_checkpoint": str(checkpoint),
            }
        )
        state_path.write_text(
            json.dumps(
                {
                    "source_checkpoint": str(Path(args.checkpoint).resolve()),
                    "sft_bin": str(Path(args.sft_bin).resolve()),
                    "completed": completed,
                    "next_checkpoint": str(checkpoint),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
