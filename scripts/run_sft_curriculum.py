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
    audit_planned_exposures,
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
    batch_size_override: int | None = None,
    lr_multiplier: float = 1.0,
) -> dict[str, int | float | str]:
    batch_size = batch_size_override or stage.batch_size or 32
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
        "lr": stage.lr * lr_multiplier,
        "eval_steps": stage.eval_steps or steps_per_epoch,
        "save_steps": stage.save_steps or steps_per_epoch,
    }


def selected_checkpoint(
    stage_root: Path,
    *,
    source_checkpoint: Path | None = None,
) -> Path:
    best_file = stage_root / "best.json"
    if best_file.exists():
        best = json.loads(best_file.read_text(encoding="utf-8"))
        checkpoint = Path(best["checkpoint"])
        if checkpoint.exists():
            return checkpoint
    # No trained checkpoint improved the held-out loss measured at stage
    # entry. Preserve that source checkpoint instead of silently advancing to
    # a worse periodic/final checkpoint.
    if source_checkpoint is not None and source_checkpoint.exists():
        return source_checkpoint
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
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-lr-multiplier",
        type=float,
        default=1.0,
        help="Multiply each stage LR when LoRA is active while preserving the stage schedule.",
    )
    parser.add_argument(
        "--lora-targets",
        default="q_proj,v_proj,o_proj,shared_gate,shared_up,shared_down",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable all validation and choose the latest completed stage checkpoint.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.world_size < 1:
        raise ValueError("--world-size must be positive")
    if args.lora_lr_multiplier <= 0:
        raise ValueError("--lora-lr-multiplier must be positive")
    curriculum_path = Path(args.curriculum_config).resolve()
    curriculum = load_curriculum(curriculum_path)
    final_index = curriculum.stage_index(args.through_stage)
    stages = curriculum.stages[: final_index + 1]
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    examples = load_example_index(args.sft_bin)
    metadata = load_projected_metadata(Path(args.sft_bin) / "projected.parquet")
    plans: list[dict[str, int | float | str]] = []
    for stage in stages:
        count = len(
            select_stage_examples(
                examples,
                curriculum,
                stage.name,
                metadata,
            )
        )
        plans.append(
            stage_plan(
                stage,
                examples=count,
                world_size=args.world_size,
                batch_size_override=args.batch_size,
                lr_multiplier=args.lora_lr_multiplier if args.lora_rank else 1.0,
            )
        )
    exposure_audit = None
    if final_index == len(curriculum.stages) - 1 and curriculum.exposure_groups:
        exposure_audit = audit_planned_exposures(examples, curriculum, metadata)
        if not exposure_audit["passed"]:
            raise ValueError(
                "planned curriculum exposure contract failed: "
                + json.dumps(exposure_audit["checks"], sort_keys=True)
            )
    print(
        json.dumps(
            {
                "curriculum": str(curriculum_path),
                "stages": plans,
                "planned_exposure_audit": exposure_audit,
            },
            indent=2,
        )
    )
    if args.dry_run:
        return

    checkpoint = Path(args.checkpoint).resolve()
    state_path = output_root / "curriculum-state.json"
    completed: list[dict[str, Any]] = []

    for stage, plan in zip(stages, plans, strict=True):
        stage_source_checkpoint = checkpoint
        stage_root = output_root / stage.name
        stage_root.mkdir(parents=True, exist_ok=True)
        command = [sys.executable]
        if args.world_size > 1:
            command.extend(
                [
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node",
                    str(args.world_size),
                    "-m",
                    "scripts.sft_tr",
                ]
            )
        else:
            command.extend(["-m", "scripts.sft_tr"])
        command.extend(
            [
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
            "--grad-ckpt",
            "--loss-chunk-tokens",
            "1024",
            "--save-steps",
            str(plan["save_steps"]),
            "--save-total-limit",
            "2",
            "--save-dir",
            str(stage_root),
            "--run-name",
            f"sft-curriculum-{stage.name}",
            "--seed",
            str(args.seed),
            "--use-custom-kernels",
            "auto",
            ]
        )
        if args.lora_rank:
            command.extend(
                [
                    "--lora-rank",
                    str(args.lora_rank),
                    "--lora-alpha",
                    str(args.lora_alpha),
                    "--lora-dropout",
                    str(args.lora_dropout),
                    "--lora-targets",
                    args.lora_targets,
                ]
            )
        if args.no_eval:
            command.extend(
                [
                    "--eval-steps",
                    "0",
                    "--no-eval-at-start",
                    "--early-stopping-patience",
                    "0",
                ]
            )
        else:
            command.extend(
                [
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
                ]
            )
        print(f"\n=== {stage.name} ===", flush=True)
        print(" ".join(command), flush=True)
        subprocess.run(command, check=True)
        checkpoint = selected_checkpoint(
            stage_root,
            source_checkpoint=None if args.no_eval else stage_source_checkpoint,
        )
        completed.append(
            {
                "stage": stage.name,
                "plan": plan,
                "stage_config": asdict(stage),
                "source_checkpoint": str(stage_source_checkpoint),
                "selected_checkpoint": str(checkpoint),
                "stage_improved": checkpoint != stage_source_checkpoint,
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
