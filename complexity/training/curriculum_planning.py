"""Local, read-only curriculum plans and preservation-first checkpoint selection.

This integrates reusable planner responsibilities. It neither starts training
nor replaces the active portable 200M ablation or the Stack SFT runtime.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

from .sequence_packing import pack_example_lengths, resolve_epoch_schedule
from .sft_curriculum import (
    CurriculumStage,
    audit_planned_exposures,
    audit_stage_loss_targets,
    load_curriculum,
    load_projected_metadata,
    select_stage_examples,
)


@dataclass(frozen=True)
class CurriculumPlanningConfig:
    checkpoint: str
    sft_bin: str
    curriculum_config: str
    through_stage: str
    output_root: str
    world_size: int = 1
    batch_size: int | None = None
    lr_multiplier: float = 1.0
    lora_rank: int = 16
    lora_alpha: float = 16.0
    expert_lr_multiplier: float = 0.25
    require_release_ready: bool = True
    pack_sequences: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CurriculumPlanningConfig:
        if not isinstance(data, Mapping):
            raise ValueError("curriculum planning config must contain an object")
        known = {field.name for field in fields(cls)}
        if unknown := data.keys() - known:
            raise ValueError(f"unknown curriculum planning fields: {sorted(unknown)}")
        required = {"checkpoint", "sft_bin", "curriculum_config", "through_stage", "output_root"}
        for name in required:
            if not isinstance(data.get(name), str) or not data[name].strip():
                raise ValueError(f"curriculum planning requires {name}")
        config = cls(**data)
        if type(config.lora_rank) is not int or config.lora_rank <= 0:
            raise ValueError("curriculum planning requires a positive LoRA rank")
        if type(config.world_size) is not int or config.world_size < 1:
            raise ValueError("curriculum planning world_size must be a positive integer")
        if config.batch_size is not None and (type(config.batch_size) is not int or config.batch_size < 1):
            raise ValueError("curriculum planning batch_size must be a positive integer")
        for name in ("lora_alpha", "expert_lr_multiplier", "lr_multiplier"):
            value = getattr(config, name)
            if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
                raise ValueError(f"curriculum planning {name} must be finite and positive")
        if config.require_release_ready is not True or config.pack_sequences is not True:
            raise ValueError("curriculum requires release-ready data and sequence packing")
        return config


def load_example_index(sft_root: str | Path) -> list[dict[str, Any]]:
    """Merge an exact metadata sidecar without replacing token offsets/counts."""
    root = Path(sft_root)
    index = root / "train" / "examples.jsonl"
    if not index.is_file():
        index = root / "examples.jsonl"
    if not index.is_file():
        raise FileNotFoundError(f"SFT example index not found under {root}")
    examples = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
    example_ids = [str(row["example_id"]) for row in examples]
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("SFT example index contains duplicate example IDs")
    sidecar = index.parent / "loss_metadata.jsonl"
    if not sidecar.is_file():
        return examples
    rows = [json.loads(line) for line in sidecar.read_text().splitlines() if line.strip()]
    metadata = {str(row["example_id"]): row for row in rows}
    if len(metadata) != len(rows):
        raise ValueError("SFT loss metadata contains duplicate example IDs")
    if metadata.keys() != set(example_ids):
        raise ValueError("SFT loss metadata must exactly match the example index")
    return [{**metadata[str(row["example_id"])], **row} for row in examples]


def validate_sft_release_manifest(
    root: str | Path, *, reasoning_envelope: bool = False
) -> dict[str, Any]:
    """Require audited release readiness and reject a second target envelope."""
    root = Path(root)
    dataset_root = root.parent if (root / "sft.idx.json").is_file() else root
    path = dataset_root / "manifest.json"
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict) or manifest.get("quality_status") != "passed":
        raise ValueError("SFT release quality_status must be passed")
    quality = manifest.get("release_quality", {})
    if not isinstance(quality, dict) or quality.get("ready") is not True:
        raise ValueError("SFT release_quality.ready must be true")
    if quality.get("reasoning_envelope_version") and reasoning_envelope:
        raise ValueError("SFT targets already contain reasoning envelopes")
    return manifest


def plan_curriculum(config: CurriculumPlanningConfig) -> dict[str, Any]:
    """Audit selected stages and mandatory packing without materializing data."""
    # Revalidate directly constructed dataclasses, too.
    config = CurriculumPlanningConfig.from_dict(
        {field.name: getattr(config, field.name) for field in fields(config)}
    )
    release = validate_sft_release_manifest(config.sft_bin)
    curriculum = load_curriculum(config.curriculum_config)
    examples = load_example_index(config.sft_bin)
    metadata = load_projected_metadata(Path(config.sft_bin) / "projected.parquet")
    final_index = curriculum.stage_index(config.through_stage)
    plans, loss_audits = [], {}
    for stage in curriculum.stages[: final_index + 1]:
        selected = select_stage_examples(examples, curriculum, stage.name, metadata)
        if any("num_tokens" not in row for row in selected):
            raise ValueError("sequence-packed curriculum requires num_tokens for every example")
        plans.append(
            stage_plan(
                stage, examples=len(selected), world_size=config.world_size,
                batch_size_override=config.batch_size, lr_multiplier=config.lr_multiplier,
                example_lengths=[row["num_tokens"] for row in selected],
            )
        )
        if stage.loss_task_targets or stage.loss_groups:
            audit = audit_stage_loss_targets(examples, curriculum, stage.name, metadata)
            if not audit["passed"]:
                raise ValueError(f"planned full-shard loss target contract failed: {stage.name}")
            loss_audits[stage.name] = audit
    exposure = None
    if final_index == len(curriculum.stages) - 1 and curriculum.exposure_groups:
        exposure = audit_planned_exposures(examples, curriculum, metadata)
        if not exposure["passed"]:
            raise ValueError("planned curriculum exposure contract failed")
    return {
        "stages": plans,
        "planned_exposure_audit": exposure,
        "planned_loss_target_audits": loss_audits,
        "quality_status": release["quality_status"],
        "adaptation": {
            "lora_rank": config.lora_rank, "lora_alpha": config.lora_alpha,
            "expert_lr_multiplier": config.expert_lr_multiplier,
        },
        "training_started": False,
        "execution_supported": False,
    }


def stage_plan(
    stage: CurriculumStage,
    *,
    examples: int,
    world_size: int,
    batch_size_override: int | None = None,
    lr_multiplier: float = 1.0,
    example_lengths: list[int] | None = None,
) -> dict[str, Any]:
    """Plan complete epochs from selected rows or their immutable packed lengths."""
    batch_size = batch_size_override if batch_size_override is not None else stage.batch_size or 32
    seq_len = stage.seq_len or 384
    for name, value in (("examples", examples), ("batch_size", batch_size), ("seq_len", seq_len)):
        if type(value) is not int or value < 1:
            raise ValueError(f"curriculum {name} must be a positive integer")
    if type(lr_multiplier) not in {int, float} or not math.isfinite(lr_multiplier) or lr_multiplier <= 0:
        raise ValueError("curriculum LR multiplier must be finite and positive")
    training_items = examples
    if example_lengths is not None:
        if len(example_lengths) != examples:
            raise ValueError("packed stage lengths must match selected examples")
        if any(type(length) is not int or not 1 <= length <= seq_len for length in example_lengths):
            raise ValueError("packed stage lengths must fit the context without truncation")
        training_items = pack_example_lengths(
            example_lengths, sequence_length=seq_len, separator_tokens=1
        ).packed_items
    schedule = resolve_epoch_schedule(
        items=training_items, world_size=world_size, batch_size_per_rank=batch_size, epochs=stage.epochs
    )
    return {
        "stage": stage.name,
        "examples": examples,
        "training_items": training_items,
        "pack_sequences": example_lengths is not None,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "epochs": stage.epochs,
        "steps_per_epoch": schedule.steps_per_epoch,
        "total_steps": schedule.total_steps,
        "lr": stage.lr * lr_multiplier,
        "eval_steps": stage.eval_steps or schedule.steps_per_epoch,
        "save_steps": stage.save_steps or schedule.steps_per_epoch,
    }


def selected_checkpoint(stage_root: str | Path, *, source_checkpoint: str | Path | None = None) -> Path:
    """Prefer recorded held-out improvement; otherwise preserve the entry model."""
    stage_root = Path(stage_root)
    best_file = stage_root / "best.json"
    if best_file.is_file():
        best = json.loads(best_file.read_text())
        checkpoint = Path(best["checkpoint"])
        if checkpoint.exists():
            return checkpoint
    if source_checkpoint is not None and Path(source_checkpoint).exists():
        return Path(source_checkpoint)
    latest = stage_root / "latest"
    if latest.is_file():
        target = latest.read_text().strip()
        checkpoint = stage_root / target
        if target and checkpoint.exists():
            return checkpoint
    candidates = sorted(path for path in stage_root.glob("step_*") if path.is_dir())
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"no completed checkpoint under {stage_root}")
