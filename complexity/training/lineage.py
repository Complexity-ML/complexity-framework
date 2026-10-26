"""Text model factories, local corpus proofs and bounded update schedules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from complexity.config import ModelConfig, get_preset

from .finetuning import (
    TEXT_REFINEMENT,
    validate_full_parameter_finetuning,
    validate_refinement_plan,
)
from .refinement import (
    REFINEMENT_LR_SCHEDULER,
    REFINEMENT_PEAK_LR,
    REFINEMENT_TOKENS_PER_STEP,
    REFINEMENT_WARMUP_TOKENS,
    REFINEMENT_WEIGHT_DECAY,
    validate_refinement_recipe,
)


def make_tr_hash_config(preset: str) -> ModelConfig:
    """Keep the canonical 32K vocabulary and composed causal routing."""
    config = get_preset(preset)
    config.vocab_size = 32000
    config.mlp_type = "tr_hash_engine"
    config.routing_strategy = "token_id_composed_modulo"
    config.num_experts = 4
    config.top_k = 2
    config.top_k_primary_weight = 0.5
    config.shared_expert = True
    config.shared_output_scale = 1.0
    config.routed_output_scale = 2.0
    config.tie_word_embeddings = True
    return ModelConfig.from_dict(config.to_dict(), strict=True)


def load_lineage_plan(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"local replay plan required for lineage audit: {path}")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError(f"lineage plan must contain a JSON object: {path}")
    return plan


def validate_lineage_plans(
    *, stage: str, tokenized_plan: str | Path, pretrain_plan: str | Path | None
) -> str | None:
    """Require exact unique-core identity for a fresh refinement stage."""
    plan = load_lineage_plan(tokenized_plan)
    if stage == "pretraining":
        if pretrain_plan is not None:
            raise ValueError("pretrain_plan is only valid for refinement")
        return None
    if stage != "refinement":
        raise ValueError(f"unknown text lineage stage: {stage!r}")
    if pretrain_plan is None:
        raise ValueError("refinement requires pretrain_plan")
    pretrain = load_lineage_plan(pretrain_plan)
    validate_full_parameter_finetuning(
        TEXT_REFINEMENT,
        unique_tokens=plan["unique_tokens"],
        pretrain_unique_tokens=pretrain["unique_tokens"],
    )
    return validate_refinement_plan(plan, pretrain)


def resolve_lineage_schedule(
    plan: Mapping,
    *,
    world_size: int,
    batch_size: int,
    gradient_accumulation: int,
    sequence_length: int,
    max_steps: int | None = None,
    target_tokens: int | None = None,
) -> dict:
    """Floor to complete global updates and reject audited-plan overdraw."""
    for name, value in (
        ("world_size", world_size),
        ("batch_size", batch_size),
        ("gradient_accumulation", gradient_accumulation),
        ("sequence_length", sequence_length),
        ("trained_tokens", plan.get("trained_tokens")),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    available = plan["trained_tokens"]
    if target_tokens is not None and (
        type(target_tokens) is not int or not 0 < target_tokens <= available
    ):
        raise ValueError("target tokens must be positive and within audited plan coverage")
    budget = available if target_tokens is None else target_tokens
    tokens_per_step = world_size * batch_size * gradient_accumulation * sequence_length
    steps = budget // tokens_per_step if max_steps is None else max_steps
    if type(steps) is not int or steps < 1:
        raise ValueError("plan contains fewer tokens than one global optimizer step")
    scheduled = steps * tokens_per_step
    if scheduled > budget:
        raise ValueError(
            f"training schedule exceeds audited plan coverage: dataset={budget:,}, schedule={scheduled:,}"
        )
    return {
        "max_steps": steps,
        "tokens_per_step": tokens_per_step,
        "scheduled_tokens": scheduled,
        "unused_tail": budget - scheduled,
        "available_tokens": available,
        "unconsumed_tokens": available - scheduled,
    }


def validate_lineage_training_args(args, *, world_size: int) -> dict:
    """Preflight recipe, stage and complete update budget before loading data."""
    validate_refinement_recipe(
        stage=args.stage,
        model_preset=args.model_preset,
        learning_rate=args.lr,
        lr_scheduler=args.lr_scheduler,
        warmup_tokens=args.warmup_tokens,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        tokens_per_step=args.batch_size * world_size * args.gradient_accumulation * args.seq_len,
    )
    if args.stage == "pretraining" and args.init_checkpoint:
        raise ValueError("pretraining is a lineage root and cannot use init_checkpoint")
    if args.stage == "refinement" and not args.init_checkpoint:
        raise ValueError("refinement requires completed weights via init_checkpoint")
    if str(args.tokenized_data).startswith("hf://") and getattr(args, "num_workers", 0) != 0:
        raise ValueError("remote pretokenized mixtures require num_workers=0")
    fingerprint = validate_lineage_plans(
        stage=args.stage, tokenized_plan=args.tokenized_plan, pretrain_plan=args.pretrain_plan
    )
    plan = load_lineage_plan(args.tokenized_plan)
    if (
        str(args.tokenized_data).startswith("hf://")
        and plan.get("dataset", args.tokenized_data) != args.tokenized_data
    ):
        raise ValueError("plan dataset identity differs from the requested corpus")
    if plan.get("revision", args.tokenized_revision) != args.tokenized_revision:
        raise ValueError("plan dataset revision differs from the requested corpus revision")
    if plan.get("seq_len", args.seq_len) != args.seq_len:
        raise ValueError("plan sequence length differs from the training context")
    schedule = resolve_lineage_schedule(
        plan,
        world_size=world_size,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        sequence_length=args.seq_len,
        max_steps=args.max_steps,
        target_tokens=args.target_tokens,
    )
    return {**schedule, "refinement_fingerprint": fingerprint}


def build_agentic_recipe(
    stage: str,
    plan: Mapping,
    *,
    world_size: int,
    batch_size: int = 16,
    sequence_length: int = 2048,
    overrides: Mapping | None = None,
) -> dict:
    """Resolve and validate canonical 100M settings before any dispatch.

    This reports settings and a bounded schedule. It does not read credentials,
    start torchrun or establish that the checkpoint/corpus is available.
    """
    if stage not in {"pretraining", "refinement"}:
        raise ValueError(f"unknown text lineage stage: {stage!r}")
    settings = {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "learning_rate": REFINEMENT_PEAK_LR if stage == "refinement" else 3e-4,
        "lr_scheduler": REFINEMENT_LR_SCHEDULER if stage == "refinement" else "wsd",
        "warmup_tokens": REFINEMENT_WARMUP_TOKENS if stage == "refinement" else 1_000_000_000,
        "warmup_steps": None,
        "weight_decay": REFINEMENT_WEIGHT_DECAY,
        "gradient_accumulation": None if stage == "refinement" else 4,
        "max_steps": None,
    }
    if overrides is not None:
        unknown = overrides.keys() - settings.keys()
        if unknown:
            raise ValueError(f"unknown Agentic recipe overrides: {sorted(unknown)}")
        settings.update(overrides)
    for name, value in (
        ("world_size", world_size),
        ("batch_size", settings["batch_size"]),
        ("sequence_length", settings["sequence_length"]),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    per_micro_step = world_size * settings["batch_size"] * settings["sequence_length"]
    if settings["gradient_accumulation"] is None:
        if REFINEMENT_TOKENS_PER_STEP % per_micro_step:
            raise ValueError(
                "reference refinement tokens/update is not divisible by nproc*batch*sequence"
            )
        settings["gradient_accumulation"] = REFINEMENT_TOKENS_PER_STEP // per_micro_step
    schedule = resolve_lineage_schedule(
        plan,
        world_size=world_size,
        batch_size=settings["batch_size"],
        gradient_accumulation=settings["gradient_accumulation"],
        sequence_length=settings["sequence_length"],
        max_steps=settings["max_steps"],
    )
    validate_refinement_recipe(
        stage=stage,
        model_preset="complexity-100m",
        learning_rate=settings["learning_rate"],
        lr_scheduler=settings["lr_scheduler"],
        warmup_tokens=settings["warmup_tokens"],
        warmup_steps=settings["warmup_steps"],
        weight_decay=settings["weight_decay"],
        tokens_per_step=schedule["tokens_per_step"],
    )
    return {
        **settings,
        **schedule,
        "stage": stage,
        "model_preset": "complexity-100m",
        "world_size": world_size,
    }
