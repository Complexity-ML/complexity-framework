#!/usr/bin/env python3
"""Train a preset TR-Hash text model on an audited pretokenized lineage.

The same entry point handles replay pretraining and lexical refinement.  A
refinement must provide the completed pretraining weights, the pretraining
plan, and a one-pass plan whose exact ``unique_core`` fingerprint matches it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from complexity.config import PRESET_CONFIGS, ModelConfig, get_preset
from complexity.training import (
    TEXT_REFINEMENT,
    PretokenizedCorpusMixtureDataset,
    TensorBoardCallback,
    TrainRunner,
    validate_full_parameter_finetuning,
    validate_refinement_plan,
)
from complexity.utils.checkpointing import peek_latest_checkpoint_step


def make_tr_hash_config(preset: str) -> ModelConfig:
    """Resolve a size preset while keeping the canonical lexical router."""

    config = get_preset(preset)
    config.vocab_size = 32000
    config.mlp_type = "tr_hash_engine"
    config.routing_strategy = "token_id_multi_hash"
    config.route_hash_count = 2
    config.num_experts = 4
    config.top_k = 2
    config.top_k_primary_weight = 0.5
    config.shared_expert = True
    config.shared_output_scale = 1.0
    config.routed_output_scale = 2.0
    config.tie_word_embeddings = True
    return config


def _load_plan(path: str | os.PathLike[str]) -> dict:
    plan_path = Path(path)
    if not plan_path.is_file():
        raise ValueError(f"local replay plan required for lineage audit: {plan_path}")
    return json.loads(plan_path.read_text(encoding="utf-8"))


def validate_lineage_plans(
    *, stage: str, tokenized_plan: str, pretrain_plan: str | None
) -> str | None:
    """Validate the stage boundary and return its refinement fingerprint."""

    plan = _load_plan(tokenized_plan)
    if stage == "pretraining":
        if pretrain_plan is not None:
            raise ValueError("--pretrain-plan is only valid for refinement")
        return None
    if stage != "refinement":
        raise ValueError(f"unknown text lineage stage: {stage!r}")
    if pretrain_plan is None:
        raise ValueError("refinement requires --pretrain-plan")
    pretrain = _load_plan(pretrain_plan)
    validate_full_parameter_finetuning(
        TEXT_REFINEMENT,
        unique_tokens=int(plan["unique_tokens"]),
        pretrain_unique_tokens=int(pretrain["unique_tokens"]),
    )
    return validate_refinement_plan(plan, pretrain)


def resume_skip_rows(args: argparse.Namespace) -> int:
    """Skip this rank's rows already consumed by a resumable run."""

    if not args.resume:
        return 0
    if args.resume == "auto":
        step = peek_latest_checkpoint_step(args.checkpoint_dir)
    else:
        state_path = Path(args.resume) / "training_state.json"
        step = (
            json.loads(state_path.read_text(encoding="utf-8"))["step"]
            if state_path.is_file()
            else None
        )
    if step is None:
        return 0
    return int(step) * args.batch_size * args.gradient_accumulation


class TRHashTextLineageRunner(TrainRunner):
    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--model-preset", choices=sorted(PRESET_CONFIGS), required=True)
        parser.add_argument("--stage", choices=("pretraining", "refinement"), required=True)
        parser.add_argument("--tokenized-data", required=True)
        parser.add_argument("--tokenized-plan", required=True)
        parser.add_argument("--pretrain-plan", default=None)
        parser.add_argument(
            "--tokenized-cache-dir",
            default=os.environ.get("TR_HASH_TOKEN_CACHE", "artifacts/tr_hash_token_cache"),
        )
        parser.add_argument("--tokenized-cache-gb", type=float, default=16.0)
        parser.add_argument("--tokenized-revision", default="main")
        parser.add_argument("--tokenized-prefetch-shards", type=int, default=0)
        parser.add_argument("--tokenized-hf-token-env", default="HF_TOKEN")

    def build_dataset(self, tokenizer, args, rank: int, world_size: int):
        del tokenizer
        if str(args.tokenized_data).startswith("hf://") and args.num_workers != 0:
            raise ValueError("remote pretokenized mixtures require --num-workers 0")
        if args.stage == "pretraining" and args.init_checkpoint:
            raise ValueError("pretraining is a lineage root and cannot use --init-checkpoint")
        if args.stage == "refinement" and not args.init_checkpoint:
            raise ValueError("refinement requires completed weights via --init-checkpoint")

        fingerprint = validate_lineage_plans(
            stage=args.stage,
            tokenized_plan=args.tokenized_plan,
            pretrain_plan=args.pretrain_plan,
        )
        if fingerprint is not None and rank == 0:
            print(
                "[refinement contract] exact pretraining unique_core verified "
                f"sha256={fingerprint}",
                flush=True,
            )

        dataset = PretokenizedCorpusMixtureDataset(
            args.tokenized_data,
            rank=rank,
            world_size=world_size,
            cache_dir=args.tokenized_cache_dir,
            cache_max_bytes=int(args.tokenized_cache_gb * 1024**3),
            revision=args.tokenized_revision,
            token=os.environ.get(args.tokenized_hf_token_env),
            prefetch_shards=args.tokenized_prefetch_shards,
            replay_plan=args.tokenized_plan,
            resume_skip_rows=resume_skip_rows(args),
        )
        if dataset.seq_len != args.seq_len:
            raise ValueError(
                f"tokenized seq_len={dataset.seq_len} does not match --seq-len={args.seq_len}"
            )
        tokens_per_step = args.batch_size * world_size * args.gradient_accumulation * args.seq_len
        scheduled_steps = (
            args.max_steps
            if args.max_steps is not None
            else math.ceil(args.target_tokens / tokens_per_step)
        )
        scheduled_tokens = scheduled_steps * tokens_per_step
        if scheduled_tokens > dataset.trained_tokens:
            raise ValueError(
                "training schedule exceeds audited plan coverage: "
                f"dataset={dataset.trained_tokens:,}, schedule={scheduled_tokens:,}"
            )
        return dataset

    def extra_callbacks(self, trainer, args, is_main: bool):
        del trainer, is_main
        return [TensorBoardCallback(log_dir=os.path.join(args.checkpoint_dir, "tensorboard"))]


def _preset_from_cli() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-preset", choices=sorted(PRESET_CONFIGS), required=True)
    args, _ = parser.parse_known_args()
    return args.model_preset


def main() -> None:
    preset = _preset_from_cli()
    TRHashTextLineageRunner(
        make_config=lambda: make_tr_hash_config(preset),
        run_name=f"tr-hash-{preset}-text-lineage",
        checkpoint_dir=f"artifacts/tr_hash_{preset}_text_lineage",
        default_lr=3e-4,
        default_batch_size=4,
        default_seq_len=1024,
        default_target_tokens=50_000_000,
        default_gradient_accumulation=1,
        default_gradient_checkpointing=False,
        default_save_steps=0,
        default_distributed_mode="ddp",
        optimizer_type="adamw",
        label_smoothing=0.0,
    ).run()


if __name__ == "__main__":
    main()
