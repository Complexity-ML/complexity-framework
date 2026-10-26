"""Canonical 200M text profile and replay execution over the shared runner.

Corpus budgets are historical profile data, not new defaults for the paused
1B run. Runtime safeguards prevent repetition beyond a pretokenized plan.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from complexity.config import ModelConfig

from .corpus_mixture import (
    PretokenizedCorpusMixtureDataset,
    TextCorpusSource,
    WeightedStreamingTextDataset,
)
from .lineage import (
    load_lineage_plan,
    resolve_lineage_schedule,
    validate_lineage_plans,
    validate_lineage_training_args,
)
from .lineage_runner import TRHashTextLineageRunner, _AuditedParser, resume_skip_rows
from .runner import TrainRunner

TARGET_TOKENS = 200_000_000_000
TOKEN_PACKS = 40
CORPUS_TOKEN_BUDGETS = {
    "dclm": 90_000_000_000,
    "fineweb_edu_dedup": 60_000_000_000,
    "stack_edu": 20_000_000_000,
    "finemath": 10_000_000_000,
    "infiwebmath": 10_000_000_000,
    "cosmopedia_v2": 10_000_000_000,
}


def make_200m_config() -> ModelConfig:
    return ModelConfig(
        hidden_size=896,
        num_hidden_layers=16,
        num_attention_heads=4,
        num_key_value_heads=1,
        intermediate_size=256,
        shared_intermediate_size=3072,
        vocab_size=32000,
        max_position_embeddings=2048,
        attention_type="gqa",
        mlp_type="tr_hash_engine",
        num_experts=4,
        top_k=2,
        top_k_primary_weight=0.5,
        routing_strategy="token_id_composed_modulo",
        shared_expert=True,
        shared_output_scale=1.0,
        routed_output_scale=2.0,
        use_qk_norm=True,
        tie_word_embeddings=True,
    )


def corpus_sources(stack_edu_data: str) -> tuple[TextCorpusSource, ...]:
    if not stack_edu_data:
        raise ValueError(
            "Stack-Edu must be materialized: its official repository contains SWHIDs, not source text"
        )
    return (
        TextCorpusSource("dclm", 0.45, dataset_id="mlfoundations/dclm-baseline-1.0-parquet"),
        TextCorpusSource(
            "fineweb_edu_dedup",
            0.30,
            dataset_id="HuggingFaceTB/smollm-corpus",
            config_name="fineweb-edu-dedup",
        ),
        TextCorpusSource("stack_edu", 0.10, data_files=stack_edu_data),
        TextCorpusSource(
            "finemath", 0.05, dataset_id="HuggingFaceTB/finemath", config_name="finemath-3plus"
        ),
        TextCorpusSource(
            "infiwebmath",
            0.05,
            dataset_id="HuggingFaceTB/finemath",
            config_name="infiwebmath-3plus",
        ),
        TextCorpusSource(
            "cosmopedia_v2",
            0.05,
            dataset_id="HuggingFaceTB/smollm-corpus",
            config_name="cosmopedia-v2",
        ),
    )


class TRHash200MPretrainRunner(TrainRunner):
    def add_args(self, parser):
        parser.set_defaults(token_packs=TOKEN_PACKS)
        parser.add_argument("--stack-edu-data")
        parser.add_argument("--tokenized-data")
        parser.add_argument("--tokenized-cache-dir", default="artifacts/tr_hash_token_cache")
        parser.add_argument("--tokenized-cache-gb", type=float, default=32.0)
        parser.add_argument("--tokenized-revision", default="main")
        parser.add_argument("--tokenized-prefetch-shards", type=int, default=1)
        parser.add_argument("--tokenized-hf-token-env", default="HF_TOKEN")
        parser.add_argument("--tokenized-plan")
        parser.add_argument(
            "--pretrain-plan", help="Required corpus proof when initializing a refinement"
        )

    def _build_parser(self):
        parent = super()._build_parser()
        if getattr(self, "_lineage_config_mode", False):
            parent.set_defaults(target_tokens=None)

        def audit(args):
            if str(args.tokenized_data).startswith("hf://") and args.num_workers != 0:
                raise ValueError("remote pretokenized mixtures require --num-workers 0")
            if getattr(self, "_lineage_config_mode", False):
                args.stage, args.model_preset = self._lineage_stage, "tr-hash-200m"
                report = validate_lineage_training_args(args, world_size=self._audit_world_size)
                args.max_steps, args.target_tokens = report["max_steps"], report["scheduled_tokens"]
                args.lineage_audit = report
                return
            if args.tokenized_data:
                if args.init_checkpoint:
                    validate_lineage_plans(
                        stage="refinement",
                        tokenized_plan=args.tokenized_plan,
                        pretrain_plan=args.pretrain_plan,
                    )
                if args.tokenized_plan and Path(args.tokenized_plan).is_file():
                    report = resolve_lineage_schedule(
                        load_lineage_plan(args.tokenized_plan),
                        world_size=getattr(
                            self, "_audit_world_size", int(os.environ.get("WORLD_SIZE", "1"))
                        ),
                        batch_size=args.batch_size,
                        gradient_accumulation=args.gradient_accumulation,
                        sequence_length=args.seq_len,
                        max_steps=args.max_steps,
                        target_tokens=args.target_tokens,
                    )
                    args.max_steps, args.target_tokens = (
                        report["max_steps"],
                        report["scheduled_tokens"],
                    )
                    args.lineage_audit = report
            elif args.init_checkpoint:
                raise ValueError("text refinement requires audited pretokenized data and plans")

        return _AuditedParser(parent, audit, getattr(self, "_configured_arguments", None))

    def build_dataset(self, tokenizer, args, rank, world_size):
        if not args.tokenized_data:
            if getattr(args, "init_checkpoint", None):
                raise ValueError("text refinement requires audited pretokenized data and plans")
            return WeightedStreamingTextDataset(
                tokenizer=tokenizer,
                seq_len=args.seq_len,
                sources=corpus_sources(args.stack_edu_data or ""),
                rank=rank,
                world_size=world_size,
            )
        if str(args.tokenized_data).startswith("hf://") and getattr(args, "num_workers", 0) != 0:
            raise ValueError("remote pretokenized mixtures require --num-workers 0")
        if getattr(args, "init_checkpoint", None):
            validate_lineage_plans(
                stage="refinement",
                tokenized_plan=args.tokenized_plan,
                pretrain_plan=getattr(args, "pretrain_plan", None),
            )
        if not hasattr(args, "resume"):
            args.resume = None
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
            raise ValueError("tokenized sequence length differs from the training context")
        try:
            resolve_lineage_schedule(
                {"trained_tokens": dataset.trained_tokens},
                world_size=world_size,
                batch_size=args.batch_size,
                gradient_accumulation=args.gradient_accumulation,
                sequence_length=args.seq_len,
                max_steps=args.max_steps,
                target_tokens=args.target_tokens,
            )
        except ValueError as exc:
            raise ValueError(f"{exc}; automatic dataset repetition is forbidden") from exc
        return dataset

    def build_compute_loss(self, trainer, model):
        return TRHashTextLineageRunner.build_compute_loss(self, trainer, model)


def build_200m_runner() -> TRHash200MPretrainRunner:
    return TRHash200MPretrainRunner(
        make_config=make_200m_config,
        run_name="tr-hash-200m-200b",
        checkpoint_dir="artifacts/tr_hash_200m_200b",
        default_lr=3e-4,
        default_batch_size=8,
        default_seq_len=1024,
        default_target_tokens=TARGET_TOKENS,
        default_gradient_accumulation=8,
        default_gradient_checkpointing=False,
        default_save_steps=0,
        default_distributed_mode="ddp",
        optimizer_type="adamw",
        label_smoothing=0.0,
    )


def build_replay_command(
    plan_path: str | Path,
    *,
    tokenizer: str,
    data: str,
    revision: str,
    output: str,
    cache_dir: str,
    world_size: int = 8,
    batch_size: int = 8,
    gradient_accumulation: int = 8,
    sequence_length: int = 1024,
    target_tokens: int | None = None,
    cache_gb: float = 24.0,
) -> tuple[str, ...]:
    """Return explicit argv after bounding the immutable replay plan; no launch."""
    plan = load_lineage_plan(plan_path)
    if target_tokens is not None and target_tokens != plan["trained_tokens"]:
        raise ValueError("target_tokens conflicts with replay plan trained_tokens")
    schedule = resolve_lineage_schedule(
        plan,
        world_size=world_size,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        sequence_length=sequence_length,
    )
    options = {
        "tokenizer": tokenizer,
        "tokenized-data": data,
        "tokenized-revision": revision,
        "tokenized-plan": str(plan_path),
        "tokenized-cache-dir": cache_dir,
        "tokenized-cache-gb": cache_gb,
        "tokenized-prefetch-shards": 1,
        "target-tokens": schedule["scheduled_tokens"],
        "max-steps": schedule["max_steps"],
        "token-packs": TOKEN_PACKS,
        "batch-size": batch_size,
        "seq-len": sequence_length,
        "gradient-accumulation": gradient_accumulation,
        "lr": 3e-4,
        "warmup-tokens": 1_000_000_000,
        "lr-scheduler": "wsd",
        "precision": "bf16",
        "distributed-mode": "ddp",
        "optimizer": "adamw",
        "use-custom-kernels": "auto",
        "save-steps": 0,
        "log-steps": 10,
        "num-workers": 0,
        "checkpoint-dir": output,
        "resume": "auto",
    }
    return (
        sys.executable,
        "-u",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(world_size),
        "-m",
        "complexity.training.text_pretraining",
        "--no-gradient-checkpointing",
        *(part for name, value in options.items() for part in (f"--{name}", str(value))),
    )


def main():
    build_200m_runner().run()


if __name__ == "__main__":
    main()
