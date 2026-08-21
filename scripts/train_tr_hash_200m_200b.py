#!/usr/bin/env python3
"""Pretrain the canonical ~200M TR-Hash text model on 200B real tokens.

The six physical streams implement the approved five-family distribution:

* 45% DCLM (90B tokens);
* 30% FineWeb-Edu-Dedup (60B tokens);
* 10% Stack-Edu (20B tokens);
* 5% FineMath + 5% InfiWebMath (20B math tokens total);
* 5% Cosmopedia v2 (10B tokens).

Documents are packed losslessly into full causal windows. Token packs are
checkpoint boundaries only; they do not compress or repeat the token budget.
"""

from __future__ import annotations

import json
import os

from complexity.config import ModelConfig
from complexity.training import (
    TEXT_CONTINUED_PRETRAINING,
    PretokenizedCorpusMixtureDataset,
    TextCorpusSource,
    WeightedStreamingTextDataset,
    validate_full_parameter_finetuning,
)
from complexity.training.runner import TrainRunner
from complexity.utils.checkpointing import peek_latest_checkpoint_step

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

# The actual unique_tokens of the pretrain corpus this 200M run was trained
# on (configs/replay_plans/tr_hash_70b_quality_replay.json's "unique_tokens"
# -- row_alignment=512 rounds this slightly below the nominal 70B target from
# scripts/build_tr_hash_70b_replay_plan.py's DEFAULT_UNIQUE_BUDGETS, so this
# must be the exact recorded value, not 70_000_000_000). --init-checkpoint (a
# full-parameter phase-2 refinement pass, see
# finetuning.TEXT_CONTINUED_PRETRAINING) is only permitted when the plan
# being trained on has this same unique_tokens total, proving it's a single
# clean pass over the exact same corpus rather than a small
# language-instruction dataset routed around the LoRA-only guard.
PRETRAIN_UNIQUE_TOKENS = 69_997_690_880


def resume_skip_rows_for(args) -> int:
    """Rows this rank's PretokenizedCorpusMixtureDataset stream must skip on
    resume so the replay plan doesn't re-serve rows already trained on.

    Without this, every process restart re-instantiates the dataset and its
    deterministic __iter__() starts over at phase 1 / shard 0 -- observed
    live tonight on the 200M run: a supervisorctl restart at step 35603 sent
    DCLM's shard cursor back to shard 0, re-serving ~28B tokens' worth of
    already-trained-on rows (DCLM is meant to get exactly one pass, no
    replay) before it caught back up to new material.

    global_step * batch_size * gradient_accumulation is the number of rows
    THIS rank has pulled from its shard of the stream so far -- not
    multiplied by world_size, since each rank's iterator is already
    restricted to its own 1/world_size slice.
    """
    if not getattr(args, "resume", None):
        return 0
    if args.resume == "auto":
        step = peek_latest_checkpoint_step(args.checkpoint_dir)
    else:
        state_path = os.path.join(args.resume, "training_state.json")
        step = json.load(open(state_path))["step"] if os.path.exists(state_path) else None
    if step is None:
        return 0
    return step * args.batch_size * args.gradient_accumulation


def make_config() -> ModelConfig:
    """~200M profile with a wide shared SwiGLU and narrow routed experts."""

    return ModelConfig(
        hidden_size=896,
        num_hidden_layers=16,
        num_attention_heads=14,
        num_key_value_heads=2,
        intermediate_size=256,
        shared_intermediate_size=3072,
        vocab_size=32000,
        max_position_embeddings=2048,
        attention_type="gqa",
        mlp_type="tr_hash_engine",
        num_experts=4,
        top_k=2,
        top_k_primary_weight=0.5,
        routing_strategy="token_id_multi_hash",
        route_hash_count=2,
        shared_expert=True,
        shared_output_scale=1.0,
        routed_output_scale=2.0,
        use_qk_norm=True,
        tie_word_embeddings=True,
    )


def corpus_sources(stack_edu_data: str) -> tuple[TextCorpusSource, ...]:
    """Return the immutable 200B mixture; Stack-Edu must be materialized."""

    if not stack_edu_data:
        raise ValueError(
            "--stack-edu-data is required: the official Stack-Edu repository "
            "contains SWHIDs, not source text"
        )
    return (
        TextCorpusSource(
            name="dclm",
            weight=0.45,
            dataset_id="mlfoundations/dclm-baseline-1.0-parquet",
        ),
        TextCorpusSource(
            name="fineweb_edu_dedup",
            weight=0.30,
            dataset_id="HuggingFaceTB/smollm-corpus",
            config_name="fineweb-edu-dedup",
        ),
        TextCorpusSource(
            name="stack_edu",
            weight=0.10,
            data_files=stack_edu_data,
        ),
        TextCorpusSource(
            name="finemath",
            weight=0.05,
            dataset_id="HuggingFaceTB/finemath",
            config_name="finemath-3plus",
        ),
        TextCorpusSource(
            name="infiwebmath",
            weight=0.05,
            dataset_id="HuggingFaceTB/finemath",
            config_name="infiwebmath-3plus",
        ),
        TextCorpusSource(
            name="cosmopedia_v2",
            weight=0.05,
            dataset_id="HuggingFaceTB/smollm-corpus",
            config_name="cosmopedia-v2",
        ),
    )


class TRHash200MPretrainRunner(TrainRunner):
    def add_args(self, parser) -> None:
        parser.set_defaults(token_packs=TOKEN_PACKS)
        parser.add_argument(
            "--stack-edu-data",
            default=None,
            help="Local JSONL/JSONL.GZ path or glob with materialized Stack-Edu text.",
        )
        parser.add_argument(
            "--tokenized-data",
            default=None,
            help=(
                "Pretokenized mixture root or hf://datasets/<owner>/<repo>. "
                "Hub shards are cached lazily for production training."
            ),
        )
        parser.add_argument(
            "--tokenized-cache-dir",
            default=os.environ.get("TR_HASH_TOKEN_CACHE", "artifacts/tr_hash_token_cache"),
            help="Bounded local NVMe cache used by remote token mixtures.",
        )
        parser.add_argument(
            "--tokenized-cache-gb",
            type=float,
            default=32.0,
            help="Maximum local .bin cache size for a remote token mixture.",
        )
        parser.add_argument(
            "--tokenized-revision",
            default="main",
            help="Hugging Face dataset revision used by a remote token mixture.",
        )
        parser.add_argument(
            "--tokenized-prefetch-shards",
            type=int,
            default=1,
            help="Number of upcoming Hub shards downloaded in the background.",
        )
        parser.add_argument(
            "--tokenized-hf-token-env",
            default="HF_TOKEN",
            help="Environment variable containing the optional Hub read token.",
        )
        parser.add_argument(
            "--tokenized-plan",
            default=None,
            help=(
                "Optional zero-copy replay-plan JSON. For Hub data this may be "
                "a path stored inside the dataset repository."
            ),
        )

    def build_dataset(self, tokenizer, args, rank: int, world_size: int):
        if args.tokenized_data:
            if (
                str(args.tokenized_data).startswith("hf://")
                and getattr(args, "num_workers", 0) != 0
            ):
                raise ValueError(
                    "remote pretokenized mixtures require --num-workers 0; "
                    "multiple DataLoader workers fetch independent multi-GiB shards "
                    "before the first optimizer step and can thrash the bounded cache"
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
                resume_skip_rows=resume_skip_rows_for(args),
            )
            if dataset.seq_len != args.seq_len:
                raise ValueError(
                    f"tokenized seq_len={dataset.seq_len} does not match --seq-len={args.seq_len}"
                )
            if getattr(args, "init_checkpoint", None):
                validate_full_parameter_finetuning(
                    TEXT_CONTINUED_PRETRAINING,
                    unique_tokens=dataset.unique_tokens,
                    pretrain_unique_tokens=PRETRAIN_UNIQUE_TOKENS,
                )
            tokens_per_step = (
                args.batch_size
                * world_size
                * args.gradient_accumulation
                * args.seq_len
            )
            planned_steps = (
                args.max_steps
                if args.max_steps is not None
                else args.target_tokens // tokens_per_step
            )
            scheduled_tokens = planned_steps * tokens_per_step
            if scheduled_tokens > dataset.trained_tokens:
                raise ValueError(
                    "training schedule exceeds the pretokenized data coverage: "
                    f"dataset={dataset.trained_tokens:,} tokens, "
                    f"schedule={scheduled_tokens:,} tokens. Set --target-tokens to at "
                    "most the replay plan's trained_tokens; automatic dataset repetition "
                    "is forbidden."
                )
            return dataset
        return WeightedStreamingTextDataset(
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            sources=corpus_sources(args.stack_edu_data or ""),
            rank=rank,
            world_size=world_size,
        )

    def extra_callbacks(self, trainer, args, is_main: bool):
        from complexity.training.callbacks import TensorBoardCallback

        return [TensorBoardCallback(log_dir=os.path.join(args.checkpoint_dir, "tensorboard"))]


def build_runner() -> TRHash200MPretrainRunner:
    return TRHash200MPretrainRunner(
        make_config=make_config,
        run_name="tr-hash-200m-200b",
        checkpoint_dir="artifacts/tr_hash_200m_200b",
        default_lr=3.0e-4,
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


def main() -> None:
    build_runner().run()


if __name__ == "__main__":
    main()
