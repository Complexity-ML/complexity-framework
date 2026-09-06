"""Fail-closed contract for the canonical Agentic 100M refinement recipe."""

from __future__ import annotations

import argparse
import math
import sys

REFINEMENT_PEAK_LR = 1e-4
REFINEMENT_TOKENS_PER_STEP = 3_932_160
REFINEMENT_WARMUP_TOKENS = 500_000_000
REFINEMENT_WEIGHT_DECAY = 0.1
REFINEMENT_LR_SCHEDULER = "cosine"


def validate_refinement_recipe(
    *,
    stage: str,
    model_preset: str,
    learning_rate: float,
    lr_scheduler: str,
    warmup_tokens: int | None,
    warmup_steps: int | None,
    weight_decay: float,
    tokens_per_step: int,
) -> None:
    """Require the optimizer regime measured on the released 200M refit."""
    if stage != "refinement" or model_preset != "complexity-100m":
        return

    mismatches: list[str] = []
    if not math.isclose(learning_rate, REFINEMENT_PEAK_LR, rel_tol=0.0, abs_tol=1e-12):
        mismatches.append(f"lr={learning_rate!r} (expected {REFINEMENT_PEAK_LR})")
    if lr_scheduler != REFINEMENT_LR_SCHEDULER:
        mismatches.append(f"lr_scheduler={lr_scheduler!r} (expected {REFINEMENT_LR_SCHEDULER!r})")
    if warmup_steps is not None or warmup_tokens != REFINEMENT_WARMUP_TOKENS:
        mismatches.append(
            "warmup must use "
            f"{REFINEMENT_WARMUP_TOKENS} tokens (got tokens={warmup_tokens!r}, "
            f"steps={warmup_steps!r})"
        )
    if not math.isclose(weight_decay, REFINEMENT_WEIGHT_DECAY, rel_tol=0.0, abs_tol=1e-12):
        mismatches.append(f"weight_decay={weight_decay!r} (expected {REFINEMENT_WEIGHT_DECAY})")
    if tokens_per_step != REFINEMENT_TOKENS_PER_STEP:
        mismatches.append(
            f"tokens_per_step={tokens_per_step:,} (expected {REFINEMENT_TOKENS_PER_STEP:,})"
        )
    if mismatches:
        raise ValueError(
            "Agentic 100M canonical refinement must reproduce the observed 200M "
            "optimizer regime: " + "; ".join(mismatches)
        )


def main() -> None:
    (
        stage,
        nproc,
        batch_size,
        gradient_accumulation,
        seq_len,
        default_lr,
        default_scheduler,
        default_warmup_tokens,
        default_weight_decay,
        *trainer_args,
    ) = sys.argv[1:]
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--batch-size", type=int, default=int(batch_size))
    parser.add_argument("--gradient-accumulation", type=int, default=int(gradient_accumulation))
    parser.add_argument("--seq-len", type=int, default=int(seq_len))
    parser.add_argument("--lr", type=float, default=float(default_lr))
    parser.add_argument("--lr-scheduler", default=default_scheduler)
    parser.add_argument("--warmup-tokens", type=int, default=int(default_warmup_tokens))
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=float(default_weight_decay))
    args, _ = parser.parse_known_args(trainer_args)
    try:
        validate_refinement_recipe(
            stage=stage,
            model_preset="complexity-100m",
            learning_rate=args.lr,
            lr_scheduler=args.lr_scheduler,
            warmup_tokens=args.warmup_tokens,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            tokens_per_step=(
                int(nproc) * args.batch_size * args.gradient_accumulation * args.seq_len
            ),
        )
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
