"""Framework-wide policy for refinement and supervised fine-tuning."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

PRETRAINING_STAGE = "pretraining"
REFINEMENT_STAGE = "refinement"
SUPERVISED_FINETUNING_STAGE = "supervised-finetuning"

TEXT_MODEL_FAMILY = "text"
VISION_MODEL_FAMILY = "vision"
IMAGE_GENERATION_MODEL_FAMILY = "image-generation"
IMAGE_EDITING_MODEL_FAMILY = "image-editing"
IMAGE_TEXT_TO_TEXT_MODEL_FAMILY = "image-text-to-text"

TEXT_SUPERVISED_FINETUNING = "supervised-finetuning"
VISION_SUPERVISED_FINETUNING = "vision-supervised-finetuning"
IMAGE_GENERATION_SUPERVISED_FINETUNING = "image-generation-supervised-finetuning"
IMAGE_EDITING_SUPERVISED_FINETUNING = "image-editing-supervised-finetuning"
IMAGE_TEXT_TO_TEXT_SUPERVISED_FINETUNING = "image-text-to-text-supervised-finetuning"
TEXT_REFINEMENT = "text-refinement"

# Compatibility alias for older imports. The old name was ambiguous: this is
# a fresh-optimizer refinement stage, not a generic continued-pretraining run.
TEXT_CONTINUED_PRETRAINING = TEXT_REFINEMENT

# Vision is the sole exception to a separate pretrain -> refinement -> SFT
# transition. Its canonical recipe already performs its refinement inside the
# pretraining lineage by annealing augmentation and finishing on clean images.
INTEGRATED_REFINEMENT_MODEL_FAMILIES = frozenset({VISION_MODEL_FAMILY})

FULL_PARAMETER_FINETUNING_PIPELINES = frozenset(
    {
        VISION_SUPERVISED_FINETUNING,
        IMAGE_GENERATION_SUPERVISED_FINETUNING,
        IMAGE_EDITING_SUPERVISED_FINETUNING,
        IMAGE_TEXT_TO_TEXT_SUPERVISED_FINETUNING,
        TEXT_SUPERVISED_FINETUNING,
        TEXT_REFINEMENT,
    }
)

_KNOWN_STAGES = frozenset(
    {PRETRAINING_STAGE, REFINEMENT_STAGE, SUPERVISED_FINETUNING_STAGE}
)


def validate_training_stage_transition(
    model_family: str,
    source_stage: str,
    target_stage: str,
    *,
    same_corpus: bool | None = None,
    fresh_optimizer: bool | None = None,
) -> None:
    """Validate the framework's stage-order contract.

    Non-Vision lineages must follow ``pretraining -> refinement -> SFT``.
    Additional SFT stages may start from an existing SFT checkpoint. A
    refinement is deliberately narrow: it starts from pretraining weights,
    uses the exact same corpus, and creates fresh optimization state.

    Vision is the only exception because the detector recipe already embeds
    clean-data refinement in its pretraining lineage.
    """

    if source_stage not in _KNOWN_STAGES:
        raise ValueError(f"unknown source training stage: {source_stage!r}")
    if target_stage not in _KNOWN_STAGES:
        raise ValueError(f"unknown target training stage: {target_stage!r}")

    if target_stage == REFINEMENT_STAGE:
        if source_stage != PRETRAINING_STAGE:
            raise ValueError("refinement must start from a pretraining checkpoint")
        if same_corpus is not True:
            raise ValueError("refinement must reuse the exact pretraining corpus")
        if fresh_optimizer is not True:
            raise ValueError("refinement must start with a fresh optimizer and scheduler")
        return

    if target_stage == SUPERVISED_FINETUNING_STAGE:
        if model_family in INTEGRATED_REFINEMENT_MODEL_FAMILIES:
            if source_stage not in {PRETRAINING_STAGE, REFINEMENT_STAGE}:
                raise ValueError(
                    "Vision SFT must start from its pretraining lineage or an "
                    "explicit refinement checkpoint"
                )
            return
        if source_stage not in {REFINEMENT_STAGE, SUPERVISED_FINETUNING_STAGE}:
            raise ValueError(
                f"{model_family} SFT requires a refinement checkpoint; direct "
                "pretraining -> SFT is forbidden"
            )
        return

    if target_stage == PRETRAINING_STAGE:
        raise ValueError("pretraining is a lineage root, not a transition target")


def _unique_core(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    phases = plan.get("phases")
    if not isinstance(phases, list):
        raise ValueError("replay plan must contain a phases list")
    matches = [phase for phase in phases if phase.get("name") == "unique_core"]
    if len(matches) != 1:
        raise ValueError("replay plan must contain exactly one unique_core phase")
    phase = matches[0]
    if phase.get("passes") != 1:
        raise ValueError("unique_core must contain exactly one pass")
    if not isinstance(phase.get("sources"), Mapping) or not phase["sources"]:
        raise ValueError("unique_core must contain non-empty source selections")
    return phase


def refinement_corpus_fingerprint(plan: Mapping[str, Any]) -> str:
    """Hash the exact unique-corpus selection, independent of replay phases."""

    unique_core = _unique_core(plan)
    contract = {
        "format": plan.get("format"),
        "dataset": plan.get("dataset"),
        "revision": plan.get("revision"),
        "seq_len": plan.get("seq_len"),
        "selection_mode": plan.get("selection_mode"),
        "row_alignment": plan.get("row_alignment"),
        "unique_tokens": plan.get("unique_tokens"),
        "source_unique_tokens": plan.get("source_unique_tokens"),
        "sources": unique_core["sources"],
    }
    encoded = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_refinement_plan(
    refinement_plan: Mapping[str, Any],
    pretrain_plan: Mapping[str, Any],
) -> str:
    """Prove that a refinement plan is one clean pass over pretrain data.

    Token totals alone are insufficient: this compares the exact source/shard
    row selections used by both ``unique_core`` phases and rejects replay in
    the refinement plan. The returned SHA-256 fingerprint is suitable for
    audit metadata and logs.
    """

    validate_training_stage_transition(
        TEXT_MODEL_FAMILY,
        PRETRAINING_STAGE,
        REFINEMENT_STAGE,
        same_corpus=True,
        fresh_optimizer=True,
    )
    refinement_core = _unique_core(refinement_plan)
    _unique_core(pretrain_plan)
    if len(refinement_plan["phases"]) != 1:
        raise ValueError("refinement plan must contain only the unique_core phase")
    if int(refinement_plan.get("trained_tokens", -1)) != int(
        refinement_plan.get("unique_tokens", -2)
    ):
        raise ValueError("refinement plan must train each unique token exactly once")
    source_passes = refinement_plan.get("source_passes")
    if (
        not isinstance(source_passes, Mapping)
        or not source_passes
        or set(source_passes) != set(refinement_core["sources"])
        or any(int(passes) != 1 for passes in source_passes.values())
    ):
        raise ValueError("refinement plan must use exactly one pass per source")
    if refinement_core.get("passes") != 1:
        raise ValueError("refinement unique_core must use exactly one pass")

    expected = refinement_corpus_fingerprint(pretrain_plan)
    actual = refinement_corpus_fingerprint(refinement_plan)
    if actual != expected:
        raise ValueError(
            "refinement corpus does not exactly match the pretraining unique_core "
            f"(expected {expected}, got {actual})"
        )
    return actual


def validate_full_parameter_finetuning(
    pipeline: str,
    *,
    unique_tokens: int | None = None,
    pretrain_unique_tokens: int | None = None,
    source_stage: str | None = None,
    model_family: str | None = None,
) -> None:
    """Reject full-parameter adaptation unless its pipeline is explicit.

    New callers should additionally provide ``source_stage``. It is required
    for non-Vision SFT and proves that the stage did not skip refinement.
    Exact-corpus proof for text refinement is performed by
    :func:`validate_refinement_plan`; token totals remain a compatibility
    preflight for older launchers.
    """

    if pipeline not in FULL_PARAMETER_FINETUNING_PIPELINES:
        allowed = ", ".join(repr(name) for name in sorted(FULL_PARAMETER_FINETUNING_PIPELINES))
        raise ValueError(
            f"full-parameter SFT is restricted to {allowed}; got {pipeline!r}"
        )
    if pipeline == TEXT_REFINEMENT:
        if unique_tokens is None or pretrain_unique_tokens is None:
            raise ValueError(
                "text-refinement requires unique_tokens and pretrain_unique_tokens"
            )
        if unique_tokens != pretrain_unique_tokens:
            raise ValueError(
                f"text-refinement requires unique_tokens ({unique_tokens:,}) to "
                "exactly match the completed pretrain's unique_tokens "
                f"({pretrain_unique_tokens:,})"
            )
        return

    families = {
        TEXT_SUPERVISED_FINETUNING: TEXT_MODEL_FAMILY,
        VISION_SUPERVISED_FINETUNING: VISION_MODEL_FAMILY,
        IMAGE_GENERATION_SUPERVISED_FINETUNING: IMAGE_GENERATION_MODEL_FAMILY,
        IMAGE_EDITING_SUPERVISED_FINETUNING: IMAGE_EDITING_MODEL_FAMILY,
        IMAGE_TEXT_TO_TEXT_SUPERVISED_FINETUNING: IMAGE_TEXT_TO_TEXT_MODEL_FAMILY,
    }
    family = model_family or families[pipeline]
    if source_stage is not None:
        validate_training_stage_transition(
            family,
            source_stage,
            SUPERVISED_FINETUNING_STAGE,
        )
    elif family != VISION_MODEL_FAMILY:
        raise ValueError(
            f"{pipeline} requires source_stage={REFINEMENT_STAGE!r}; direct "
            "pretraining -> SFT is forbidden"
        )
