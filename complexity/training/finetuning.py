"""Framework-wide policy for supervised fine-tuning methods."""

from __future__ import annotations

TEXT_SUPERVISED_FINETUNING = "supervised-finetuning"
VISION_SUPERVISED_FINETUNING = "vision-supervised-finetuning"

# Full-parameter adaptation is deliberately exceptional. Text SFT remains
# LoRA-only; detector/vision refinement may update the complete model because
# it operates on a task checkpoint rather than a language instruction corpus.
FULL_PARAMETER_FINETUNING_PIPELINES = frozenset({VISION_SUPERVISED_FINETUNING})


def validate_full_parameter_finetuning(pipeline: str) -> None:
    """Reject full-parameter SFT unless the pipeline is explicitly exempted."""

    if pipeline not in FULL_PARAMETER_FINETUNING_PIPELINES:
        raise ValueError(
            "full-parameter SFT is restricted to "
            f"{VISION_SUPERVISED_FINETUNING!r}; got {pipeline!r}"
        )
