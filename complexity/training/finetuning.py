"""Framework-wide policy for supervised fine-tuning methods."""

from __future__ import annotations

TEXT_SUPERVISED_FINETUNING = "supervised-finetuning"
VISION_SUPERVISED_FINETUNING = "vision-supervised-finetuning"
IMAGE_GENERATION_SUPERVISED_FINETUNING = "image-generation-supervised-finetuning"
IMAGE_TEXT_TO_TEXT_SUPERVISED_FINETUNING = "image-text-to-text-supervised-finetuning"

# Full-parameter adaptation is deliberately exceptional. Text SFT remains
# LoRA-only; detector/vision refinement and text-to-image aesthetic SFT may
# update the complete model because they operate on a task checkpoint or an
# image corpus rather than a language instruction corpus.
#
# image-text-to-text SFT is exempted too, but only in the narrow sense that
# matches this rationale: its curated stage trains on an image-grounded
# question/answer/dialogue corpus (the language-instruction shape this
# restriction exists to guard), so the pipeline freezes the language decoder
# entirely for that stage -- only vision parameters (tower/resampler/
# projection) update. The exemption covers "full-parameter vision update",
# never full-parameter text update; see
# complexity/generative/vision_language/training.py's
# freeze_decoder_for_vision_only_sft.
FULL_PARAMETER_FINETUNING_PIPELINES = frozenset(
    {
        VISION_SUPERVISED_FINETUNING,
        IMAGE_GENERATION_SUPERVISED_FINETUNING,
        IMAGE_TEXT_TO_TEXT_SUPERVISED_FINETUNING,
    }
)


def validate_full_parameter_finetuning(pipeline: str) -> None:
    """Reject full-parameter SFT unless the pipeline is explicitly exempted."""

    if pipeline not in FULL_PARAMETER_FINETUNING_PIPELINES:
        allowed = ", ".join(repr(name) for name in sorted(FULL_PARAMETER_FINETUNING_PIPELINES))
        raise ValueError(
            f"full-parameter SFT is restricted to {allowed}; got {pipeline!r}"
        )
