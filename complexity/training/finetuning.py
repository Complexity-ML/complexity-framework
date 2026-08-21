"""Framework-wide policy for supervised fine-tuning methods."""

from __future__ import annotations

TEXT_SUPERVISED_FINETUNING = "supervised-finetuning"
VISION_SUPERVISED_FINETUNING = "vision-supervised-finetuning"
IMAGE_GENERATION_SUPERVISED_FINETUNING = "image-generation-supervised-finetuning"
IMAGE_TEXT_TO_TEXT_SUPERVISED_FINETUNING = "image-text-to-text-supervised-finetuning"
TEXT_CONTINUED_PRETRAINING = "text-continued-pretraining"

# Full-parameter adaptation is deliberately explicit. Text SFT may update the
# complete model only when its trainer selects the dedicated full-parameter
# mode; LoRA remains the default. Detector/vision refinement and text-to-image
# aesthetic SFT may also update the complete model.
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
#
# text-continued-pretraining is the text analogue of vision's phase-2
# refinement (scripts/vast_finetune_detector_coco_v08_nano.sh reruns the
# same COCO images with augmentation stripped, not a new dataset): a single
# clean pass, fresh optimizer/scheduler, over the *exact same* pretrain
# corpus with its replay repetition removed. It is exempted on the same
# grounds -- it operates on the pretrain corpus itself, not a language
# instruction corpus -- but only that one narrow case, so
# validate_full_parameter_finetuning additionally requires proof: its plan's
# unique_tokens must exactly match the completed pretrain's unique_tokens.
# Without that check, this exemption would be a generic backdoor around the
# LoRA-only restriction for any small instruction corpus.
FULL_PARAMETER_FINETUNING_PIPELINES = frozenset(
    {
        VISION_SUPERVISED_FINETUNING,
        IMAGE_GENERATION_SUPERVISED_FINETUNING,
        IMAGE_TEXT_TO_TEXT_SUPERVISED_FINETUNING,
        TEXT_SUPERVISED_FINETUNING,
        TEXT_CONTINUED_PRETRAINING,
    }
)


def validate_full_parameter_finetuning(
    pipeline: str,
    *,
    unique_tokens: int | None = None,
    pretrain_unique_tokens: int | None = None,
) -> None:
    """Reject full-parameter SFT unless the pipeline is explicitly allowed.

    text-continued-pretraining carries an extra condition: unique_tokens
    (the plan actually being trained on) must exactly equal
    pretrain_unique_tokens (the completed pretrain's unique_tokens) -- proof
    this run is a clean single pass over the same corpus, not a small
    language-instruction dataset sneaking through under the exemption's name.
    """

    if pipeline not in FULL_PARAMETER_FINETUNING_PIPELINES:
        allowed = ", ".join(repr(name) for name in sorted(FULL_PARAMETER_FINETUNING_PIPELINES))
        raise ValueError(
            f"full-parameter SFT is restricted to {allowed}; got {pipeline!r}"
        )
    if pipeline == TEXT_CONTINUED_PRETRAINING:
        if unique_tokens is None or pretrain_unique_tokens is None:
            raise ValueError(
                "text-continued-pretraining requires unique_tokens and "
                "pretrain_unique_tokens to prove the plan reuses the exact "
                "pretrain corpus rather than a language-instruction dataset"
            )
        if unique_tokens != pretrain_unique_tokens:
            raise ValueError(
                f"text-continued-pretraining requires unique_tokens "
                f"({unique_tokens:,}) to exactly match the completed "
                f"pretrain's unique_tokens ({pretrain_unique_tokens:,}); a "
                f"mismatch means this isn't a clean single pass over the "
                f"same corpus, so it must go through the LoRA-only text-SFT "
                f"pipeline instead"
            )
