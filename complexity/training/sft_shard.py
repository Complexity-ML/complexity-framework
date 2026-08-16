"""Contracts shared by framework SFT token-shard readers."""

from __future__ import annotations

from typing import Any, Mapping

SHARD_FORMAT_V2 = "complexity-sft-token-shard-v2"
LEGACY_ALL_ASSISTANT_SUPERVISION = "all_assistant_turns"
FINAL_ASSISTANT_SUPERVISION = "final_assistant_only"
MASKED_ASSISTANT_HISTORY = "masked_context"


def validate_shard_supervision(metadata: Mapping[str, Any]) -> None:
    """Validate how a V2 shard maps conversation turns to loss labels.

    ``all_assistant_turns`` remains readable for the historical relabeled
    shards.  New multi-turn Card Corpus shards mask prior assistant history and
    supervise only the final assistant turn; that mode must declare the masked
    history contract explicitly so ambiguous shards fail closed.
    """

    if metadata.get("format") != SHARD_FORMAT_V2:
        return
    supervision = metadata.get("assistant_supervision")
    if supervision == LEGACY_ALL_ASSISTANT_SUPERVISION:
        return
    if supervision == FINAL_ASSISTANT_SUPERVISION:
        if metadata.get("history_assistant_turns") != MASKED_ASSISTANT_HISTORY:
            raise ValueError(
                "final-assistant-only SFT shards must declare masked assistant history"
            )
        return
    raise ValueError(
        "SFT shard v2 requires all-assistant-turn or final-assistant-only supervision"
    )


__all__ = (
    "FINAL_ASSISTANT_SUPERVISION",
    "LEGACY_ALL_ASSISTANT_SUPERVISION",
    "MASKED_ASSISTANT_HISTORY",
    "SHARD_FORMAT_V2",
    "validate_shard_supervision",
)
