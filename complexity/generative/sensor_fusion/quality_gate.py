"""Evidence gate required before CUHK-X test inference can emit a CSV."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def _checkpoint_contract(checkpoint: Path) -> dict[str, Any]:
    metadata = json.loads((checkpoint / "config.json").read_text())
    return {
        "model": metadata["model"],
        "preprocessing": metadata["preprocessing"],
    }


def validate_cross_subject_quality(
    candidate_checkpoint: str | Path,
    validation_checkpoints: Iterable[str | Path],
    *,
    minimum_folds: int = 3,
    minimum_fold_top1: float = 0.86,
) -> dict[str, Any]:
    """Require compatible, subject-disjoint folds before test prediction."""

    if minimum_folds < 2:
        raise ValueError("minimum_folds must be at least 2")
    if not 0.0 < minimum_fold_top1 <= 1.0:
        raise ValueError("minimum_fold_top1 must be in (0, 1]")
    candidate = Path(candidate_checkpoint)
    candidate_contract = _checkpoint_contract(candidate)
    checkpoints = tuple(Path(value) for value in validation_checkpoints)
    if len(checkpoints) < minimum_folds:
        raise ValueError(
            f"submission requires at least {minimum_folds} cross-subject folds"
        )

    folds = []
    seen_users: set[int] = set()
    for checkpoint in checkpoints:
        if _checkpoint_contract(checkpoint) != candidate_contract:
            raise ValueError(
                f"validation checkpoint is incompatible with candidate: {checkpoint}"
            )
        metrics = json.loads((checkpoint / "metrics.json").read_text())
        users = tuple(int(value) for value in metrics.get("validation_users", ()))
        if not users:
            raise ValueError(
                f"validation checkpoint lacks cross-subject user evidence: {checkpoint}"
            )
        overlap = seen_users.intersection(users)
        if overlap:
            raise ValueError(
                "validation folds must use disjoint held-out users; overlap="
                + ",".join(map(str, sorted(overlap)))
            )
        seen_users.update(users)
        top1 = float(metrics["top1_accuracy"])
        macro = float(metrics["macro_accuracy"])
        if top1 < minimum_fold_top1:
            raise ValueError(
                f"fold {checkpoint} top1={top1:.4f} is below "
                f"required {minimum_fold_top1:.4f}"
            )
        folds.append(
            {
                "checkpoint": str(checkpoint),
                "validation_users": list(users),
                "top1_accuracy": top1,
                "macro_accuracy": macro,
            }
        )

    return {
        "passed": True,
        "candidate_checkpoint": str(candidate),
        "minimum_folds": minimum_folds,
        "minimum_fold_top1": minimum_fold_top1,
        "folds": folds,
        "mean_top1_accuracy": sum(item["top1_accuracy"] for item in folds)
        / len(folds),
        "worst_top1_accuracy": min(item["top1_accuracy"] for item in folds),
        "held_out_users": sorted(seen_users),
    }
