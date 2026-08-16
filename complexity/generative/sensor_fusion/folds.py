"""Reproducible cross-subject folds for CUHK-X model selection."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from .cuhkx_records import CUHKX_TRAIN_USERS, CUHKXRecord


@dataclass(frozen=True)
class CrossSubjectFold:
    """One held-out subject group used only for validation."""

    name: str
    validation_users: tuple[int, ...]


# The three groups were balanced against the official training manifest.  They
# cover every official training subject exactly once, contain all 40 actions in
# each validation split, and have 1,057 / 1,008 / 971 examples respectively.
CUHKX_CROSS_SUBJECT_FOLDS: tuple[CrossSubjectFold, ...] = (
    CrossSubjectFold("fold_a", (1, 2, 8, 9, 16, 22)),
    CrossSubjectFold("fold_b", (3, 5, 7, 18, 19, 21)),
    CrossSubjectFold("fold_c", (4, 6, 17, 20, 23, 24)),
)


def resolve_cross_subject_fold(name: str) -> CrossSubjectFold:
    normalized = name.strip().lower().replace("-", "_")
    aliases = {"a": "fold_a", "b": "fold_b", "c": "fold_c"}
    normalized = aliases.get(normalized, normalized)
    for fold in CUHKX_CROSS_SUBJECT_FOLDS:
        if fold.name == normalized:
            return fold
    choices = ", ".join(fold.name for fold in CUHKX_CROSS_SUBJECT_FOLDS)
    raise ValueError(f"unknown CUHK-X cross-subject fold {name!r}; choose {choices}")


def validate_cross_subject_folds(
    folds: Sequence[CrossSubjectFold] = CUHKX_CROSS_SUBJECT_FOLDS,
    *,
    records: Iterable[CUHKXRecord] | None = None,
    num_classes: int = 40,
) -> dict:
    """Prove disjoint subject coverage and optional per-fold class coverage."""

    if not folds:
        raise ValueError("at least one cross-subject fold is required")
    names = [fold.name for fold in folds]
    if len(set(names)) != len(names):
        raise ValueError("cross-subject fold names must be unique")

    flattened = [user for fold in folds for user in fold.validation_users]
    duplicates = sorted(user for user, count in Counter(flattened).items() if count > 1)
    if duplicates:
        raise ValueError(
            "cross-subject validation users overlap: "
            + ",".join(map(str, duplicates))
        )
    expected = set(CUHKX_TRAIN_USERS)
    actual = set(flattened)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"cross-subject folds must cover official train users; missing={missing} "
            f"extra={extra}"
        )

    report = {
        "official_train_users": list(CUHKX_TRAIN_USERS),
        "folds": [
            {
                "name": fold.name,
                "validation_users": list(fold.validation_users),
            }
            for fold in folds
        ],
    }
    if records is None:
        return report

    records = list(records)
    available_users = {record.user_id for record in records}
    if not expected.issubset(available_users):
        missing = sorted(expected - available_users)
        raise ValueError(f"manifest lacks official CUHK-X train users: {missing}")
    for fold, item in zip(folds, report["folds"]):
        counts = Counter(
            record.action_id
            for record in records
            if record.user_id in fold.validation_users
        )
        missing_classes = [class_id for class_id in range(num_classes) if not counts[class_id]]
        if missing_classes:
            raise ValueError(
                f"{fold.name} lacks validation actions: {missing_classes}"
            )
        item.update(
            {
                "validation_examples": sum(counts.values()),
                "minimum_class_examples": min(counts.values()),
                "maximum_class_examples": max(counts.values()),
                "class_examples": [counts[class_id] for class_id in range(num_classes)],
            }
        )
    return report
