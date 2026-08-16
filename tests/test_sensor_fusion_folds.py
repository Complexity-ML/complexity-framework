from pathlib import Path

import pytest

from complexity.generative.sensor_fusion import (
    CUHKX_CROSS_SUBJECT_FOLDS,
    CrossSubjectFold,
    CUHKXRecord,
    resolve_cross_subject_fold,
    validate_cross_subject_folds,
)


def test_default_cross_subject_folds_cover_every_train_user_once():
    report = validate_cross_subject_folds()
    assert len(report["folds"]) == 3
    users = [user for fold in report["folds"] for user in fold["validation_users"]]
    assert len(users) == len(set(users)) == 18
    assert resolve_cross_subject_fold("A") == CUHKX_CROSS_SUBJECT_FOLDS[0]
    assert resolve_cross_subject_fold("fold-c") == CUHKX_CROSS_SUBJECT_FOLDS[2]


def test_cross_subject_folds_reject_overlap_and_missing_users():
    bad = (
        CrossSubjectFold("fold_a", (1, 2)),
        CrossSubjectFold("fold_b", (2, 3)),
    )
    with pytest.raises(ValueError, match="overlap"):
        validate_cross_subject_folds(bad)

    with pytest.raises(ValueError, match="cover official"):
        validate_cross_subject_folds((CrossSubjectFold("fold_a", (1, 2)),))


def test_manifest_validation_requires_all_actions_in_each_fold(tmp_path: Path):
    records = []
    for fold in CUHKX_CROSS_SUBJECT_FOLDS:
        user = fold.validation_users[0]
        for class_id in range(40):
            records.append(
                CUHKXRecord(
                    action_id=class_id,
                    action_name=f"action-{class_id}",
                    user_id=user,
                    trial_id=f"trial-{class_id}",
                    paths={},
                )
            )
    # Include every official user even when the synthetic user has no extra
    # class record. Fold coverage is supplied by the first user in each fold.
    present = {record.user_id for record in records}
    for fold in CUHKX_CROSS_SUBJECT_FOLDS:
        for user in fold.validation_users:
            if user not in present:
                records.append(CUHKXRecord(0, "action-0", user, "trial-0", {}))
    report = validate_cross_subject_folds(records=records)
    assert [item["minimum_class_examples"] for item in report["folds"]] == [1, 1, 1]

    incomplete = [record for record in records if not (
        record.user_id in CUHKX_CROSS_SUBJECT_FOLDS[0].validation_users
        and record.action_id == 39
    )]
    with pytest.raises(ValueError, match="lacks validation actions"):
        validate_cross_subject_folds(records=incomplete)
