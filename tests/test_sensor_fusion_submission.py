import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.torch import save_file

from complexity.generative.sensor_fusion import (
    CUHKXPreprocessingConfig,
    CUHKXSmallTrackTestDataset,
    TRHashSensorFusionClassifier,
    TRHashSensorFusionConfig,
    build_cuhkx_test_records,
    collate_cuhkx_test,
)
from complexity.generative.sensor_fusion.quality_gate import (
    validate_cross_subject_quality,
)
from complexity.generative.sensor_fusion.submission import (
    validate_submission,
    write_submission,
)
from tests.test_cuhkx_sensor_data import (
    MODALITY_FOLDERS,
    _trial,
    _write_complete_trial,
)


def _write_test_tree(root: Path, count: int = 2) -> tuple[Path, Path]:
    source = root / "source"
    _write_complete_trial(source, user=1)
    data_root = root / "test-data"
    paths = []
    for index in range(1, count + 1):
        relative = f"small_model_track_test/SM_test_{index:04d}/"
        sample = data_root / relative
        for modality, folder in MODALITY_FOLDERS.items():
            shutil.copytree(_trial(source, modality, 1), sample / folder)
        paths.append(relative)
    test_csv = root / "test.csv"
    with test_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("path", "prediction"))
        writer.writerows((path, "") for path in paths)
    return data_root, test_csv


def _tiny_config() -> TRHashSensorFusionConfig:
    return TRHashSensorFusionConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_experts=8,
        top_k=2,
        shared_width=16,
        expert_width=4,
        precision="fp32",
        visual_token_grid=(1, 1, 1),
        vision_image_size=16,
        vision_patch_size=4,
        vision_hidden_size=16,
        vision_layers=1,
        vision_heads=4,
        vision_num_experts=4,
        vision_top_k=2,
        vision_expert_width=4,
        vision_stage_depths=(1,),
        vision_window_size=2,
        sequence_tokens=2,
    )


def _preprocessing() -> CUHKXPreprocessingConfig:
    return CUHKXPreprocessingConfig(image_size=16, clip_frames=4, sensor_steps=8)


def test_official_test_index_and_dataset_preserve_csv_order(tmp_path):
    data_root, test_csv = _write_test_tree(tmp_path)
    records = build_cuhkx_test_records(data_root, test_csv)
    assert [record.index for record in records] == [0, 1]
    assert [record.sample_id for record in records] == ["SM_test_0001", "SM_test_0002"]
    assert [record.path for record in records] == [
        "small_model_track_test/SM_test_0001/",
        "small_model_track_test/SM_test_0002/",
    ]
    assert all(set(record.paths) == set(MODALITY_FOLDERS) for record in records)

    dataset = CUHKXSmallTrackTestDataset(
        data_root,
        test_csv,
        preprocessing=_preprocessing(),
        records=records,
    )
    batch = collate_cuhkx_test([dataset[1], dataset[0]])
    assert batch["indices"].tolist() == [1, 0]
    assert batch["inputs"]["imu"].shape == (2, 8, 45)
    assert batch["inputs"]["depth"].shape == (2, 3, 4, 16, 16)
    assert batch["modality_mask"]["radar"].all()


def test_test_index_rejects_any_populated_prediction(tmp_path):
    data_root, test_csv = _write_test_tree(tmp_path, count=1)
    test_csv.write_text(
        "path,prediction\nsmall_model_track_test/SM_test_0001/,17\n"
    )
    try:
        build_cuhkx_test_records(data_root, test_csv)
    except ValueError as error:
        assert "leakage" in str(error)
    else:
        raise AssertionError("test labels or predictions must never be consumed")


def test_submission_writer_and_validator_enforce_exact_order(tmp_path):
    _, test_csv = _write_test_tree(tmp_path)
    paths = [
        "small_model_track_test/SM_test_0001/",
        "small_model_track_test/SM_test_0002/",
    ]
    output = tmp_path / "submission.csv"
    write_submission(output, paths, [3, 39])
    assert validate_submission(output, test_csv) == {
        "rows": 2,
        "num_classes": 40,
        "predicted_classes": 2,
        "min_class_count": 0,
        "max_class_count": 1,
    }
    write_submission(output, list(reversed(paths)), [3, 39])
    try:
        validate_submission(output, test_csv)
    except ValueError as error:
        assert "ordering" in str(error)
    else:
        raise AssertionError("a reordered submission must fail validation")


def test_submission_cli_generates_csv_and_reusable_logits(monkeypatch, tmp_path):
    from complexity.generative.sensor_fusion import submission

    data_root, test_csv = _write_test_tree(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    model = TRHashSensorFusionClassifier(_tiny_config())
    save_file(
        {name: value.detach().contiguous() for name, value in model.state_dict().items()},
        str(checkpoint / "model.safetensors"),
    )
    contract = {
        "model": model.config.to_dict(),
        "preprocessing": _preprocessing().to_dict(),
    }
    (checkpoint / "config.json").write_text(json.dumps(contract))
    validation_checkpoints = []
    for index, users in enumerate(((1, 16), (2, 17), (3, 18))):
        fold = tmp_path / f"fold-{index}"
        fold.mkdir()
        (fold / "config.json").write_text(json.dumps(contract))
        (fold / "metrics.json").write_text(
            json.dumps(
                {
                    "validation_users": users,
                    "top1_accuracy": 0.9,
                    "macro_accuracy": 0.88,
                }
            )
        )
        validation_checkpoints.append(fold)
    output = tmp_path / "submission.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cf-sensor-fusion-submit",
            "--checkpoint",
            str(checkpoint),
            "--data-root",
            str(data_root),
            "--test-csv",
            str(test_csv),
            "--output",
            str(output),
            "--validation-checkpoints",
            *(str(path) for path in validation_checkpoints),
            "--batch-size",
            "2",
            "--workers",
            "0",
            "--device",
            "cpu",
        ],
    )
    submission.main()
    report = validate_submission(output, test_csv)
    assert report["rows"] == 2
    logits = np.load(output.with_suffix(".logits.npz"))
    assert logits["logits"].shape == (2, 40)
    assert logits["paths"].tolist() == [
        "small_model_track_test/SM_test_0001/",
        "small_model_track_test/SM_test_0002/",
    ]


def test_cross_subject_gate_rejects_overlap_and_weak_folds(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    contract = {
        "model": _tiny_config().to_dict(),
        "preprocessing": _preprocessing().to_dict(),
    }
    (candidate / "config.json").write_text(json.dumps(contract))

    def fold(name: str, users: tuple[int, ...], top1: float) -> Path:
        path = tmp_path / name
        path.mkdir()
        (path / "config.json").write_text(json.dumps(contract))
        (path / "metrics.json").write_text(
            json.dumps(
                {
                    "validation_users": users,
                    "top1_accuracy": top1,
                    "macro_accuracy": top1 - 0.02,
                }
            )
        )
        return path

    first = fold("first", (1, 16), 0.9)
    overlap = fold("overlap", (1, 17), 0.9)
    weak = fold("weak", (2, 18), 0.7)

    with pytest.raises(ValueError, match="overlap"):
        validate_cross_subject_quality(
            candidate,
            (first, overlap),
            minimum_folds=2,
        )
    with pytest.raises(ValueError, match="below"):
        validate_cross_subject_quality(
            candidate,
            (first, weak),
            minimum_folds=2,
        )
