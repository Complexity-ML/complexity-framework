import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from complexity.generative.sensor_fusion import (
    CUHKXPreprocessingConfig,
    CUHKXSmallTrackDataset,
    TRHashSensorFusionClassifier,
    TRHashSensorFusionConfig,
    build_cuhkx_records,
    collate_cuhkx,
    load_cuhkx_manifest,
    save_cuhkx_manifest,
    subject_disjoint_records,
)
from complexity.generative.sensor_fusion.preprocessing import (
    _visual_frame_paths,
    load_imu_sequence,
    load_radar_sequence,
    load_skeleton_sequence,
)

MODALITY_FOLDERS = {
    "depth": "Depth_Color",
    "ir": "IR",
    "thermal": "Thermal",
    "imu": "IMU",
    "radar": "Radar",
    "skeleton": "Skeleton",
}


def _trial(root: Path, modality: str, user: int, trial: str = "1-1-1") -> Path:
    path = (
        root
        / "HAR"
        / "data"
        / MODALITY_FOLDERS[modality]
        / "03_Walk_forward"
        / f"user{user}"
        / trial
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_visual(root: Path, modality: str, user: int) -> None:
    path = _trial(root, modality, user)
    mode = "L" if modality == "ir" else "RGB"
    for index in range(3):
        shape = (18, 24) if mode == "L" else (18, 24, 3)
        values = np.full(shape, 32 + index * 64, dtype=np.uint8)
        Image.fromarray(values, mode=mode).save(path / f"frame_{index:03d}.png")


def _write_imu(root: Path, user: int) -> None:
    path = _trial(root, "imu", user)
    header = ["timestamp", "device"] + [f"value_{index}" for index in range(9)]
    devices = ("WTLA", "WTRA", "WTC", "WTLL", "WTRL")
    with (path / "imu.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for step in range(5):
            for device_index, device in enumerate(devices):
                writer.writerow(
                    [step, f"{device}(device)"]
                    + [step + device_index + value / 10 for value in range(9)]
                )


def _write_radar(root: Path, user: int) -> None:
    path = _trial(root, "radar", user)
    with (path / "radar.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "frame", "object", "x", "y", "z", "v", "snr", "noise"])
        for frame in range(5):
            for point in range(2):
                writer.writerow(
                    [frame, frame, point, frame, point, 1.0, 0.1 * frame, 5 + point, 0.2]
                )


def _write_skeleton(root: Path, user: int) -> None:
    path = _trial(root, "skeleton", user)
    for frame in range(4):
        keypoints = [[joint + frame, joint * 0.5, 1.0] for joint in range(17)]
        (path / f"pose_{frame:03d}.json").write_text(
            json.dumps([{"keypoints": keypoints, "scores": [1.0] * 17}])
        )


def _write_complete_trial(root: Path, user: int) -> None:
    for modality in ("depth", "ir", "thermal"):
        _write_visual(root, modality, user)
    _write_imu(root, user)
    _write_radar(root, user)
    _write_skeleton(root, user)


def _preprocessing() -> CUHKXPreprocessingConfig:
    return CUHKXPreprocessingConfig(image_size=16, clip_frames=4, sensor_steps=8)


def test_index_merges_modalities_and_cross_subject_split(tmp_path):
    _write_complete_trial(tmp_path, user=1)
    _write_complete_trial(tmp_path, user=8)

    records = build_cuhkx_records(tmp_path)
    assert len(records) == 2
    assert set(records[0].paths) == set(MODALITY_FOLDERS)
    assert records[0].action_id == 3
    assert records[0].action_name == "Walk_forward"
    assert {record.user_id for record in subject_disjoint_records(records, split="train")} == {1}
    assert {
        record.user_id for record in subject_disjoint_records(records, split="validation")
    } == {8}

    manifest = tmp_path / "manifest.json"
    save_cuhkx_manifest(records, manifest)
    assert load_cuhkx_manifest(manifest) == records


def test_dataset_preprocessing_collation_and_model_forward(tmp_path):
    _write_complete_trial(tmp_path, user=1)
    dataset = CUHKXSmallTrackDataset(
        tmp_path,
        split="all",
        preprocessing=_preprocessing(),
    )
    sample = dataset[0]
    assert sample["inputs"]["depth"].shape == (3, 4, 16, 16)
    assert sample["inputs"]["ir"].shape == (1, 4, 16, 16)
    assert sample["inputs"]["thermal"].shape == (3, 4, 16, 16)
    assert sample["inputs"]["imu"].shape == (8, 45)
    assert sample["inputs"]["radar"].shape == (8, 16)
    assert sample["inputs"]["skeleton"].shape == (8, 17, 3)
    assert all(value.item() for value in sample["modality_mask"].values())
    assert all(torch.isfinite(value).all() for value in sample["inputs"].values())
    assert sample["label"].item() == 3
    assert sample["loss_weight"].item() == 1.0

    dataset.set_loss_weights(torch.tensor([2.5]))
    sample = dataset[0]
    batch = collate_cuhkx([sample, sample])
    assert batch["loss_weights"].tolist() == [2.5, 2.5]
    config = TRHashSensorFusionConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_experts=8,
        top_k=2,
        shared_width=32,
        expert_width=8,
        precision="fp32",
        visual_token_grid=(2, 2, 2),
        vision_image_size=32,
        vision_patch_size=4,
        vision_hidden_size=32,
        vision_layers=2,
        vision_heads=4,
        vision_num_experts=4,
        vision_top_k=2,
        vision_expert_width=8,
        vision_stage_depths=(1, 1),
        vision_window_size=2,
        sequence_tokens=4,
    )
    output = TRHashSensorFusionClassifier(config)(
        batch["inputs"],
        batch["labels"],
        modality_mask=batch["modality_mask"],
    )
    assert output["logits"].shape == (2, 40)
    assert torch.isfinite(output["loss"])


def test_training_data_cache_avoids_reindexing_and_reparsing_each_epoch(
    tmp_path,
    monkeypatch,
):
    import complexity.generative.sensor_fusion.data as data_module

    _write_complete_trial(tmp_path, user=1)
    _visual_frame_paths.cache_clear()
    imu_calls = 0
    original_imu = data_module.load_imu_sequence

    def counted_imu(path, config):
        nonlocal imu_calls
        imu_calls += 1
        return original_imu(path, config)

    monkeypatch.setattr(data_module, "load_imu_sequence", counted_imu)
    dataset = CUHKXSmallTrackDataset(
        tmp_path,
        split="train",
        preprocessing=_preprocessing(),
        training_augmentation=True,
    )

    first = dataset[(0, 0)]
    first_cache = _visual_frame_paths.cache_info()
    second = dataset[(0, 1)]
    second_cache = _visual_frame_paths.cache_info()

    assert imu_calls == 1
    assert second_cache.misses == first_cache.misses
    assert second_cache.hits >= first_cache.hits + 3
    assert first["inputs"]["imu"].data_ptr() == second["inputs"]["imu"].data_ptr()
    # Visual augmentation remains epoch-dependent even though directory scans
    # and deterministic sensor parsing are cached.
    assert not torch.equal(first["inputs"]["depth"], second["inputs"]["depth"])


def test_missing_modality_emits_zero_tensor_and_false_mask(tmp_path):
    _write_complete_trial(tmp_path, user=1)
    thermal = _trial(tmp_path, "thermal", 1)
    for path in thermal.iterdir():
        path.unlink()

    dataset = CUHKXSmallTrackDataset(
        tmp_path,
        split="all",
        preprocessing=_preprocessing(),
    )
    sample = dataset[0]
    assert not sample["modality_mask"]["thermal"]
    assert torch.count_nonzero(sample["inputs"]["thermal"]) == 0
    assert sample["modality_mask"]["depth"]


def test_corrupt_visual_frames_use_nearest_decodable_frame(tmp_path):
    _write_complete_trial(tmp_path, user=1)
    ir = _trial(tmp_path, "ir", 1)
    (ir / "frame_001.png").write_bytes(bytes(1024))

    sample = CUHKXSmallTrackDataset(
        tmp_path,
        split="all",
        preprocessing=_preprocessing(),
    )[0]
    assert sample["modality_mask"]["ir"]
    assert torch.isfinite(sample["inputs"]["ir"]).all()
    assert torch.count_nonzero(sample["inputs"]["ir"]) > 0


def test_all_corrupt_visual_frames_emit_zero_tensor_and_false_mask(tmp_path):
    _write_complete_trial(tmp_path, user=1)
    ir = _trial(tmp_path, "ir", 1)
    for path in ir.glob("*.png"):
        path.write_bytes(bytes(1024))

    sample = CUHKXSmallTrackDataset(
        tmp_path,
        split="all",
        preprocessing=_preprocessing(),
    )[0]
    assert not sample["modality_mask"]["ir"]
    assert torch.count_nonzero(sample["inputs"]["ir"]) == 0


def test_preprocessing_v2_sorts_imu_and_preserves_absolute_sensor_values(tmp_path):
    imu = _trial(tmp_path, "imu", 1)
    devices = ("WTLA", "WTRA", "WTC", "WTLL", "WTRL")
    with (imu / "imu.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "device", *[f"v{index}" for index in range(9)]])
        for step in reversed(range(4)):
            for device in devices:
                writer.writerow([f"2026-01-01 00:00:0{step}", device, *([step + 1] * 9)])

    config = CUHKXPreprocessingConfig(version=2, sensor_steps=4)
    values = load_imu_sequence(imu, config)
    assert values[0, 0] < values[-1, 0]
    assert values.abs().sum() > 0

    _write_radar(tmp_path, user=1)
    radar = _trial(tmp_path, "radar", 1)
    path = radar / "radar.csv"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:3]) + "\n")
    radar_values = load_radar_sequence(radar, config)
    assert radar_values.abs().sum() > 0


def test_preprocessing_v3_preserves_normalized_root_trajectory(tmp_path):
    skeleton = _trial(tmp_path, "skeleton", 1)
    for frame in range(4):
        keypoints = [
            [frame * 0.5 + joint * 0.1, joint * 0.2, 1.0]
            for joint in range(17)
        ]
        (skeleton / f"pose_{frame:03d}.json").write_text(
            json.dumps([{"keypoints": keypoints, "keypoint_scores": [1.0] * 17}])
        )

    legacy = load_skeleton_sequence(
        skeleton,
        CUHKXPreprocessingConfig(version=2, sensor_steps=4),
    )
    trajectory = load_skeleton_sequence(
        skeleton,
        CUHKXPreprocessingConfig(version=3, sensor_steps=4),
    )

    assert torch.count_nonzero(legacy[:, 0]) == 0
    assert torch.count_nonzero(trajectory[1:, 0, 0]) == 3
    assert torch.all(trajectory[1:, 0, 0] > trajectory[:-1, 0, 0])


def test_training_visual_sampling_is_epoch_deterministic(tmp_path):
    _write_complete_trial(tmp_path, user=1)
    dataset = CUHKXSmallTrackDataset(
        tmp_path,
        split="train",
        preprocessing=_preprocessing(),
        training_augmentation=True,
        seed=17,
    )
    first = dataset[(0, 3)]["inputs"]["depth"]
    repeated = dataset[(0, 3)]["inputs"]["depth"]
    assert torch.equal(first, repeated)
    alternatives = [dataset[(0, epoch)]["inputs"]["depth"] for epoch in range(4, 12)]
    assert any(not torch.equal(first, candidate) for candidate in alternatives)

    validation = CUHKXSmallTrackDataset(
        tmp_path,
        split="validation",
        preprocessing=_preprocessing(),
        validation_users=(1,),
    )
    assert torch.equal(
        validation[0]["inputs"]["depth"],
        validation[0]["inputs"]["depth"],
    )


def test_invalid_validation_people_are_rejected(tmp_path):
    _write_complete_trial(tmp_path, user=1)
    records = build_cuhkx_records(tmp_path)
    try:
        subject_disjoint_records(records, split="train", validation_users=(99,))
    except ValueError as error:
        assert "train users" in str(error)
    else:
        raise AssertionError("expected invalid validation users to fail")
