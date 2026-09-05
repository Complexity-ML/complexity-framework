from pathlib import Path

import pytest

from scripts.check_onnx_quantized_artifacts import (
    assert_disjoint_image_ids,
    image_id_manifest_sha256,
    load_calibration_manifest,
    load_quantization_thresholds,
)

HASH = "a" * 64
IMAGE_IDS = [9, 25, 30]
IMAGE_IDS_HASH = image_id_manifest_sha256(IMAGE_IDS)


def test_threshold_config_requires_explicit_release_policy(tmp_path: Path) -> None:
    path = tmp_path / "thresholds.json"
    path.write_text(
        '{"schema_version": 1, "precisions": {"fp16": {}, "int8": {}}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="release_policy"):
        load_quantization_thresholds(path)


def test_threshold_config_requires_fp16_and_int8_policies(tmp_path: Path) -> None:
    path = tmp_path / "thresholds.json"
    path.write_text(
        '{"schema_version": 1, "release_policy": {}, "precisions": {"fp16": {}}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fp16 and int8"):
        load_quantization_thresholds(path)


def test_calibration_manifest_pins_int8_settings(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(
        """
        {
          "schema_version": 1,
          "dataset": {
            "name": "coco-2017-train",
            "image_ids_sha256": "%s",
            "annotations_sha256": "%s",
            "disjoint_from": "coco-2017-val2017"
          },
          "image_ids": [9, 25, 30],
          "images": [
            "artifacts/COCO/images/train2017/000000000009.jpg",
            "artifacts/COCO/images/train2017/000000000025.jpg",
            "artifacts/COCO/images/train2017/000000000030.jpg"
          ],
          "quantization": {
            "calibration_method": "minmax",
            "per_channel": true,
            "symmetric_activations": false,
            "symmetric_weights": true,
            "activation_type": "quint8",
            "weight_type": "qint8",
            "batch_size": 8
          }
        }
        """
        % (IMAGE_IDS_HASH, HASH),
        encoding="utf-8",
    )

    manifest = load_calibration_manifest(path)

    assert manifest["quantization"]["calibration_method"] == "minmax"
    assert manifest["quantization"]["batch_size"] == 8


def test_calibration_manifest_rejects_missing_quantization_setting(
    tmp_path: Path,
) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(
        """
        {
          "schema_version": 1,
          "dataset": {
            "name": "coco-2017-train",
            "image_ids_sha256": "%s",
            "annotations_sha256": "%s",
            "disjoint_from": "coco-2017-val2017"
          },
          "image_ids": [9, 25, 30],
          "images": [
            "artifacts/COCO/images/train2017/000000000009.jpg",
            "artifacts/COCO/images/train2017/000000000025.jpg",
            "artifacts/COCO/images/train2017/000000000030.jpg"
          ],
          "quantization": {
            "calibration_method": "minmax",
            "per_channel": true,
            "symmetric_activations": false,
            "symmetric_weights": true,
            "activation_type": "quint8",
            "weight_type": "qint8"
          }
        }
        """
        % (IMAGE_IDS_HASH, HASH),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="batch_size"):
        load_calibration_manifest(path)


def test_calibration_manifest_rejects_placeholder_dataset_hash(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(
        """
        {
          "schema_version": 1,
          "dataset": {
            "name": "coco-2017-train",
            "image_ids_sha256": "replace-with-calibration-image-id-manifest-sha256",
            "annotations_sha256": "%s",
            "disjoint_from": "coco-2017-val2017"
          },
          "image_ids": [9, 25, 30],
          "images": [
            "artifacts/COCO/images/train2017/000000000009.jpg",
            "artifacts/COCO/images/train2017/000000000025.jpg",
            "artifacts/COCO/images/train2017/000000000030.jpg"
          ],
          "quantization": {
            "calibration_method": "minmax",
            "per_channel": true,
            "symmetric_activations": false,
            "symmetric_weights": true,
            "activation_type": "quint8",
            "weight_type": "qint8",
            "batch_size": 8
          }
        }
        """
        % HASH,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="image_ids_sha256"):
        load_calibration_manifest(path)


def test_calibration_manifest_requires_pinned_image_identity(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(
        """
        {
          "schema_version": 1,
          "dataset": {
            "name": "coco-2017-train",
            "image_ids_sha256": "%s",
            "annotations_sha256": "%s",
            "disjoint_from": "coco-2017-val2017"
          },
          "quantization": {
            "calibration_method": "minmax",
            "per_channel": true,
            "symmetric_activations": false,
            "symmetric_weights": true,
            "activation_type": "quint8",
            "weight_type": "qint8",
            "batch_size": 8
          }
        }
        """
        % (HASH, HASH),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="image_ids"):
        load_calibration_manifest(path)


def test_calibration_manifest_requires_calibration_image_paths(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(
        """
        {
          "schema_version": 1,
          "dataset": {
            "name": "coco-2017-train",
            "image_ids_sha256": "%s",
            "annotations_sha256": "%s",
            "disjoint_from": "coco-2017-val2017"
          },
          "image_ids": [9, 25, 30],
          "quantization": {
            "calibration_method": "minmax",
            "per_channel": true,
            "symmetric_activations": false,
            "symmetric_weights": true,
            "activation_type": "quint8",
            "weight_type": "qint8",
            "batch_size": 8
          }
        }
        """
        % (IMAGE_IDS_HASH, HASH),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="images"):
        load_calibration_manifest(path)


def test_calibration_and_eval_image_ids_must_be_disjoint() -> None:
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_image_ids({1, 2, 3}, {3, 4, 5})


def test_image_id_manifest_hash_is_order_stable() -> None:
    assert image_id_manifest_sha256([3, 1, 2]) == image_id_manifest_sha256([1, 2, 3])
