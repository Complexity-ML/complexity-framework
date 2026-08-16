import json

import numpy as np
import pytest
import torch
from PIL import Image

from complexity.generative.detection.albumentations_backend import (
    _sanitize_yolo_targets,
    apply_albumentations,
    build_transform,
)
from complexity.generative.detection.coco_evaluation import (
    detections_to_coco,
    evaluate_coco_predictions,
)
from complexity.generative.detection.data import CocoDetectionDataset, collate_detection
from complexity.generative.detection.image_io import load_rgb_image
from complexity.generative.detection.training import validation_selection_score


def _coco_fixture(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (100, 50), color=(20, 40, 60)).save(images / "sample.png")
    annotations = {
        "info": {},
        "licenses": [],
        "images": [{"id": 7, "file_name": "sample.png", "width": 100, "height": 50}],
        "categories": [{"id": 5, "name": "object", "supercategory": "object"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 7,
                "category_id": 5,
                "bbox": [10.0, 5.0, 20.0, 10.0],
                "area": 200.0,
                "iscrowd": 0,
            }
        ],
    }
    path = tmp_path / "instances.json"
    path.write_text(json.dumps(annotations))
    return path, images


def test_coco_metadata_and_prediction_restoration(tmp_path):
    annotations, images = _coco_fixture(tmp_path)
    dataset = CocoDetectionDataset(
        annotations,
        images,
        image_size=640,
        return_metadata=True,
    )
    pixels, targets, metadata = dataset[0]
    batch = collate_detection([(pixels, targets, metadata)])

    assert batch[0].shape == (1, 3, 640, 640)
    assert batch[2][0]["image_id"] == 7
    box = targets[0, :4]
    cx, cy, width, height = box.tolist()
    xyxy = torch.tensor([[cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2]])
    predictions = detections_to_coco(
        [{"boxes": xyxy, "scores": torch.tensor([0.9]), "labels": torch.tensor([0])}],
        [metadata],
        dataset.class_to_category,
    )

    assert predictions[0]["image_id"] == 7
    assert predictions[0]["category_id"] == 5
    assert predictions[0]["bbox"] == pytest.approx([10.0, 5.0, 20.0, 10.0], abs=1e-4)


def test_official_coco_evaluators_are_numerically_equivalent(tmp_path):
    pytest.importorskip("pycocotools")
    pytest.importorskip("faster_coco_eval")
    annotations, _ = _coco_fixture(tmp_path)
    predictions = [
        {
            "image_id": 7,
            "category_id": 5,
            "bbox": [10.0, 5.0, 20.0, 10.0],
            "score": 0.99,
        }
    ]
    reference = evaluate_coco_predictions(
        annotations, predictions, [7], backend="pycocotools"
    )
    accelerated = evaluate_coco_predictions(
        annotations, predictions, [7], backend="faster"
    )

    for key in ("map50", "map50_95", "ap_small", "ap_medium", "ap_large", "ar_100"):
        assert accelerated[key] == pytest.approx(reference[key], abs=1e-10)
    for key in ("precision", "recall", "f1", "best_f1"):
        assert accelerated[key] == pytest.approx(1.0)
        assert accelerated[key] == pytest.approx(reference[key], abs=1e-10)


@pytest.mark.parametrize("backend", ("pycocotools", "faster"))
def test_official_coco_evaluator_handles_no_predictions(tmp_path, backend):
    pytest.importorskip("pycocotools")
    if backend == "faster":
        pytest.importorskip("faster_coco_eval")
    annotations, _ = _coco_fixture(tmp_path)

    metrics = evaluate_coco_predictions(annotations, [], [7], backend=backend)

    assert metrics["official_coco"] is True
    assert metrics["map50_95"] == 0.0
    assert metrics["ar_100"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["best_f1"] == 0.0


def test_checkpoint_selection_prefers_official_coco_map50_95():
    name, score = validation_selection_score(
        {
            "official_coco": True,
            "map50": 0.99,
            "coco_map50_95": 0.25,
        }
    )
    assert name == "coco_map50_95"
    assert score == 0.25
    assert validation_selection_score({"map50": 0.4}) == ("map50", 0.4)


def test_albumentations_is_seeded_and_keeps_valid_boxes():
    pytest.importorskip("albumentations")
    image = Image.new("RGB", (96, 64), color=(40, 90, 130))
    targets = torch.tensor([[0.5, 0.5, 0.4, 0.4, 2.0]])
    transform = build_transform("strong")

    first_image, first_targets = apply_albumentations(transform, image, targets, seed=123)
    second_image, second_targets = apply_albumentations(transform, image, targets, seed=123)

    np.testing.assert_array_equal(np.asarray(first_image), np.asarray(second_image))
    torch.testing.assert_close(first_targets, second_targets)
    assert first_targets.shape[1] == 5
    assert torch.all((first_targets[:, :4] >= 0.0) & (first_targets[:, :4] <= 1.0))


def test_albumentations_sanitizes_degenerate_boxes_before_validation():
    targets = torch.tensor(
        [
            [0.01, 0.230125, 0.004, 0.0, 0.0],
            [1.05, 0.50, 0.20, 0.20, 1.0],
            [0.50, 0.50, 0.40, 0.40, 2.0],
            [float("nan"), 0.50, 0.20, 0.20, 3.0],
        ]
    )

    cleaned = _sanitize_yolo_targets(targets)

    assert cleaned.shape == (2, 5)
    assert cleaned[:, 4].tolist() == [1.0, 2.0]
    assert torch.all(cleaned[:, 2:4] > 0.0)
    assert torch.all((cleaned[:, :4] >= 0.0) & (cleaned[:, :4] <= 1.0))


def test_opencv_and_pillow_decoders_preserve_rgb_geometry(tmp_path):
    pytest.importorskip("cv2")
    path = tmp_path / "image.png"
    pixels = np.zeros((12, 17, 3), dtype=np.uint8)
    pixels[..., 0] = 17
    pixels[..., 1] = 83
    pixels[..., 2] = 211
    Image.fromarray(pixels).save(path)

    pillow = load_rgb_image(path, "pillow")
    opencv = load_rgb_image(path, "opencv")

    assert pillow.size == opencv.size == (17, 12)
    np.testing.assert_array_equal(np.asarray(pillow), np.asarray(opencv))
