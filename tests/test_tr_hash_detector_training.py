import json

import torch
from PIL import Image

from complexity.generative.detection import (
    CocoDetectionDataset,
    SyntheticShapesDataset,
    TRHashDetectorConfig,
    TRHashObjectDetector,
    YoloDetectionDataset,
    class_aware_nms,
    collate_detection,
    complete_iou_loss,
    sigmoid_focal_loss,
)


def test_synthetic_dataset_yields_valid_targets():
    dataset = SyntheticShapesDataset(length=4, image_size=64, seed=0)
    pixel_values, targets = dataset[0]
    assert pixel_values.shape == (3, 64, 64)
    assert targets.ndim == 2 and targets.shape[1] == 5
    assert targets.shape[0] >= 1
    cx, cy, w, h, class_id = targets.unbind(dim=-1)
    assert torch.all((cx >= 0) & (cx <= 1))
    assert torch.all((cy >= 0) & (cy <= 1))
    assert torch.all((w > 0) & (w <= 1))
    assert torch.all((h > 0) & (h <= 1))
    assert torch.all((class_id >= 0) & (class_id < 3))


def test_synthetic_dataset_deterministic_across_instances():
    first = SyntheticShapesDataset(length=4, image_size=64, seed=7)
    second = SyntheticShapesDataset(length=4, image_size=64, seed=7)
    pixels_a, targets_a = first[2]
    pixels_b, targets_b = second[2]
    assert torch.equal(pixels_a, pixels_b)
    assert torch.equal(targets_a, targets_b)


def test_synthetic_dataset_varies_across_indices():
    dataset = SyntheticShapesDataset(length=4, image_size=64, seed=0)
    _, targets_0 = dataset[0]
    _, targets_1 = dataset[1]
    assert not torch.equal(targets_0, targets_1) or targets_0.shape != targets_1.shape


def test_synthetic_dataset_can_resample_each_epoch_deterministically():
    first = SyntheticShapesDataset(
        length=4, image_size=64, seed=7, resample_each_epoch=True
    )
    second = SyntheticShapesDataset(
        length=4, image_size=64, seed=7, resample_each_epoch=True
    )
    pixels_epoch_0, targets_epoch_0 = first[2]
    first.set_epoch(1)
    second.set_epoch(1)
    pixels_epoch_1, targets_epoch_1 = first[2]
    expected_pixels, expected_targets = second[2]

    assert torch.equal(pixels_epoch_1, expected_pixels)
    assert torch.equal(targets_epoch_1, expected_targets)
    assert not torch.equal(pixels_epoch_0, pixels_epoch_1) or not torch.equal(
        targets_epoch_0, targets_epoch_1
    )


def test_class_aware_nms_preserves_overlapping_different_classes():
    boxes = torch.tensor([[0.1, 0.1, 0.9, 0.9], [0.1, 0.1, 0.9, 0.9]])
    scores = torch.tensor([0.9, 0.8])
    labels = torch.tensor([0, 1])
    kept = class_aware_nms(boxes, scores, labels, iou_threshold=0.5)
    assert kept.tolist() == [0, 1]


def test_complete_iou_loss_is_zero_for_identical_boxes():
    boxes = torch.tensor([[0.5, 0.5, 0.25, 0.4]])
    assert torch.allclose(complete_iou_loss(boxes, boxes), torch.zeros(1), atol=1e-6)


def test_focal_loss_downweights_easy_examples():
    targets = torch.tensor([1.0, 0.0])
    easy = sigmoid_focal_loss(
        torch.tensor([8.0, -8.0]), targets, alpha=0.75, gamma=2.0
    )
    hard = sigmoid_focal_loss(
        torch.tensor([0.0, 0.0]), targets, alpha=0.75, gamma=2.0
    )
    assert easy < hard


def test_collate_detection_batches_variable_length_targets():
    dataset = SyntheticShapesDataset(length=3, image_size=64, seed=1)
    batch = [dataset[index] for index in range(3)]
    pixel_values, targets = collate_detection(batch)
    assert pixel_values.shape == (3, 3, 64, 64)
    assert len(targets) == 3
    for target, (_, expected) in zip(targets, batch):
        assert torch.equal(target, expected)


def test_coco_dataset_loads_from_json_and_normalizes_boxes(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (100, 50), color=(10, 20, 30)).save(images_dir / "a.png")

    annotations = {
        "images": [{"id": 1, "file_name": "a.png"}],
        "annotations": [
            {"image_id": 1, "bbox": [10.0, 5.0, 20.0, 10.0], "category_id": 5},
            {"image_id": 1, "bbox": [50.0, 25.0, 10.0, 10.0], "category_id": 9},
        ],
    }
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(json.dumps(annotations))

    dataset = CocoDetectionDataset(annotations_path, images_dir, image_size=32)
    assert len(dataset) == 1
    assert dataset.num_classes == 2

    pixel_values, targets = dataset[0]
    assert pixel_values.shape == (3, 32, 32)
    assert targets.shape == (2, 5)

    cx, cy, w, h, class_id = targets[0].tolist()
    assert abs(cx - 20.0 / 100.0) < 1e-5
    assert abs(cy - 0.35) < 1e-5
    assert abs(w - 20.0 / 100.0) < 1e-5
    assert abs(h - 0.1) < 1e-5
    assert class_id == 0.0
    assert targets[1, 4].item() == 1.0


def test_yolo_dataset_loads_labels_letterboxes_and_augments(tmp_path):
    images_dir = tmp_path / "images" / "nested"
    labels_dir = tmp_path / "labels" / "nested"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    Image.new("RGB", (100, 50), color=(10, 20, 30)).save(images_dir / "a.png")
    (labels_dir / "a.txt").write_text("2 0.2 0.2 0.2 0.2\n")

    dataset = YoloDetectionDataset(
        tmp_path / "images",
        tmp_path / "labels",
        image_size=32,
        augment=True,
        seed=4,
    )
    pixels, targets = dataset[0]
    assert dataset.num_classes == 3
    assert pixels.shape == (3, 32, 32)
    assert targets.shape == (1, 5)
    assert torch.all((targets[:, :4] >= 0.0) & (targets[:, :4] <= 1.0))


def test_coco_dataset_handles_images_with_no_annotations(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (40, 40), color=(1, 2, 3)).save(images_dir / "b.png")
    annotations = {
        "images": [{"id": 1, "file_name": "b.png"}],
        "annotations": [],
    }
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(json.dumps(annotations))

    dataset = CocoDetectionDataset(annotations_path, images_dir, image_size=32)
    _, targets = dataset[0]
    assert targets.shape == (0, 5)


def test_training_loop_reduces_loss_on_synthetic_shapes():
    torch.manual_seed(0)
    config = TRHashDetectorConfig(
        image_size=64,
        patch_size=16,
        vision_hidden_size=64,
        vision_layers=2,
        vision_heads=4,
        vision_num_experts=4,
        vision_top_k=2,
        vision_expert_width=32,
        num_classes=3,
    )
    model = TRHashObjectDetector(config)
    dataset = SyntheticShapesDataset(length=8, image_size=64, seed=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

    losses = []
    for step in range(40):
        pixel_values, targets = dataset[step % len(dataset)]
        raw = model(pixel_values.unsqueeze(0))
        result = model.compute_loss(raw, [targets])
        optimizer.zero_grad(set_to_none=True)
        result["loss"].backward()
        optimizer.step()
        losses.append(float(result["loss"].detach()))

    early = sum(losses[:5]) / 5
    late = sum(losses[-5:]) / 5
    assert late < early
