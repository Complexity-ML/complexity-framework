import json

import torch
from PIL import Image

from complexity.generative.detection import (
    CocoDetectionDataset,
    SyntheticShapesDataset,
    TRHashDetectorConfig,
    TRHashObjectDetector,
    collate_detection,
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
    assert abs(cy - 10.0 / 50.0) < 1e-5
    assert abs(w - 20.0 / 100.0) < 1e-5
    assert abs(h - 10.0 / 50.0) < 1e-5
    assert class_id == 0.0
    assert targets[1, 4].item() == 1.0


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
