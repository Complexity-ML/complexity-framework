import json

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from complexity.generative.detection import (
    CocoDetectionDataset,
    SyntheticShapesDataset,
    TRHashDetectorConfig,
    TRHashObjectDetector,
    YoloDetectionDataset,
    class_aware_nms,
    collate_detection,
    complete_iou_loss,
    quality_focal_loss,
)
from complexity.generative.detection.checkpointing import (
    load_training_state,
    save_training_state,
)
from complexity.generative.detection.training import (
    _average_precision_from_matches,
    _match_image_detections,
    load_pretrained_detector,
    load_pretrained_tower,
)


def _checkpoint_test_optimizer():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.9**step)
    inputs = torch.tensor([[1.0, 2.0]])
    loss = model(inputs).square().sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return model, optimizer, scheduler


def test_exact_training_state_roundtrip_restores_cursor_optimizer_scheduler_and_rng(tmp_path):
    _, optimizer, scheduler = _checkpoint_test_optimizer()
    checkpoint = tmp_path / "step_000001"
    checkpoint.mkdir()
    options = {"optimizer": "sgd", "batch_size": 2}
    torch.manual_seed(1234)
    save_training_state(
        checkpoint,
        optimizer,
        scheduler,
        epoch=2,
        batch_in_epoch=3,
        step=11,
        best_map50=0.25,
        running_losses={"loss": 1.5},
        running_loss_steps=1,
        total_epochs=5,
        steps_per_epoch=4,
        training_options=options,
    )
    expected_random = torch.rand(4)

    _, restored_optimizer, restored_scheduler = _checkpoint_test_optimizer()
    state = load_training_state(
        checkpoint,
        restored_optimizer,
        restored_scheduler,
        total_epochs=5,
        steps_per_epoch=4,
        training_options=options,
    )

    assert (state["epoch"], state["batch_in_epoch"], state["step"]) == (2, 3, 11)
    assert restored_scheduler.last_epoch == scheduler.last_epoch
    original_momentum = next(iter(optimizer.state.values()))["momentum_buffer"]
    restored_momentum = next(iter(restored_optimizer.state.values()))["momentum_buffer"]
    assert torch.equal(restored_momentum, original_momentum)
    assert torch.equal(torch.rand(4), expected_random)


def test_exact_resume_rejects_weights_only_checkpoint(tmp_path):
    checkpoint = tmp_path / "old_checkpoint"
    checkpoint.mkdir()
    _, optimizer, scheduler = _checkpoint_test_optimizer()
    with pytest.raises(ValueError, match="weights-only"):
        load_training_state(
            checkpoint,
            optimizer,
            scheduler,
            total_epochs=5,
            steps_per_epoch=4,
            training_options={},
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


def test_class_aware_nms_caps_detections_after_score_sorting():
    boxes = torch.tensor(
        [[0.0, 0.0, 0.1, 0.1], [0.2, 0.2, 0.3, 0.3], [0.4, 0.4, 0.5, 0.5]]
    )
    scores = torch.tensor([0.7, 0.9, 0.8])
    labels = torch.tensor([0, 0, 0])
    kept = class_aware_nms(
        boxes, scores, labels, iou_threshold=0.5, max_detections=2
    )
    assert kept.tolist() == [1, 2]


def test_vectorized_evaluation_matching_preserves_score_greedy_assignment():
    boxes = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.05, 0.05, 1.0, 1.0],
            [2.0, 2.0, 3.0, 3.0],
        ]
    )
    scores = torch.tensor([0.9, 0.8, 0.7])
    labels = torch.tensor([0, 0, 0])
    target_boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]])
    target_labels = torch.tensor([0, 0])

    matched = _match_image_detections(
        boxes, scores, labels, target_boxes, target_labels, num_classes=1
    )
    class_scores, true_positives = matched[0]
    assert torch.equal(class_scores, scores)
    assert true_positives.tolist() == [1.0, 0.0, 1.0]
    average_precision = _average_precision_from_matches(
        class_scores, true_positives, total_ground_truth=2
    )
    assert abs(average_precision - 5.0 / 6.0) < 1e-6


def test_complete_iou_loss_is_zero_for_identical_boxes():
    boxes = torch.tensor([[0.5, 0.5, 0.25, 0.4]])
    assert torch.allclose(complete_iou_loss(boxes, boxes), torch.zeros(1), atol=1e-6)


def test_quality_focal_loss_downweights_easy_examples():
    targets = torch.tensor([[1.0, 0.0]])
    easy = quality_focal_loss(torch.tensor([[8.0, -8.0]]), targets)
    hard = quality_focal_loss(torch.tensor([[0.0, 0.0]]), targets)
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
        augmentation="strong",
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
        dynamic_assignment=False,
        stal_enabled=False,
        progressive_loss_enabled=False,
        vision_precision="fp32",
    )
    model = TRHashObjectDetector(config)
    dataset = SyntheticShapesDataset(length=1, image_size=64, seed=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, nesterov=True)

    losses = []
    for step in range(80):
        pixel_values, targets = dataset[0]
        raw = model(pixel_values.unsqueeze(0))
        result = model.compute_loss(raw, [targets])
        optimizer.zero_grad(set_to_none=True)
        result["loss"].backward()
        optimizer.step()
        losses.append(float(result["loss"].detach()))

    early = sum(losses[:5]) / 5
    late = sum(losses[-5:]) / 5
    assert late < early


def test_pretrained_tower_resizes_positions_and_regenerates_routes(tmp_path):
    source = TRHashObjectDetector(
        TRHashDetectorConfig(
            image_size=32,
            patch_size=8,
            vision_hidden_size=32,
            vision_layers=1,
            vision_heads=4,
            vision_num_experts=4,
            vision_top_k=2,
            vision_expert_width=16,
            num_classes=3,
        )
    )
    checkpoint = tmp_path / "backbone"
    checkpoint.mkdir()
    save_file(
        {name: value.detach().contiguous() for name, value in source.tower.state_dict().items()},
        str(checkpoint / "tower.safetensors"),
    )
    target = TRHashObjectDetector(
        TRHashDetectorConfig(
            image_size=64,
            patch_size=8,
            vision_hidden_size=32,
            vision_layers=1,
            vision_heads=4,
            vision_num_experts=4,
            vision_top_k=2,
            vision_expert_width=16,
            num_classes=3,
        )
    )

    load_pretrained_tower(target, checkpoint)

    assert target.tower.position_embedding.shape == (1, 64, 32)
    assert target.tower.route_ids.shape == (64,)
    assert target.tower.blocks[0].mlp.route_table.shape == (2, 64)
    assert torch.equal(
        target.tower.blocks[0].mlp.expert_gate,
        source.tower.blocks[0].mlp.expert_gate,
    )


def test_detector_transfer_preserves_regression_and_mapped_classes(tmp_path):
    common = dict(
        image_size=32,
        patch_size=8,
        vision_hidden_size=32,
        vision_layers=1,
        vision_heads=4,
        vision_num_experts=4,
        vision_top_k=2,
        vision_expert_width=16,
    )
    source_config = TRHashDetectorConfig(**common, num_classes=3)
    source = TRHashObjectDetector(source_config)
    with torch.no_grad():
        source.tower.patch_embed.weight.fill_(0.125)
        source.fpn_downsamples[0][0].weight.fill_(0.25)
        source.head.regression_heads[0][1].weight.fill_(0.375)
        for head in source.head.classification_heads:
            final = head[-1]
            for output_index in range(final.out_features):
                final.weight[output_index].fill_(float(output_index))
                final.bias[output_index] = float(output_index)

    checkpoint = tmp_path / "source_detector"
    checkpoint.mkdir()
    save_file(
        {
            name: value.detach().contiguous()
            for name, value in source.state_dict().items()
        },
        str(checkpoint / "model.safetensors"),
    )
    (checkpoint / "config.json").write_text(json.dumps(source_config.to_dict()))

    target = TRHashObjectDetector(TRHashDetectorConfig(**common, num_classes=2))
    unmapped_weight = target.head.classification_heads[0][-1].weight[1].detach().clone()
    unmapped_bias = target.head.classification_heads[0][-1].bias[1].detach().clone()
    load_pretrained_detector(target, checkpoint, class_mapping={0: 2})

    assert torch.all(target.tower.patch_embed.weight == 0.125)
    assert torch.all(target.fpn_downsamples[0][0].weight == 0.25)
    assert torch.all(target.head.regression_heads[0][1].weight == 0.375)
    for level, head in enumerate(target.head.classification_heads):
        final = head[-1]
        source_final = source.head.classification_heads[level][-1]
        assert torch.equal(final.weight[0], source_final.weight[2])
        assert torch.equal(final.bias[0], source_final.bias[2])
    assert torch.equal(target.head.classification_heads[0][-1].weight[1], unmapped_weight)
    assert torch.equal(target.head.classification_heads[0][-1].bias[1], unmapped_bias)
