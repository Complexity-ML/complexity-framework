import json
import math

import pytest
import torch

from complexity.generative.detection import TRHashDetectorConfig, TRHashObjectDetector
from complexity.generative.vision_tasks import (
    SUPPORTED_VISION_TASKS,
    TRHashDepthEstimator,
    TRHashImageClassifier,
    TRHashInstanceSegmenter,
    TRHashOBBDetector,
    TRHashPoseEstimator,
    TRHashSemanticSegmenter,
    create_vision_model,
    load_vision_task_checkpoint,
    save_vision_task_checkpoint,
)


def _config(**overrides) -> TRHashDetectorConfig:
    values = dict(
        image_size=32,
        patch_size=8,
        vision_hidden_size=32,
        vision_layers=1,
        vision_heads=4,
        vision_num_experts=4,
        vision_top_k=2,
        vision_expert_width=16,
        vision_precision="fp32",
        num_classes=4,
    )
    values.update(overrides)
    return TRHashDetectorConfig(**values)


def test_factory_exposes_all_seven_task_families():
    config = _config()
    expected = {
        "detection": TRHashObjectDetector,
        "instance_segmentation": TRHashInstanceSegmenter,
        "semantic_segmentation": TRHashSemanticSegmenter,
        "depth": TRHashDepthEstimator,
        "classification": TRHashImageClassifier,
        "pose": TRHashPoseEstimator,
        "obb": TRHashOBBDetector,
    }

    assert set(SUPPORTED_VISION_TASKS) == set(expected)
    for task, model_type in expected.items():
        assert isinstance(create_vision_model(task, config), model_type)


def test_classification_forward_and_loss():
    model = TRHashImageClassifier(_config(), num_classes=6)
    output = model(torch.randn(2, 3, 32, 32), torch.tensor([1, 4]))

    assert output["logits"].shape == (2, 6)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert model.head.weight.grad is not None


def test_semantic_segmentation_forward_and_loss():
    model = TRHashSemanticSegmenter(_config(), num_classes=5)
    labels = torch.randint(0, 5, (2, 32, 32))
    output = model(torch.randn(2, 3, 32, 32), labels)

    assert output["logits"].shape == (2, 5, 32, 32)
    output["loss"].backward()
    assert model.decoder.layers[-1].weight.grad is not None


def test_depth_is_positive_and_scale_invariant_loss_is_finite():
    model = TRHashDepthEstimator(_config(), max_depth=80.0)
    target = torch.rand(2, 1, 32, 32) * 20.0 + 0.1
    output = model(torch.randn(2, 3, 32, 32), target)

    assert output["depth"].shape == target.shape
    assert torch.all(output["depth"] > 0)
    assert torch.all(output["depth"] <= 80.0)
    assert torch.isfinite(output["loss"])


def test_pose_heatmaps_and_visibility_weighted_loss():
    model = TRHashPoseEstimator(_config(), num_keypoints=7)
    targets = torch.rand(2, 7, 32, 32)
    visibility = torch.tensor([[1, 1, 1, 0, 0, 1, 1], [1, 0, 1, 1, 1, 1, 0]])
    output = model(torch.randn(2, 3, 32, 32), targets, visibility)

    assert output["heatmaps"].shape == (2, 7, 32, 32)
    assert torch.all((output["heatmaps"] >= 0) & (output["heatmaps"] <= 1))
    assert torch.isfinite(output["loss"])


def test_instance_segmentation_uses_one_backbone_pass_and_trains_masks():
    model = TRHashInstanceSegmenter(_config(p2_head=True), num_prototypes=8)
    tower_calls = 0

    def count_tower_calls(_module, _inputs, _output):
        nonlocal tower_calls
        tower_calls += 1

    handle = model.tower.register_forward_hook(count_tower_calls)
    outputs = model.forward_instance(torch.randn(1, 3, 32, 32))
    handle.remove()
    targets = [torch.tensor([[0.5, 0.5, 0.3, 0.4, 2.0]])]
    masks = torch.zeros(1, 32, 32)
    masks[:, 8:24, 10:22] = 1
    losses = model.compute_instance_loss(outputs, targets, [masks])

    assert tower_calls == 1
    assert outputs["mask_coefficients"].shape == (1, model.config.num_cells, 8)
    assert outputs["prototypes"].shape == (1, 8, 8, 8)
    assert torch.isfinite(losses["mask_loss"])
    losses["loss"].backward()
    assert model.prototype_head[-1].weight.grad is not None

    predictions = model.predict_instance(
        torch.randn(1, 3, 32, 32),
        confidence_threshold=0.0,
        max_detections=3,
    )
    assert predictions[0]["masks"].shape == (3, 32, 32)
    assert predictions[0]["masks"].dtype == torch.bool


def test_obb_periodic_angle_loss_and_nms_prediction(monkeypatch):
    model = TRHashOBBDetector(_config())
    outputs = model.forward_obb(torch.randn(1, 3, 32, 32))
    targets = [torch.tensor([[0.5, 0.5, 0.3, 0.4, math.pi / 4, 2.0]])]
    losses = model.compute_obb_loss(outputs, targets)

    assert outputs["angles"].shape == (1, model.config.num_cells)
    assert torch.all(outputs["angles"] >= -math.pi)
    assert torch.all(outputs["angles"] <= math.pi)
    assert torch.isfinite(losses["angle_loss"])

    calls = []

    def record_nms(boxes, scores, labels, iou_threshold, max_detections):
        calls.append(len(boxes))
        return torch.argsort(scores, descending=True)[:max_detections]

    monkeypatch.setattr("complexity.generative.vision_tasks.model.class_aware_nms", record_nms)
    predictions = model.predict_obb(
        torch.randn(1, 3, 32, 32), confidence_threshold=0.0, max_detections=5
    )
    assert predictions[0]["boxes"].shape == (5, 4)
    assert predictions[0]["angles"].shape == (5,)
    assert calls


def test_unknown_task_is_rejected():
    with pytest.raises(ValueError, match="unsupported vision task"):
        create_vision_model("optical_flow", _config())


def test_classification_checkpoint_round_trip(tmp_path):
    model = TRHashImageClassifier(_config(), num_classes=6).eval()
    pixels = torch.randn(2, 3, 32, 32)
    expected = model(pixels)["logits"]

    checkpoint = save_vision_task_checkpoint(
        model,
        tmp_path / "classifier",
        task="classification",
        class_names=("a", "b", "c", "d", "e", "f"),
    )
    restored = load_vision_task_checkpoint(checkpoint)

    torch.testing.assert_close(restored(pixels)["logits"], expected)
    assert restored.vision_task == "classification"
    assert json.loads((checkpoint / "vision_task.json").read_text())["format_version"] == 5


def test_semantic_checkpoint_round_trip(tmp_path):
    model = TRHashSemanticSegmenter(_config(), num_classes=3).eval()
    pixels = torch.randn(1, 3, 32, 32)
    expected = model(pixels)["logits"]

    checkpoint = save_vision_task_checkpoint(
        model,
        tmp_path / "segmenter",
        task="semantic_segmentation",
        class_names=("road", "person", "sky"),
    )
    restored = load_vision_task_checkpoint(checkpoint)

    torch.testing.assert_close(restored(pixels)["logits"], expected)
    assert restored.vision_task == "semantic_segmentation"


def test_depth_checkpoint_round_trip(tmp_path):
    model = TRHashDepthEstimator(_config(), max_depth=80.0).eval()
    pixels = torch.randn(1, 3, 32, 32)
    expected = model(pixels)["depth"]

    checkpoint = save_vision_task_checkpoint(
        model,
        tmp_path / "depth",
        task="depth",
    )
    restored = load_vision_task_checkpoint(checkpoint)

    torch.testing.assert_close(restored(pixels)["depth"], expected)
    assert restored.vision_task == "depth"
    assert json.loads((checkpoint / "class_names.json").read_text()) == []


def test_pose_checkpoint_round_trip(tmp_path):
    model = TRHashPoseEstimator(_config(), num_keypoints=5).eval()
    pixels = torch.randn(1, 3, 32, 32)
    expected = model(pixels)["heatmaps"]

    checkpoint = save_vision_task_checkpoint(
        model,
        tmp_path / "pose",
        task="pose",
        class_names=("nose", "left_eye", "right_eye", "left_ear", "right_ear"),
    )
    restored = load_vision_task_checkpoint(checkpoint)

    torch.testing.assert_close(restored(pixels)["heatmaps"], expected)
    assert restored.vision_task == "pose"


def test_instance_segmentation_checkpoint_round_trip(tmp_path):
    model = TRHashInstanceSegmenter(_config(), num_prototypes=6).eval()
    pixels = torch.randn(1, 3, 32, 32)
    expected = model.forward_instance(pixels)

    checkpoint = save_vision_task_checkpoint(
        model,
        tmp_path / "instance-segmenter",
        task="instance_segmentation",
        class_names=("a", "b", "c", "d"),
    )
    restored = load_vision_task_checkpoint(checkpoint)
    actual = restored.forward_instance(pixels)

    for name in ("raw", "mask_coefficients", "prototypes"):
        torch.testing.assert_close(actual[name], expected[name])
    assert restored.vision_task == "instance_segmentation"


def test_obb_checkpoint_round_trip(tmp_path):
    model = TRHashOBBDetector(_config()).eval()
    pixels = torch.randn(1, 3, 32, 32)
    expected = model.forward_obb(pixels)

    checkpoint = save_vision_task_checkpoint(
        model,
        tmp_path / "obb",
        task="obb",
        class_names=("a", "b", "c", "d"),
    )
    restored = load_vision_task_checkpoint(checkpoint)
    actual = restored.forward_obb(pixels)

    for name in ("raw", "angle_vectors", "angles"):
        torch.testing.assert_close(actual[name], expected[name])
    assert restored.vision_task == "obb"
