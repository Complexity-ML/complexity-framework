import torch

from complexity.generative.detection import TRHashDetectorConfig, TRHashObjectDetector
from complexity.generative.detection.model import box_iou, greedy_nms


def _tiny_config(**overrides) -> TRHashDetectorConfig:
    fields = dict(
        image_size=32,
        patch_size=8,
        vision_hidden_size=32,
        vision_layers=2,
        vision_heads=4,
        vision_num_experts=4,
        vision_top_k=2,
        vision_expert_width=16,
        num_classes=5,
    )
    fields.update(overrides)
    return TRHashDetectorConfig(**fields)


def test_box_iou_matches_known_values():
    a = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    b = torch.tensor([[0.0, 0.0, 1.0, 1.0], [10.0, 10.0, 11.0, 11.0]])
    iou = box_iou(a, b)
    assert torch.allclose(iou[0, 0], torch.tensor(1.0))
    assert torch.allclose(iou[0, 1], torch.tensor(0.0))


def test_greedy_nms_suppresses_overlapping_lower_score_boxes():
    boxes = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.05, 0.05, 1.05, 1.05],
            [5.0, 5.0, 6.0, 6.0],
        ]
    )
    scores = torch.tensor([0.9, 0.8, 0.7])
    keep = greedy_nms(boxes, scores, iou_threshold=0.5)
    assert keep.tolist() == [0, 2]


def test_backbone_uses_real_multi_expert_moe():
    config = _tiny_config()
    model = TRHashObjectDetector(config)
    for block in model.tower.blocks:
        assert torch.unique(block.mlp.route_table).numel() > 1


def test_forward_shape():
    config = _tiny_config()
    model = TRHashObjectDetector(config)
    pixels = torch.randn(2, 3, 32, 32)
    raw = model(pixels)
    assert raw.shape == (2, config.num_cells, config.prediction_width)


def test_default_head_uses_three_feature_scales():
    config = _tiny_config()
    assert config.grid_sizes == (4, 2, 1)
    assert config.num_cells == 21
    model = TRHashObjectDetector(config)
    assert len(model.head.regression_heads) == 3
    assert len(model.head.classification_heads) == 3
    assert len(model.fpn_downsamples) == 2


def test_optional_p2_head_adds_a_stride_four_prediction_grid():
    config = _tiny_config(p2_head=True)
    model = TRHashObjectDetector(config)
    raw = model(torch.randn(1, 3, 32, 32))

    assert config.grid_sizes == (8, 4, 2, 1)
    assert raw.shape == (1, 8**2 + 4**2 + 2**2 + 1, config.prediction_width)
    assert model.fpn_upsample is not None


def test_forward_predictions_exposes_stable_o2m_predictions():
    config = _tiny_config()
    model = TRHashObjectDetector(config)
    pixels = torch.randn(2, 3, 32, 32)

    one_to_many = model.forward_predictions(pixels)

    assert one_to_many.shape == (2, config.num_cells, config.prediction_width)


def test_default_224_configuration_uses_rounded_pyramid_grids():
    config = TRHashDetectorConfig()
    assert config.grid_sizes == (14, 7, 4)
    model = TRHashObjectDetector(config)
    raw = model(torch.randn(1, 3, 224, 224))
    assert raw.shape[1] == 14**2 + 7**2 + 4**2


def test_dynamic_assignment_selects_multiple_quality_weighted_cells():
    torch.manual_seed(0)
    config = _tiny_config(assignment_top_k=3)
    model = TRHashObjectDetector(config)
    raw = model(torch.randn(1, 3, 32, 32))
    targets = [torch.tensor([[0.5, 0.5, 0.25, 0.25, 2.0]])]
    assigned = model._assign_targets(targets, raw.device, decoded=model.decode(raw))
    assert 1 <= assigned["positive_mask"].sum() <= 3
    quality = assigned["quality"][assigned["positive_mask"]]
    assert torch.all((quality >= 0.05) & (quality <= 1.0))


def test_stal_assigns_more_fine_grid_positives_to_small_objects():
    config = _tiny_config(
        assignment_top_k=1,
        stal_enabled=True,
        stal_small_object_threshold=0.10,
        stal_top_k=4,
    )
    model = TRHashObjectDetector(config)
    raw = torch.zeros(1, config.num_cells, config.prediction_width)
    targets = [torch.tensor([[0.5, 0.5, 0.05, 0.05, 2.0]])]

    assigned = model._assign_targets(targets, raw.device, decoded=model.decode(raw))

    positives = torch.nonzero(assigned["positive_mask"][0]).flatten()
    assert positives.numel() >= 1
    assert torch.all(positives < config.grid_sizes[0] ** 2)


def test_decode_produces_bounded_normalized_boxes():
    config = _tiny_config()
    model = TRHashObjectDetector(config).eval()
    pixels = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        decoded = model.decode(model(pixels))
    boxes = decoded["boxes"]
    assert boxes.shape == (2, config.num_cells, 4)
    assert torch.all(boxes >= 0.0) and torch.all(boxes <= 1.0)
    assert torch.all(boxes[..., 0] <= boxes[..., 2])
    assert torch.all(boxes[..., 1] <= boxes[..., 3])
    scores = decoded["class_scores"]
    assert torch.all((scores >= 0.0) & (scores <= 1.0))


def test_decode_ltrb_distances_are_local_to_each_stride():
    config = _tiny_config(reg_max=0)
    model = TRHashObjectDetector(config).eval()
    raw = torch.zeros(1, config.num_cells, config.prediction_width)
    baseline = model.decode(raw)["boxes"][0, 0, 2]
    raw[0, 0, 2] = 2.0
    expanded = model.decode(raw)["boxes"][0, 0, 2]

    assert expanded > baseline


def test_loss_backward_reaches_head_and_backbone_experts():
    torch.manual_seed(1)
    config = _tiny_config()
    model = TRHashObjectDetector(config)
    pixels = torch.randn(2, 3, 32, 32)
    raw = model(pixels)
    targets = [
        torch.tensor([[0.3, 0.4, 0.2, 0.3, 1.0], [0.7, 0.6, 0.1, 0.15, 3.0]]),
        torch.tensor([[0.5, 0.5, 0.4, 0.4, 0.0]]),
    ]
    losses = model.compute_loss(raw, targets)
    assert losses["loss"].ndim == 0
    assert torch.isfinite(losses["loss"])
    for key in ("quality_loss", "box_loss", "dfl_loss"):
        assert torch.isfinite(losses[key])

    losses["loss"].backward()
    assert model.head.regression_heads[0][-1].weight.grad is not None
    assert model.head.classification_heads[0][-1].weight.grad is not None
    grad = model.tower.blocks[0].mlp.expert_gate.grad
    assert grad is not None
    assert grad.abs().sum() > 0


def test_progressive_loss_interpolates_quality_and_box_weights():
    config = _tiny_config(
        progressive_loss_enabled=True,
        progressive_box_start=0.5,
        progressive_quality_start=1.5,
    )
    model = TRHashObjectDetector(config)
    raw = model(torch.randn(1, 3, 32, 32))
    targets = [torch.tensor([[0.5, 0.5, 0.3, 0.3, 1.0]])]

    start = model.compute_loss(raw, targets, training_progress=0.0)
    end = model.compute_loss(raw, targets, training_progress=1.0)
    expected_start = (
        1.5 * config.quality_loss_weight * start["quality_loss"]
        + 0.5 * config.box_loss_weight * start["box_loss"]
    )

    assert torch.allclose(start["loss"], expected_start)
    assert torch.allclose(
        end["loss"],
        config.quality_loss_weight * end["quality_loss"]
        + config.box_loss_weight * end["box_loss"],
    )


def test_loss_handles_images_with_no_ground_truth_boxes():
    config = _tiny_config()
    model = TRHashObjectDetector(config)
    pixels = torch.randn(2, 3, 32, 32)
    raw = model(pixels)
    targets = [torch.empty(0, 5), torch.tensor([[0.5, 0.5, 0.2, 0.2, 2.0]])]
    losses = model.compute_loss(raw, targets)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()


def test_loss_accepts_bfloat16_predictions_with_float32_targets():
    config = _tiny_config()
    model = TRHashObjectDetector(config)
    raw = model(torch.randn(1, 3, 32, 32)).to(torch.bfloat16)
    targets = [torch.tensor([[0.5, 0.5, 0.2, 0.2, 2.0]])]

    losses = model.compute_loss(raw, targets)

    assert torch.isfinite(losses["loss"])


def test_predict_returns_per_image_detections_with_matching_lengths():
    config = _tiny_config()
    model = TRHashObjectDetector(config).eval()
    pixels = torch.randn(3, 3, 32, 32)
    detections = model.predict(pixels, confidence_threshold=0.0, iou_threshold=0.5)
    assert len(detections) == 3
    for entry in detections:
        n = entry["boxes"].shape[0]
        assert entry["scores"].shape == (n,)
        assert entry["labels"].shape == (n,)
        assert n <= config.num_cells


def test_predict_threshold_reduces_or_keeps_detection_count():
    config = _tiny_config()
    model = TRHashObjectDetector(config).eval()
    pixels = torch.randn(2, 3, 32, 32)
    loose = model.predict(pixels, confidence_threshold=0.0, iou_threshold=0.5)
    strict = model.predict(pixels, confidence_threshold=0.99, iou_threshold=0.5)
    for loose_entry, strict_entry in zip(loose, strict):
        assert strict_entry["boxes"].shape[0] <= loose_entry["boxes"].shape[0]


def test_prediction_runs_nms(monkeypatch):
    model = TRHashObjectDetector(_tiny_config()).eval()
    calls = []

    def record_nms(boxes, scores, labels, iou_threshold, max_detections):
        calls.append(len(boxes))
        return torch.argsort(scores, descending=True)[:max_detections]

    monkeypatch.setattr(
        "complexity.generative.detection.model.class_aware_nms",
        record_nms,
    )
    model.predict(
        torch.randn(1, 3, 32, 32),
        confidence_threshold=0.0,
    )

    assert calls
