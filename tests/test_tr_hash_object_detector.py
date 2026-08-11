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
    assert raw.shape == (2, config.num_cells, 5 + config.num_classes)


def test_default_head_uses_three_feature_scales():
    config = _tiny_config()
    assert config.grid_sizes == (4, 2, 1)
    assert config.num_cells == 21
    model = TRHashObjectDetector(config)
    assert len(model.scale_heads) == 3
    assert len(model.fpn_downsamples) == 2


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
    quality = assigned["objectness"][assigned["positive_mask"]]
    assert torch.all((quality >= 0.2) & (quality <= 1.0))


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
    probs = decoded["class_probs"]
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2, config.num_cells), atol=1e-5)


def test_decode_center_offsets_can_leave_the_source_cell():
    config = _tiny_config()
    model = TRHashObjectDetector(config).eval()
    raw = torch.zeros(1, config.num_cells, 5 + config.num_classes)

    # The first fine-grid cell spans [0, 0.25]. A neighbouring-cell positive
    # must still be able to regress a target center well beyond x=0.25.
    raw[0, 0, 0] = 2.0
    decoded = model.decode(raw)

    assert torch.allclose(
        decoded["boxes_cxcywh"][0, 0, 0], torch.tensor(0.625)
    )


def test_unversioned_checkpoint_config_keeps_legacy_sigmoid_centers():
    values = _tiny_config().to_dict()
    values.pop("center_offset_mode")
    config = TRHashDetectorConfig.from_dict(values)
    model = TRHashObjectDetector(config).eval()
    raw = torch.zeros(1, config.num_cells, 5 + config.num_classes)
    raw[0, 0, 0] = 2.0

    center_x = model.decode(raw)["boxes_cxcywh"][0, 0, 0]

    assert config.center_offset_mode == "sigmoid"
    assert 0.0 <= center_x <= 0.25


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
    for key in ("objectness_loss", "box_loss", "class_loss"):
        assert torch.isfinite(losses[key])

    losses["loss"].backward()
    assert model.scale_heads[0][-1].weight.grad is not None
    grad = model.tower.blocks[0].mlp.expert_gate.grad
    assert grad is not None
    assert grad.abs().sum() > 0


def test_loss_handles_images_with_no_ground_truth_boxes():
    config = _tiny_config()
    model = TRHashObjectDetector(config)
    pixels = torch.randn(2, 3, 32, 32)
    raw = model(pixels)
    targets = [torch.empty(0, 5), torch.tensor([[0.5, 0.5, 0.2, 0.2, 2.0]])]
    losses = model.compute_loss(raw, targets)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()


def test_predict_returns_per_image_detections_with_matching_lengths():
    config = _tiny_config()
    model = TRHashObjectDetector(config).eval()
    pixels = torch.randn(3, 3, 32, 32)
    detections = model.predict(pixels, objectness_threshold=0.0, iou_threshold=0.5)
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
    loose = model.predict(pixels, objectness_threshold=0.0, iou_threshold=0.5)
    strict = model.predict(pixels, objectness_threshold=0.99, iou_threshold=0.5)
    for loose_entry, strict_entry in zip(loose, strict):
        assert strict_entry["boxes"].shape[0] <= loose_entry["boxes"].shape[0]
