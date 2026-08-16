import json
import sys

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file
from torch.utils.data import DataLoader, TensorDataset

from complexity.generative.detection import (
    CocoDetectionDataset,
    CocoVideoDetectionDataset,
    HuggingFaceDetectionDataset,
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
from complexity.generative.detection.data import _bound_and_split_stream
from complexity.generative.detection.distributed import DistributedEvalSampler
from complexity.generative.detection.training import (
    _average_precision_from_matches,
    _match_image_detections,
    accumulation_group,
    build_object_weight_table,
    detector_epoch_steps,
    detector_step_schedule,
    format_loss_metrics_for_logging,
    load_object_bucket_count_cache,
    load_pretrained_detector,
    normalized_weight_decay,
    optimizer_step_schedule,
    parse_args,
    resize_detector_inputs,
    resolve_accumulation_steps,
    resolve_warmup_steps,
    save_object_bucket_count_cache,
    should_validate_epoch,
    verify_loader_cardinality,
    vision_backend_summary,
)


def test_detection_framework_defaults_to_packed_mosaic_epochs(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["training", "--output", "run", "--optimizer", "musgd"],
    )
    assert parse_args().mosaic_packed_epoch is True

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "training",
            "--output",
            "run",
            "--optimizer",
            "musgd",
            "--no-mosaic-packed-epoch",
        ],
    )
    assert parse_args().mosaic_packed_epoch is False


def test_nominal_batch_recipe_is_world_size_invariant() -> None:
    assert (
        resolve_accumulation_steps(
            per_device_batch_size=16,
            world_size=1,
            nominal_batch_size=64,
        )
        == 4
    )
    assert (
        resolve_accumulation_steps(
            per_device_batch_size=8,
            world_size=4,
            nominal_batch_size=64,
        )
        == 2
    )
    assert (
        resolve_accumulation_steps(
            per_device_batch_size=16,
            world_size=8,
            nominal_batch_size=64,
        )
        == 1
    )
    assert (
        resolve_accumulation_steps(
            per_device_batch_size=8,
            world_size=4,
            nominal_batch_size=64,
            requested_steps=3,
        )
        == 3
    )


def test_optimizer_schedule_and_warmup_use_optimizer_steps() -> None:
    schedule = optimizer_step_schedule((10, 11, 3), accumulation_steps=4)
    assert schedule == (3, 3, 1)
    assert (
        resolve_warmup_steps(
            requested_steps=999,
            warmup_epochs=2.5,
            steps_in_first_epoch=schedule[0],
            total_steps=sum(schedule),
        )
        == 7
    )

    # Mosaic packing shrinks the realized optimizer-step budget; a raw
    # warmup-step count tuned against the unpacked schedule must scale down
    # by the same ratio instead of covering a larger fraction of the shorter
    # packed run.
    assert (
        resolve_warmup_steps(
            requested_steps=100,
            warmup_epochs=None,
            steps_in_first_epoch=100,
            total_steps=700,
            unpacked_total_steps=2800,
        )
        == 25
    )
    assert (
        resolve_warmup_steps(
            requested_steps=100,
            warmup_epochs=None,
            steps_in_first_epoch=100,
            total_steps=700,
            unpacked_total_steps=700,
        )
        == 100
    )

    groups = [accumulation_group(i, epoch_batches=10, accumulation_steps=4) for i in range(10)]
    assert groups == [
        (4, False),
        (4, False),
        (4, False),
        (4, True),
        (4, False),
        (4, False),
        (4, False),
        (4, True),
        (2, False),
        (2, True),
    ]


def test_weight_decay_is_normalized_to_effective_batch() -> None:
    assert normalized_weight_decay(
        6.4e-4,
        global_batch_size=32,
        accumulation_steps=2,
        nominal_batch_size=64,
    ) == pytest.approx(6.4e-4)


def test_accumulated_microbatches_match_one_large_batch_update() -> None:
    torch.manual_seed(123)
    large = torch.nn.Linear(3, 2, bias=False, dtype=torch.float64)
    accumulated = torch.nn.Linear(3, 2, bias=False, dtype=torch.float64)
    accumulated.load_state_dict(large.state_dict())
    inputs = torch.randn(4, 3, dtype=torch.float64)
    targets = torch.randn(4, 2, dtype=torch.float64)
    large_optimizer = torch.optim.SGD(large.parameters(), lr=0.1)
    accumulated_optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.1)

    torch.nn.functional.mse_loss(large(inputs), targets).backward()
    large_optimizer.step()

    for index in range(4):
        group_size, boundary = accumulation_group(
            index,
            epoch_batches=4,
            accumulation_steps=4,
        )
        loss = torch.nn.functional.mse_loss(
            accumulated(inputs[index : index + 1]),
            targets[index : index + 1],
        )
        (loss / group_size).backward()
        if boundary:
            accumulated_optimizer.step()
            accumulated_optimizer.zero_grad(set_to_none=True)

    assert torch.allclose(large.weight, accumulated.weight, atol=1e-12, rtol=1e-12)
    assert normalized_weight_decay(
        6.4e-4,
        global_batch_size=128,
        accumulation_steps=1,
        nominal_batch_size=64,
    ) == pytest.approx(1.28e-3)
    assert normalized_weight_decay(
        6.4e-4,
        global_batch_size=128,
        accumulation_steps=1,
        nominal_batch_size=0,
    ) == pytest.approx(6.4e-4)


def test_mosaic_alone_does_not_shrink_the_step_schedule():
    # packed_epochs defaults to 1: Mosaic-active epochs keep the natural,
    # un-packed step count. Mosaic tiles are downscaled, so crediting them
    # 1:1 against full-resolution steps would silently trade step count for
    # per-tile quality -- packed_epochs is the sole, explicit step divisor.
    assert (
        detector_epoch_steps(
            925,
            0,
            total_epochs=245,
            mosaic_probability=0.909,
            close_mosaic_epochs=10,
            mosaic_packed_epoch=True,
        )
        == 925
    )
    assert (
        detector_epoch_steps(
            925,
            235,
            total_epochs=245,
            mosaic_probability=0.909,
            close_mosaic_epochs=10,
            mosaic_packed_epoch=True,
        )
        == 925
    )

    schedule = detector_step_schedule(
        925,
        total_epochs=245,
        mosaic_probability=0.909,
        close_mosaic_epochs=10,
        mosaic_packed_epoch=True,
    )
    assert schedule == (925,) * 245


def test_four_packed_epochs_split_one_full_pass_into_quarters():
    schedule = detector_step_schedule(
        3697,
        total_epochs=4,
        mosaic_probability=1.0,
        close_mosaic_epochs=0,
        mosaic_packed_epoch=True,
        packed_epochs=4,
    )

    assert schedule == (925, 925, 925, 925)
    assert sum(schedule) >= 3697


def test_mosaic_packed_epoch_is_opt_in():
    schedule = detector_step_schedule(
        925,
        total_epochs=3,
        mosaic_probability=1.0,
        close_mosaic_epochs=0,
        mosaic_packed_epoch=False,
    )
    assert schedule == (925, 925, 925)


def test_loss_logger_separates_stationary_monitor_from_optimization_objective():
    logged = format_loss_metrics_for_logging(
        {
            "loss": 9.5,
            "monitor_loss": 4.2,
            "one_to_many_loss": 5.0,
            "one_to_many_monitor_loss": 4.1,
            "one_to_one_loss": 4.5,
            "one_to_one_monitor_loss": 4.3,
            "one_to_one_weight": 1.0,
        }
    )

    assert logged["loss"] == 4.2
    assert logged["optimization_loss"] == 9.5
    assert logged["one_to_many_loss"] == 4.1
    assert logged["one_to_many_optimization_loss"] == 5.0
    assert logged["one_to_one_loss"] == 4.3
    assert logged["one_to_one_optimization_loss"] == 4.5

    legacy = format_loss_metrics_for_logging({"loss": 7.0})
    assert legacy["loss"] == 7.0
    assert legacy["optimization_loss"] == 7.0


def test_memory_efficient_quality_focal_loss_matches_dense_autograd():
    torch.manual_seed(11)
    targets = torch.zeros(2, 5, 3, dtype=torch.float64)
    targets[0, 1, 2] = 0.7
    targets[1, 3, 0] = 0.4
    weights = torch.rand(2, 5, 1, dtype=torch.float64) + 0.2
    expected_logits = torch.randn(2, 5, 3, dtype=torch.float64, requires_grad=True)
    actual_logits = expected_logits.detach().clone().requires_grad_(True)

    probabilities = expected_logits.sigmoid()
    modulation = (targets - probabilities).abs().square()
    dense_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        expected_logits,
        targets,
        reduction="none",
    )
    broadcast_weights = torch.broadcast_to(weights, targets.shape)
    normalizer = (broadcast_weights * (targets > 0)).sum().clamp_min(1)
    expected = (dense_loss * modulation * broadcast_weights).sum() / normalizer
    actual = quality_focal_loss(actual_logits, targets, weights=weights)

    expected.backward()
    actual.backward()
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)
    assert torch.allclose(
        actual_logits.grad,
        expected_logits.grad,
        atol=1e-10,
        rtol=1e-10,
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


def test_validation_cadence_always_includes_final_epoch():
    selected = [
        epoch for epoch in range(12) if should_validate_epoch(epoch, total_epochs=12, eval_every=5)
    ]
    assert selected == [4, 9, 11]


def test_vision_backend_summary_reports_and_can_require_triton():
    model = TRHashObjectDetector(
        TRHashDetectorConfig(
            image_size=32,
            patch_size=8,
            vision_hidden_size=32,
            vision_layers=3,
            vision_stage_depths=(1, 1, 1),
            vision_window_size=2,
            vision_heads=4,
            vision_expert_width=8,
        )
    )

    summary = vision_backend_summary(model, "cpu")
    assert summary["selected_backend"] == "pytorch"
    with pytest.raises(RuntimeError, match="Triton is required"):
        vision_backend_summary(model, "cpu", require_triton=True)


def test_distributed_validation_sampler_has_no_padding_duplicates():
    dataset = list(range(11))
    shards = [list(DistributedEvalSampler(dataset, rank, world_size=4)) for rank in range(4)]
    flattened = [index for shard in shards for index in shard]
    assert sorted(flattened) == list(range(len(dataset)))
    assert len(flattened) == len(set(flattened))


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


def test_distributed_checkpoint_records_and_validates_world_size(tmp_path):
    _, optimizer, scheduler = _checkpoint_test_optimizer()
    checkpoint = tmp_path / "distributed"
    checkpoint.mkdir()
    rng_states = [
        {"torch": torch.get_rng_state(), "cuda": torch.tensor([rank], dtype=torch.uint8)}
        for rank in range(2)
    ]
    save_training_state(
        checkpoint,
        optimizer,
        scheduler,
        epoch=0,
        batch_in_epoch=0,
        step=1,
        best_map50=0.0,
        running_losses={},
        running_loss_steps=0,
        total_epochs=2,
        steps_per_epoch=3,
        training_options={"world_size": 2},
        distributed_rng_states=rng_states,
    )
    state = torch.load(checkpoint / "training_state.pt", weights_only=True)
    assert state["world_size"] == 2
    assert len(state["distributed_rng_states"]) == 2

    _, restored_optimizer, restored_scheduler = _checkpoint_test_optimizer()
    with pytest.raises(ValueError, match="world size differs"):
        load_training_state(
            checkpoint,
            restored_optimizer,
            restored_scheduler,
            total_epochs=2,
            steps_per_epoch=3,
            training_options={"world_size": 2},
            world_size=1,
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


class _InMemoryHFDetectionDataset(HuggingFaceDetectionDataset):
    def __init__(self, rows, **kwargs):
        self.rows = rows
        super().__init__("test/detection", "train", **kwargs)

    def _stream(self, *, metadata_only: bool = False):
        if metadata_only:
            return (
                {name: value for name, value in row.items() if name != "image"} for row in self.rows
            )
        return iter(self.rows[: self.local_examples])


def test_hf_detection_stream_normalizes_boxes_and_has_equal_rank_length():
    row = {
        "image": Image.new("RGB", (100, 50), "white"),
        "width": 100,
        "height": 50,
        "annotations": json.dumps(
            [
                {"bbox": [-10, 5, 30, 20], "category_id": 1},
                {"bbox": [90, 40, 30, 20], "category_id": 3},
            ]
        ),
    }
    dataset = _InMemoryHFDetectionDataset(
        [row] * 4,
        num_examples=4,
        num_classes=3,
        image_size=32,
        rank=0,
        world_size=2,
    )

    assert len(dataset) == 2
    samples = list(dataset)
    assert len(samples) == 2
    assert samples[0][0].shape == (3, 32, 32)
    targets = samples[0][1]
    assert targets.shape == (2, 5)
    assert targets[:, :4].min() >= 0
    assert targets[:, :4].max() <= 1
    assert targets[:, 4].tolist() == [0.0, 2.0]


def test_hf_detection_metadata_counts_three_dimensional_buckets():
    rows = [
        {
            "image": Image.new("RGB", (100, 100)),
            "width": 100,
            "height": 100,
            "annotations": json.dumps([{"bbox": [0, 0, 10, 10], "category_id": 1}] * density),
        }
        for density in (1, 5, 12)
    ]
    dataset = _InMemoryHFDetectionDataset(
        rows,
        num_examples=3,
        num_classes=2,
        image_size=32,
    )

    counts = dataset.object_bucket_counts()
    assert counts.shape == (2, 3, 3)
    assert counts[0, 0].tolist() == [1.0, 5.0, 12.0]


def test_hf_detection_dataset_retains_metadata_projection_glob():
    dataset = HuggingFaceDetectionDataset(
        "owner/detection",
        "train",
        num_examples=8,
        num_classes=2,
        image_size=32,
        metadata_file_glob="data/train-*.parquet",
    )

    assert dataset.metadata_file_glob == "data/train-*.parquet"


def test_hf_detection_dataset_resolves_local_parquet_glob(tmp_path):
    dataset_root = tmp_path / "object365"
    dataset_root.mkdir()
    dataset = HuggingFaceDetectionDataset(
        str(dataset_root),
        "train",
        num_examples=8,
        num_classes=2,
        image_size=32,
    )

    assert dataset._repository_file_uri("data/train-*.parquet") == str(
        dataset_root / "data/train-*.parquet"
    )


def test_hf_detection_stream_bounds_globally_before_rank_sharding():
    operations = []

    class FakeStream:
        def take(self, count):
            operations.append(("take", count))
            return self

    def split(stream, *, rank, world_size):
        operations.append(("split", rank, world_size))
        return stream

    stream = FakeStream()
    result = _bound_and_split_stream(
        stream,
        local_examples=4,
        rank=2,
        world_size=4,
        splitter=split,
    )

    assert result is stream
    assert operations == [("take", 16), ("split", 2, 4)]


def test_loader_cardinality_guard_rejects_silent_truncation():
    loader = DataLoader(TensorDataset(torch.arange(4)), batch_size=2)

    verify_loader_cardinality(loader, 2, None, phase="test")
    with pytest.raises(RuntimeError, match="test loader ended"):
        verify_loader_cardinality(loader, 1, None, phase="test")


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
    first = SyntheticShapesDataset(length=4, image_size=64, seed=7, resample_each_epoch=True)
    second = SyntheticShapesDataset(length=4, image_size=64, seed=7, resample_each_epoch=True)
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
    boxes = torch.tensor([[0.0, 0.0, 0.1, 0.1], [0.2, 0.2, 0.3, 0.3], [0.4, 0.4, 0.5, 0.5]])
    scores = torch.tensor([0.7, 0.9, 0.8])
    labels = torch.tensor([0, 0, 0])
    kept = class_aware_nms(boxes, scores, labels, iou_threshold=0.5, max_detections=2)
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


def test_coco_video_dataset_builds_boundary_safe_center_labeled_clips(tmp_path):
    images_dir = tmp_path / "video"
    images_dir.mkdir()
    for frame_id, value in enumerate((10, 80, 160)):
        Image.new("RGB", (32, 32), color=(value, 20, 30)).save(images_dir / f"{frame_id}.png")
    annotations = {
        "images": [
            {
                "id": frame_id + 1,
                "video_id": 7,
                "frame_id": frame_id,
                "file_name": f"{frame_id}.png",
                "width": 32,
                "height": 32,
            }
            for frame_id in range(3)
        ],
        "annotations": [
            {"image_id": 2, "bbox": [8, 8, 8, 8], "category_id": 4},
        ],
    }
    annotations_path = tmp_path / "video.json"
    annotations_path.write_text(json.dumps(annotations))
    dataset = CocoVideoDetectionDataset(
        annotations_path,
        images_dir,
        image_size=32,
        clip_frames=3,
    )

    boundary_clip, boundary_targets = dataset[0]
    center_clip, center_targets = dataset[1]

    assert boundary_clip.shape == (3, 3, 32, 32)
    torch.testing.assert_close(boundary_clip[0], boundary_clip[1])
    assert boundary_targets.shape == (0, 5)
    assert center_targets.shape == (1, 5)
    torch.testing.assert_close(
        center_targets[0],
        torch.tensor([0.375, 0.375, 0.25, 0.25, 0.0]),
    )
    assert not torch.equal(center_clip[0], center_clip[2])


def test_resize_detector_inputs_preserves_video_axes():
    clips = torch.randn(2, 3, 3, 16, 16)
    resized = resize_detector_inputs(clips, (24, 24))
    assert resized.shape == (2, 3, 3, 24, 24)


def test_object_weight_table_upweights_rare_three_dimensional_buckets(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for index in range(4):
        Image.new("RGB", (32, 32)).save(images / f"{index}.png")
        common = "\n".join("0 0.5 0.5 0.2 0.2" for _ in range(4))
        rare = "\n1 0.5 0.5 0.05 0.05" if index == 0 else ""
        (labels / f"{index}.txt").write_text(common + rare + "\n")
    dataset = YoloDetectionDataset(images, labels, image_size=32)

    weights, counts = build_object_weight_table(
        dataset,
        2,
        beta=0.9,
        max_weight=4.0,
    )

    assert counts.sum() == 17
    assert weights[1, 0, 1] > weights[0, 1, 1]


def test_stream_object_bucket_cache_requires_matching_dataset_identity(tmp_path):
    cache = tmp_path / "object_bucket_counts.json"
    counts = torch.arange(18, dtype=torch.float64).reshape(2, 3, 3)
    identity = {
        "dataset_id": "owner/detection",
        "split": "train",
        "num_examples": 123,
        "num_classes": 2,
        "category_id_offset": 1,
        "metadata_file_glob": "data/train-*.parquet",
    }

    save_object_bucket_count_cache(cache, counts, **identity)

    loaded = load_object_bucket_count_cache(cache, **identity)
    assert loaded is not None
    assert torch.equal(loaded, counts)
    assert (
        load_object_bucket_count_cache(
            cache,
            **{**identity, "num_examples": 124},
        )
        is None
    )


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
        vision_layers=3,
        vision_stage_depths=(1, 1, 1),
        vision_window_size=2,
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


def test_detector_transfer_preserves_regression_and_mapped_classes(tmp_path):
    common = dict(
        image_size=32,
        patch_size=8,
        vision_hidden_size=32,
        vision_layers=3,
        vision_stage_depths=(1, 1, 1),
        vision_window_size=2,
        vision_heads=4,
        vision_num_experts=4,
        vision_top_k=2,
        vision_expert_width=16,
        level_adapters_enabled=True,
        class_level_hash_gate_enabled=True,
        object_weighting_enabled=True,
    )
    source_config = TRHashDetectorConfig(**common, num_classes=3)
    source = TRHashObjectDetector(source_config)
    with torch.no_grad():
        source.tower.patch_embed[-1].weight.fill_(0.125)
        source.head.regression_heads[0][1].weight.fill_(0.375)
        source.level_adapters.adapters[0].layers[-1].weight.fill_(0.25)
        source.class_level_hash_gate.score.weight.fill_(0.5)
        for head in source.head.classification_heads:
            final = head[-1]
            for output_index in range(final.out_features):
                final.weight[output_index].fill_(float(output_index))
                final.bias[output_index] = float(output_index)

    checkpoint = tmp_path / "source_detector"
    checkpoint.mkdir()
    save_file(
        {name: value.detach().contiguous() for name, value in source.state_dict().items()},
        str(checkpoint / "model.safetensors"),
    )
    (checkpoint / "config.json").write_text(json.dumps(source_config.to_dict()))

    target = TRHashObjectDetector(TRHashDetectorConfig(**common, num_classes=2))
    unmapped_weight = target.head.classification_heads[0][-1].weight[1].detach().clone()
    unmapped_bias = target.head.classification_heads[0][-1].bias[1].detach().clone()
    load_pretrained_detector(target, checkpoint, class_mapping={0: 2})

    assert torch.all(target.tower.patch_embed[-1].weight == 0.125)
    assert torch.all(target.head.regression_heads[0][1].weight == 0.375)
    assert torch.all(target.level_adapters.adapters[0].layers[-1].weight == 0.25)
    assert torch.all(target.class_level_hash_gate.score.weight == 0.5)
    for level, head in enumerate(target.head.classification_heads):
        final = head[-1]
        source_final = source.head.classification_heads[level][-1]
        assert torch.equal(final.weight[0], source_final.weight[2])
        assert torch.equal(final.bias[0], source_final.bias[2])
    assert torch.equal(target.head.classification_heads[0][-1].weight[1], unmapped_weight)
    assert torch.equal(target.head.classification_heads[0][-1].bias[1], unmapped_bias)
