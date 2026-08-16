import torch

from complexity.generative.detection.metrics import DetectionMetricsAccumulator


def _target(size: float, class_id: int) -> list[float]:
    return [0.5, 0.5, size, size, float(class_id)]


def _box(size: float) -> list[float]:
    return [0.5 - size / 2, 0.5 - size / 2, 0.5 + size / 2, 0.5 + size / 2]


def test_metrics_report_perfect_small_medium_and_large_ap():
    image_size = 640
    sizes = (20 / image_size, 60 / image_size, 120 / image_size)
    metrics = DetectionMetricsAccumulator(num_classes=3, image_size=image_size)
    metrics.update(
        boxes=torch.tensor([_box(size) for size in sizes]),
        scores=torch.tensor([0.9, 0.8, 0.7]),
        labels=torch.tensor([0, 1, 2]),
        targets=torch.tensor([_target(size, index) for index, size in enumerate(sizes)]),
    )

    result = metrics.compute(confidence_threshold=0.2)

    assert result["map50"] == 1.0
    assert result["map50_95"] == 1.0
    assert result["ap_small"] == 1.0
    assert result["ap_medium"] == 1.0
    assert result["ap_large"] == 1.0


def test_map50_95_penalizes_loose_localization_more_than_map50():
    metrics = DetectionMetricsAccumulator(num_classes=1, image_size=640)
    metrics.update(
        boxes=torch.tensor([[0.2, 0.2, 0.8, 0.8]]),
        scores=torch.tensor([0.9]),
        labels=torch.tensor([0]),
        targets=torch.tensor([_target(0.5, 0)]),
    )

    result = metrics.compute(confidence_threshold=0.2)

    assert result["map50"] == 1.0
    assert 0.0 < result["map50_95"] < result["map50"]


def test_compact_metric_states_merge_without_changing_global_ap():
    first = DetectionMetricsAccumulator(num_classes=1, image_size=640)
    second = DetectionMetricsAccumulator(num_classes=1, image_size=640)
    combined = DetectionMetricsAccumulator(num_classes=1, image_size=640)
    target = torch.tensor([_target(0.25, 0)])
    for accumulator, score in ((first, 0.9), (second, 0.8), (combined, 0.9)):
        accumulator.update(
            boxes=torch.tensor([_box(0.25)]),
            scores=torch.tensor([score]),
            labels=torch.tensor([0]),
            targets=target,
        )
    combined.update(
        boxes=torch.tensor([_box(0.25)]),
        scores=torch.tensor([0.8]),
        labels=torch.tensor([0]),
        targets=target,
    )

    merged = DetectionMetricsAccumulator(num_classes=1, image_size=640)
    merged.merge_state_dict(first.state_dict())
    merged.merge_state_dict(second.state_dict())

    assert merged.compute(0.2) == combined.compute(0.2)
