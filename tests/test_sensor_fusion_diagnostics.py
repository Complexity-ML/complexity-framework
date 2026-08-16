from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from complexity.generative.sensor_fusion import (
    TRHashSensorFusionClassifier,
    TRHashSensorFusionConfig,
)
from complexity.generative.sensor_fusion.diagnostics import (
    classification_metrics_from_confusion,
    evaluate_late_fusion_sweep,
    evaluate_sensor_mode,
    hash_route_diagnostics,
)


def test_classification_metrics_from_confusion_reports_macro_and_missing_classes():
    confusion = torch.tensor(
        [
            [3, 1, 0],
            [0, 2, 2],
            [0, 0, 0],
        ]
    )

    metrics = classification_metrics_from_confusion(confusion)

    assert metrics["examples"] == 8
    assert metrics["top1_accuracy"] == pytest.approx(5 / 8)
    assert metrics["macro_accuracy"] == pytest.approx((0.75 + 0.5) / 2)
    assert metrics["per_class_examples"] == [4, 4, 0]
    assert metrics["per_class_accuracy"] == [0.75, 0.5, 0.0]


def test_classification_metrics_rejects_non_square_matrix():
    with pytest.raises(ValueError, match="square"):
        classification_metrics_from_confusion(torch.zeros(2, 3))


def test_late_fusion_sweep_can_recover_reliable_skeleton_predictions():
    class CalibrationModel:
        config = SimpleNamespace(num_classes=2, late_fusion_weight=0.5)

        def __call__(self, inputs, *, modality_mask):
            del inputs, modality_mask
            fused = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
            modality = torch.zeros(2, 6, 2)
            modality[:, 5] = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
            return {
                "fused_logits": fused,
                "modality_logits": modality,
                "modality_weights": torch.full((2, 6), 1 / 6),
            }

    loader = [
        {
            "inputs": {},
            "modality_mask": {},
            "labels": torch.tensor([0, 1]),
        }
    ]
    report = evaluate_late_fusion_sweep(
        CalibrationModel(),
        loader,
        torch.device("cpu"),
        precision="fp32",
    )

    assert report["best"]["top1_accuracy"] == 1.0
    assert report["best"]["skeleton_boost"] > 1.0


def test_sensor_mode_reports_each_validation_subject_separately():
    class SubjectModel:
        config = SimpleNamespace(num_classes=2)

        def __call__(self, inputs, labels, *, modality_mask):
            del inputs, modality_mask
            logits = torch.tensor([[9.0, 0.0], [9.0, 0.0], [0.0, 9.0]])
            return {
                "logits": logits,
                "loss": torch.nn.functional.cross_entropy(logits, labels),
            }

    loader = [
        {
            "inputs": {},
            "modality_mask": {},
            "labels": torch.tensor([0, 1, 1]),
            "metadata": [
                {"user_id": 8},
                {"user_id": 8},
                {"user_id": 23},
            ],
        }
    ]
    report = evaluate_sensor_mode(
        SubjectModel(),
        loader,
        torch.device("cpu"),
        modality=None,
        precision="fp32",
    )

    assert report["top1_accuracy"] == pytest.approx(2 / 3)
    assert report["by_subject"]["8"]["top1_accuracy"] == pytest.approx(0.5)
    assert report["by_subject"]["23"]["top1_accuracy"] == pytest.approx(1.0)


def test_hash_route_diagnostics_covers_class_modality_routes():
    config = TRHashSensorFusionConfig(
        num_classes=4,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_experts=8,
        top_k=2,
        shared_width=32,
        expert_width=8,
        precision="fp32",
        vision_image_size=16,
        vision_patch_size=4,
        vision_hidden_size=32,
        vision_layers=2,
        vision_heads=4,
        vision_num_experts=4,
        vision_top_k=2,
        vision_expert_width=8,
        vision_stage_depths=(1, 1),
        vision_window_size=2,
        visual_token_grid=(2, 2, 2),
        radar_features=5,
        sequence_tokens=4,
    )
    report = hash_route_diagnostics(TRHashSensorFusionClassifier(config))

    gate = report["class_modality_gate"]
    assert gate is not None
    assert len(gate["route_table"]) == config.top_k
    assert len(gate["route_table"][0]) == 6 * config.num_classes
    assert set(gate["expert_assignments_by_modality"]) == {
        "depth",
        "ir",
        "thermal",
        "imu",
        "radar",
        "skeleton",
    }
    assert len(gate["expert_assignments_by_class"]) == config.num_classes
    assert all(
        sum(counts) == config.top_k * 6
        for counts in gate["expert_assignments_by_class"]
    )
