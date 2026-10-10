import hashlib
import json
from types import SimpleNamespace

import pytest

from complexity.deploy.onnx_detector.types import (
    Detection,
    DetectionResult,
    TimingBreakdown,
)
from complexity.vision.detection import report_merging
from complexity.vision.detection.evaluation import (
    BRANCHES,
    _branch_contract,
    _branches_to_run,
    _checkpoint_sha256,
    _image_list_sha256,
    _percentile,
    _timing_summary,
    _write_markdown_report,
)
from complexity.vision.detection.onnx_evaluation import _run_onnx_branch
from complexity.vision.detection.quantized_artifacts import check_quantized_accuracy_report
from complexity.vision.detection.report_merging import merge_reports


def test_both_branches_require_end_to_end_head():
    with pytest.raises(ValueError, match="NMS-free"):
        _branches_to_run("both", has_nms_free=False)


def test_o2m_only_does_not_require_end_to_end_head():
    assert _branches_to_run("o2m-nms", has_nms_free=False) == ("o2m-nms",)


def test_both_branches_keep_fixed_comparison_order():
    assert _branches_to_run("both", has_nms_free=True) == BRANCHES


def test_timing_summary_uses_measured_batches_and_image_count():
    summary = _timing_summary([0.1, 0.3, 0.2], image_count=12)

    assert summary["mean_batch_ms"] == pytest.approx(200.0)
    assert summary["p50_batch_ms"] == pytest.approx(200.0)
    assert summary["p95_batch_ms"] == pytest.approx(300.0)
    assert summary["images_per_second"] == pytest.approx(20.0)
    assert summary["measured_seconds"] == pytest.approx(0.6)


def test_percentile_handles_empty_measurements():
    assert _percentile([], 0.95) == 0.0


def test_image_list_hash_uses_sorted_manifest_contract():
    class Coco:
        imgs = {
            7: {"file_name": "000000000007.jpg", "width": 640, "height": 427},
            3: {"file_name": "000000000003.jpg", "width": 500, "height": 375},
        }

    first = _image_list_sha256(Coco(), [3, 7])
    second = _image_list_sha256(Coco(), [3, 7])
    different_order = _image_list_sha256(Coco(), [7, 3])

    assert first == second
    assert first != different_order


def test_checkpoint_sha256_hashes_loaded_directory_weights(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    model_weights = checkpoint / "model.safetensors"
    ema_weights = checkpoint / "ema.safetensors"
    model_weights.write_bytes(b"model weights")
    ema_weights.write_bytes(b"ema weights")

    assert _checkpoint_sha256(checkpoint) == hashlib.sha256(b"ema weights").hexdigest()

    ema_weights.unlink()

    assert _checkpoint_sha256(checkpoint) == hashlib.sha256(b"model weights").hexdigest()


def test_branch_contract_distinguishes_nms_requirements():
    assert "class-aware NMS" in _branch_contract("o2m-nms")["postprocess"]
    assert "no NMS" in _branch_contract("nms-free")["postprocess"]


def test_markdown_report_contains_release_metrics(tmp_path):
    report = {
        "backend": "pytorch",
        "framework_commit": "abc123",
        "checkpoint": "checkpoint.pt",
        "checkpoint_sha256": "hash",
        "dataset": {
            "name": "coco-2017",
            "split": "val2017",
            "evaluated_images": 5000,
            "annotations_sha256": "annotations",
            "image_list_sha256": "images",
        },
        "environment": {
            "python": "3.11",
            "os": "linux",
            "torch": "2.6.0",
            "onnxruntime": "1.23.2",
            "cuda_available": False,
            "torch_cuda": None,
            "tensorrt": None,
        },
        "branches": {
            "o2m-nms": {
                "metrics": {
                    "map50_95": 0.2,
                    "map50": 0.3,
                    "map75": 0.1,
                    "ap_small": 0.01,
                    "ap_medium": 0.2,
                    "ap_large": 0.3,
                    "ar_100": 0.4,
                }
            }
        },
    }
    output = tmp_path / "evaluation.md"

    _write_markdown_report(report, output)

    text = output.read_text(encoding="utf-8")
    assert "Vision v8 COCO Accuracy Report" in text
    assert "| o2m-nms | 0.200000 | 0.300000" in text


def test_merge_onnx_reports_keeps_both_branch_artifact_hashes():
    report = {
        "backend": "onnx",
        "framework_commit": "abc123",
        "checkpoint": "o2m.onnx",
        "checkpoint_sha256": "a" * 64,
        "model": "o2m.onnx",
        "metadata": "o2m.json",
        "metadata_sha256": "b" * 64,
        "dataset": {"name": "coco-2017"},
        "environment": {},
        "protocol": {"seed": 0},
        "branches": {"o2m-nms": {"metrics": {}}},
    }
    nms_free = {
        **report,
        "checkpoint": "nms_free.onnx",
        "checkpoint_sha256": "c" * 64,
        "model": "nms_free.onnx",
        "metadata": "nms_free.json",
        "metadata_sha256": "d" * 64,
        "branches": {"nms-free": {"metrics": {}}},
    }

    merged = merge_reports([report, nms_free])

    assert set(merged["branches"]) == {"o2m-nms", "nms-free"}
    assert merged["branches"]["o2m-nms"]["checkpoint_sha256"] == "a" * 64
    assert merged["branches"]["nms-free"]["checkpoint_sha256"] == "c" * 64
    assert merged["branches"]["nms-free"]["metadata_sha256"] == "d" * 64


def test_merge_onnx_reports_can_build_precision_nested_quantized_report():
    fp32 = {
        "backend": "onnx",
        "precision": "fp32",
        "framework_commit": "abc123",
        "checkpoint": "o2m_fp32.onnx",
        "checkpoint_sha256": "a" * 64,
        "model": "o2m_fp32.onnx",
        "metadata": "o2m.json",
        "metadata_sha256": "b" * 64,
        "dataset": {"name": "coco-2017", "image_ids": [1, 2]},
        "environment": {},
        "protocol": {"seed": 0},
        "branches": {
            "o2m-nms": {
                "branch": "o2m-nms",
                "metrics": {"map50_95": 0.2, "map50": 0.3},
            }
        },
    }
    fp16 = {
        **fp32,
        "precision": "fp16",
        "checkpoint": "o2m_fp16.onnx",
        "checkpoint_sha256": "c" * 64,
        "model": "o2m_fp16.onnx",
        "branches": {
            "o2m-nms": {
                "branch": "o2m-nms",
                "metrics": {"map50_95": 0.198, "map50": 0.295},
            }
        },
    }

    merged = merge_reports([fp32, fp16])
    thresholds = {
        "release_policy": {"required_precisions": ["fp32", "fp16"]},
        "precisions": {"fp16": {"max_map50_95_drop": 0.005, "max_map50_drop": 0.01}},
    }

    assert set(merged["branches"]["o2m-nms"]) == {"fp32", "fp16"}
    assert merged["branches"]["o2m-nms"]["fp16"]["checkpoint_sha256"] == "c" * 64
    assert (
        check_quantized_accuracy_report(
            merged,
            thresholds,
            required_branches=["o2m-nms"],
        )
        == []
    )


def test_merge_onnx_reports_preserves_environment_per_precision():
    fp32 = {
        "backend": "onnx",
        "precision": "fp32",
        "framework_commit": "abc123",
        "checkpoint": "o2m_fp32.onnx",
        "checkpoint_sha256": "a" * 64,
        "model": "o2m_fp32.onnx",
        "metadata": "o2m.json",
        "metadata_sha256": "b" * 64,
        "dataset": {"name": "coco-2017", "image_ids": [1, 2]},
        "environment": {
            "requested_provider": ["CPUExecutionProvider"],
            "actual_provider": "CPUExecutionProvider",
        },
        "protocol": {"seed": 0},
        "branches": {
            "o2m-nms": {
                "branch": "o2m-nms",
                "metrics": {"map50_95": 0.2, "map50": 0.3},
            }
        },
    }
    fp16 = {
        **fp32,
        "precision": "fp16",
        "checkpoint": "o2m_fp16.onnx",
        "checkpoint_sha256": "c" * 64,
        "environment": {
            "requested_provider": ["CUDAExecutionProvider"],
            "actual_provider": "CUDAExecutionProvider",
        },
        "branches": {
            "o2m-nms": {
                "branch": "o2m-nms",
                "metrics": {"map50_95": 0.198, "map50": 0.295},
            }
        },
    }

    merged = merge_reports([fp32, fp16])

    assert merged["branches"]["o2m-nms"]["fp32"]["environment"] == fp32["environment"]
    assert merged["branches"]["o2m-nms"]["fp16"]["environment"] == fp16["environment"]


def test_merge_onnx_reports_rejects_mismatched_protocol():
    first = {
        "backend": "onnx",
        "framework_commit": "abc123",
        "dataset": {"name": "coco-2017"},
        "protocol": {"seed": 0},
        "branches": {"o2m-nms": {"metrics": {}}},
    }
    second = {
        **first,
        "protocol": {"seed": 1},
        "branches": {"nms-free": {"metrics": {}}},
    }

    with pytest.raises(ValueError, match="protocol"):
        merge_reports([first, second])


def test_packaged_merge_writes_json_and_markdown_without_repository_paths(
    tmp_path, monkeypatch
):
    report = {
        "backend": "onnx",
        "framework_commit": "abc123",
        "checkpoint": "detector.onnx",
        "checkpoint_sha256": "a" * 64,
        "dataset": {
            "name": "coco-2017",
            "split": "val2017",
            "evaluated_images": 2,
            "annotations_sha256": "b" * 64,
            "image_list_sha256": "c" * 64,
        },
        "environment": {
            "python": "3.11",
            "os": "linux",
            "torch": "2.6.0",
            "onnxruntime": "1.23.2",
            "cuda_available": False,
            "torch_cuda": None,
            "tensorrt": None,
        },
        "protocol": {"seed": 0},
        "branches": {
            "o2m-nms": {
                "metrics": {
                    "map50_95": 0.2,
                    "map50": 0.3,
                    "map75": 0.1,
                    "ap_small": 0.01,
                    "ap_medium": 0.2,
                    "ap_large": 0.3,
                    "ar_100": 0.4,
                }
            }
        },
    }
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(report), encoding="utf-8")
    output_path = tmp_path / "merged"
    monkeypatch.chdir(tmp_path)

    report_merging.main([str(input_path), "--output", str(output_path)])

    merged = json.loads((output_path / "evaluation.json").read_text(encoding="utf-8"))
    assert merged["branches"]["o2m-nms"]["checkpoint_sha256"] == "a" * 64
    assert "o2m-nms" in (output_path / "evaluation.md").read_text(encoding="utf-8")


def test_packaged_onnx_evaluator_maps_boxes_categories_and_measured_latency(tmp_path):
    image_root = tmp_path / "images"
    paths = []
    coco = SimpleNamespace(imgs={7: {"file_name": "000000000007.jpg"}})

    class Pipeline:
        def predict(self, path):
            paths.append(path)
            return DetectionResult(
                detections=[
                    Detection(
                        box_norm=(0.1, 0.2, 0.4, 0.6),
                        box_pixel=(10.0, 20.0, 40.0, 60.0),
                        class_id=1,
                        score=0.75,
                    )
                ],
                timing=TimingBreakdown(
                    preprocess_ms=2.0, inference_ms=4.0, postprocess_ms=1.0
                ),
            )

    predictions, timing = _run_onnx_branch(
        Pipeline(), coco, image_root, [7], [1, 3]
    )

    assert paths == [image_root / "000000000007.jpg"]
    assert predictions == [
        {"image_id": 7, "category_id": 3, "bbox": [10.0, 20.0, 30.0, 40.0], "score": 0.75}
    ]
    assert timing["inference"]["measured_ms"] == 4.0
    assert timing["images_per_second"] == 250.0


def test_packaged_onnx_evaluator_rejects_class_outside_coco_mapping(tmp_path):
    result = DetectionResult(
        detections=[Detection((0, 0, 1, 1), (0, 0, 10, 10), 1, 0.9)]
    )
    pipeline = SimpleNamespace(predict=lambda path: result)
    coco = SimpleNamespace(imgs={7: {"file_name": "image.jpg"}})

    with pytest.raises(ValueError, match="class ID outside COCO mapping"):
        _run_onnx_branch(pipeline, coco, tmp_path, [7], [1])
