from scripts.check_vision_v8_coco_report import (
    check_report,
    compare_repeated_reports,
    determinism_tolerance,
)

HASH = "a" * 64
OTHER_HASH = "b" * 64


def _config() -> dict:
    return {
        "dataset": {
            "name": "coco-2017",
            "split": "val2017",
            "required_image_count": 2,
            "annotations_sha256": HASH,
            "image_list_sha256": OTHER_HASH,
        },
        "determinism": {
            "seed": 0,
            "metric_tolerance": 1e-12,
            "cpu_metric_tolerance": 1e-12,
            "cuda_metric_tolerance": 1e-6,
            "tensorrt_metric_tolerance": 1e-5,
        },
        "required_branches": ["o2m-nms", "nms-free"],
        "branches": {
            "o2m-nms": {
                "baseline_metrics": {
                    "map50_95": 0.200,
                    "map50": 0.325,
                    "ar_100": 0.379,
                },
                "absolute_floors": {
                    "map50_95": 0.190,
                    "map50": 0.310,
                    "ar_100": 0.360,
                },
                "max_regressions": {
                    "map50_95": 0.005,
                    "map50": 0.010,
                    "ar_100": 0.010,
                },
            },
            "nms-free": {
                "baseline_metrics": {
                    "map50_95": 0.096,
                    "map50": 0.140,
                },
                "absolute_floors": {
                    "map50_95": 0.090,
                    "map50": 0.130,
                },
                "max_regressions": {
                    "map50_95": 0.005,
                    "map50": 0.010,
                },
            },
        },
    }


def _metrics(**overrides) -> dict:
    metrics = {
        "map50_95": 0.200,
        "map50": 0.325,
        "map75": 0.190,
        "ap_small": 0.050,
        "ap_medium": 0.190,
        "ap_large": 0.310,
        "ar_100": 0.379,
    }
    metrics.update(overrides)
    return metrics


def _report(metrics: dict | None = None) -> dict:
    return {
        "schema_version": 1,
        "backend": "pytorch",
        "framework_commit": "abc123",
        "checkpoint": "checkpoint.pt",
        "checkpoint_sha256": HASH,
        "dataset": {
            "name": "coco-2017",
            "split": "val2017",
            "evaluated_images": 2,
            "annotations_sha256": HASH,
            "image_list_sha256": OTHER_HASH,
        },
        "environment": {
            "python": "3.11",
            "os": "linux",
            "torch": "2.6.0",
            "onnxruntime": "1.23.2",
        },
        "protocol": {
            "seed": 0,
            "release_eligible": True,
        },
        "branches": {
            "o2m-nms": {
                "metrics": metrics if metrics is not None else _metrics(),
            },
            "nms-free": {
                "metrics": _metrics(map50_95=0.096, map50=0.140),
            },
        },
    }


def test_accuracy_gate_accepts_valid_report() -> None:
    assert check_report(_report(), _config()) == []


def test_accuracy_gate_rejects_missing_dataset_hash() -> None:
    report = _report()
    del report["dataset"]["annotations_sha256"]

    failures = check_report(report, _config())

    assert any(failure.kind == "malformed_report" for failure in failures)
    assert any("annotations_sha256" in failure.message for failure in failures)


def test_accuracy_gate_rejects_unpinned_config_dataset_hashes() -> None:
    config = _config()
    config["dataset"]["annotations_sha256"] = None

    failures = check_report(_report(), config)

    assert any(failure.kind == "config" for failure in failures)
    assert any("annotations_sha256" in failure.message for failure in failures)


def test_accuracy_gate_rejects_mismatched_canonical_dataset_hash() -> None:
    report = _report()
    report["dataset"]["image_list_sha256"] = HASH

    failures = check_report(report, _config())

    assert any(failure.kind == "malformed_report" for failure in failures)
    assert any("image_list_sha256" in failure.message for failure in failures)


def test_accuracy_gate_requires_complete_configured_branch_set() -> None:
    report = _report()
    del report["branches"]["nms-free"]

    failures = check_report(report, _config())

    assert any(failure.kind == "malformed_report" for failure in failures)
    assert any("missing required branch" in failure.message for failure in failures)


def test_accuracy_gate_rejects_non_finite_metrics() -> None:
    failures = check_report(_report(_metrics(map50_95=float("nan"))), _config())

    assert any(failure.kind == "malformed_report" for failure in failures)
    assert any("non-finite" in failure.message for failure in failures)


def test_accuracy_gate_rejects_out_of_range_metrics() -> None:
    failures = check_report(_report(_metrics(map50=1.1)), _config())

    assert any(failure.kind == "malformed_report" for failure in failures)
    assert any("outside [0, 1]" in failure.message for failure in failures)


def test_accuracy_gate_rejects_missing_release_metadata() -> None:
    report = _report()
    del report["checkpoint_sha256"]
    report["protocol"]["release_eligible"] = False

    failures = check_report(report, _config())

    assert any("checkpoint_sha256" in failure.message for failure in failures)
    assert any("release_eligible" in failure.message for failure in failures)


def test_accuracy_gate_rejects_missing_backend_and_seed() -> None:
    report = _report()
    del report["backend"]
    del report["protocol"]["seed"]

    failures = check_report(report, _config())

    assert any("backend" in failure.message for failure in failures)
    assert any("protocol.seed" in failure.message for failure in failures)


def test_accuracy_gate_rejects_onnx_provider_fallback() -> None:
    report = _report()
    report["backend"] = "onnx"
    report["model"] = "model.onnx"
    report["metadata"] = "model.json"
    report["metadata_sha256"] = OTHER_HASH
    report["environment"]["requested_provider"] = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    report["environment"]["actual_provider"] = "CPUExecutionProvider"

    failures = check_report(report, _config())

    assert any(failure.kind == "malformed_report" for failure in failures)
    assert any("actual_provider" in failure.message for failure in failures)


def test_accuracy_gate_separates_absolute_floor_from_baseline_regression() -> None:
    floor_failures = check_report(_report(_metrics(map50_95=0.180)), _config())
    regression_failures = check_report(_report(_metrics(map50_95=0.194)), _config())

    assert any(failure.kind == "absolute_floor" for failure in floor_failures)
    assert any(failure.kind == "baseline_regression" for failure in floor_failures)
    assert not any(failure.kind == "absolute_floor" for failure in regression_failures)
    assert any(failure.kind == "baseline_regression" for failure in regression_failures)


def test_repeated_report_comparison_flags_metric_drift() -> None:
    first = _report()
    second = _report(_metrics(map50=0.3250002))

    failures = compare_repeated_reports(first, second, tolerance=1e-7)

    assert len(failures) == 1
    assert failures[0].kind == "determinism"
    assert "map50" in failures[0].message


def test_repeated_report_comparison_rejects_non_finite_metric() -> None:
    failures = compare_repeated_reports(
        _report(),
        _report(_metrics(map50_95=float("nan"))),
        tolerance=1e-7,
    )

    assert any(failure.kind == "determinism" for failure in failures)
    assert any("non-finite" in failure.message for failure in failures)


def test_determinism_tolerance_uses_actual_onnx_provider() -> None:
    report = _report()
    report["backend"] = "onnx"
    report["environment"]["actual_provider"] = "CUDAExecutionProvider"

    assert determinism_tolerance(report, _config()) == 1e-6
