from scripts.check_vision_v8_coco_report import (
    check_report,
    compare_repeated_reports,
)


def _config() -> dict:
    return {
        "dataset": {
            "name": "coco-2017",
            "split": "val2017",
            "required_image_count": 2,
            "annotations_sha256": "annotations",
            "image_list_sha256": "image-list",
        },
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
            }
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
        "framework_commit": "abc123",
        "dataset": {
            "name": "coco-2017",
            "split": "val2017",
            "evaluated_images": 2,
            "annotations_sha256": "annotations",
            "image_list_sha256": "image-list",
        },
        "environment": {
            "python": "3.11",
            "os": "linux",
            "torch": "2.6.0",
            "onnxruntime": "1.23.2",
        },
        "branches": {
            "o2m-nms": {
                "metrics": metrics if metrics is not None else _metrics(),
            }
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
