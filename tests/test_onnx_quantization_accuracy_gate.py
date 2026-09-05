from scripts.check_onnx_quantized_artifacts import (
    check_quantized_accuracy_report,
    check_quantized_parity_report,
)


def test_quantized_accuracy_fails_when_map_drop_exceeds_precision_threshold() -> None:
    report = {
        "reference": {
            "precision": "fp32",
            "branch": "o2m-nms",
            "metrics": {"map50_95": 0.2, "map50": 0.32},
        },
        "candidate": {
            "precision": "int8",
            "branch": "o2m-nms",
            "metrics": {"map50_95": 0.17, "map50": 0.31},
        },
    }
    thresholds = {
        "precisions": {
            "int8": {"max_map50_95_drop": 0.02, "max_map50_drop": 0.03}
        }
    }

    failures = check_quantized_accuracy_report(report, thresholds)

    assert any("map50_95" in failure for failure in failures)
    assert not any("map50=" in failure for failure in failures)


def test_quantized_accuracy_accepts_candidate_within_threshold() -> None:
    report = {
        "reference": {
            "precision": "fp32",
            "branch": "nms-free",
            "metrics": {"map50_95": 0.1, "map50": 0.14},
        },
        "candidate": {
            "precision": "fp16",
            "branch": "nms-free",
            "metrics": {"map50_95": 0.098, "map50": 0.135},
        },
    }
    thresholds = {
        "precisions": {
            "fp16": {"max_map50_95_drop": 0.005, "max_map50_drop": 0.01}
        }
    }

    assert check_quantized_accuracy_report(report, thresholds) == []


def test_quantized_accuracy_rejects_branch_mismatch() -> None:
    report = {
        "reference": {
            "precision": "fp32",
            "branch": "o2m-nms",
            "metrics": {"map50_95": 0.2, "map50": 0.32},
        },
        "candidate": {
            "precision": "fp16",
            "branch": "nms-free",
            "metrics": {"map50_95": 0.2, "map50": 0.32},
        },
    }
    thresholds = {
        "precisions": {
            "fp16": {"max_map50_95_drop": 0.005, "max_map50_drop": 0.01}
        }
    }

    failures = check_quantized_accuracy_report(report, thresholds)

    assert failures == [
        "candidate branch nms-free does not match FP32 reference branch o2m-nms"
    ]


def test_quantized_accuracy_accepts_evaluator_branch_report() -> None:
    report = {
        "reference_precision": "fp32",
        "candidate_precision": "fp16",
        "branches": {
            "o2m-nms": {
                "fp32": {
                    "precision": "fp32",
                    "branch": "o2m-nms",
                    "metrics": {"map50_95": 0.2, "map50": 0.32},
                },
                "fp16": {
                    "precision": "fp16",
                    "branch": "o2m-nms",
                    "metrics": {"map50_95": 0.198, "map50": 0.315},
                },
            }
        },
    }
    thresholds = {
        "precisions": {
            "fp16": {"max_map50_95_drop": 0.005, "max_map50_drop": 0.01}
        }
    }

    assert check_quantized_accuracy_report(report, thresholds) == []


def test_quantized_accuracy_rejects_empty_evaluator_branch_report() -> None:
    report = {
        "reference_precision": "fp32",
        "candidate_precision": "fp16",
        "branches": {},
    }
    thresholds = {
        "precisions": {
            "fp16": {"max_map50_95_drop": 0.005, "max_map50_drop": 0.01}
        }
    }

    failures = check_quantized_accuracy_report(report, thresholds)

    assert failures == ["quantized COCO report must contain branch comparisons"]


def test_quantized_accuracy_rejects_non_finite_metrics() -> None:
    report = {
        "reference": {
            "precision": "fp32",
            "branch": "o2m-nms",
            "metrics": {"map50_95": 0.2, "map50": 0.32},
        },
        "candidate": {
            "precision": "fp16",
            "branch": "o2m-nms",
            "metrics": {"map50_95": float("nan"), "map50": 0.32},
        },
    }
    thresholds = {
        "precisions": {
            "fp16": {"max_map50_95_drop": 0.005, "max_map50_drop": 0.01}
        }
    }

    failures = check_quantized_accuracy_report(report, thresholds)

    assert failures == ["non-finite metric map50_95 in FP32 or candidate report"]


def test_quantized_parity_consumes_raw_and_decoded_thresholds() -> None:
    report = {
        "precision": "int8",
        "branch": "o2m-nms",
        "max_raw_logit_abs_error": 0.13,
        "max_decoded_box_px_error": 3.0,
        "max_score_abs_error": 0.04,
    }
    thresholds = {
        "precisions": {
            "int8": {
                "max_raw_logit_abs_error": 0.12,
                "max_decoded_box_px_error": 4.0,
                "max_score_abs_error": 0.05,
            }
        }
    }

    failures = check_quantized_parity_report(report, thresholds)

    assert failures == [
        "int8 o2m-nms max_raw_logit_abs_error 0.130000 exceeds 0.120000"
    ]
