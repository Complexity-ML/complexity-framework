from scripts.check_onnx_quantized_artifacts import check_quantized_accuracy_report


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

    assert failures == ["candidate branch nms-free does not match FP32 reference branch o2m-nms"]
