import pytest

from scripts.check_onnx_quantized_artifacts import (
    check_accuracy_artifact_bindings,
    check_quantized_accuracy_report,
    check_quantized_parity_report,
    evaluation_image_ids_from_report,
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
    thresholds = {"precisions": {"int8": {"max_map50_95_drop": 0.02, "max_map50_drop": 0.03}}}

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
    thresholds = {"precisions": {"fp16": {"max_map50_95_drop": 0.005, "max_map50_drop": 0.01}}}

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
    thresholds = {"precisions": {"fp16": {"max_map50_95_drop": 0.005, "max_map50_drop": 0.01}}}

    failures = check_quantized_accuracy_report(report, thresholds)

    assert failures == ["candidate branch nms-free does not match FP32 reference branch o2m-nms"]


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
    thresholds = {"precisions": {"fp16": {"max_map50_95_drop": 0.005, "max_map50_drop": 0.01}}}

    assert check_quantized_accuracy_report(report, thresholds) == []


def test_quantized_accuracy_checks_every_precision_in_branch_report() -> None:
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
                    "metrics": {"map50_95": 0.199, "map50": 0.319},
                },
                "int8": {
                    "precision": "int8",
                    "branch": "o2m-nms",
                    "metrics": {"map50_95": 0.01, "map50": 0.02},
                },
            }
        },
    }
    thresholds = {
        "precisions": {
            "fp16": {"max_map50_95_drop": 0.005, "max_map50_drop": 0.01},
            "int8": {"max_map50_95_drop": 0.02, "max_map50_drop": 0.03},
        }
    }

    failures = check_quantized_accuracy_report(report, thresholds)

    assert any("int8 o2m-nms map50_95" in failure for failure in failures)


def test_quantized_accuracy_requires_release_policy_precisions() -> None:
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
                    "metrics": {"map50_95": 0.199, "map50": 0.319},
                },
            }
        },
    }
    thresholds = {
        "release_policy": {"required_precisions": ["fp32", "fp16", "int8"]},
        "precisions": {
            "fp16": {"max_map50_95_drop": 0.005, "max_map50_drop": 0.01},
            "int8": {"max_map50_95_drop": 0.02, "max_map50_drop": 0.03},
        },
    }

    failures = check_quantized_accuracy_report(report, thresholds)

    assert failures == ["o2m-nms missing fp32 or int8 metrics"]


def test_quantized_accuracy_rejects_empty_evaluator_branch_report() -> None:
    report = {
        "reference_precision": "fp32",
        "candidate_precision": "fp16",
        "branches": {},
    }
    thresholds = {"precisions": {"fp16": {"max_map50_95_drop": 0.005, "max_map50_drop": 0.01}}}

    failures = check_quantized_accuracy_report(report, thresholds)

    assert failures == ["quantized COCO report must contain branch comparisons"]


def test_quantized_accuracy_requires_configured_branches() -> None:
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
    thresholds = {"precisions": {"fp16": {"max_map50_95_drop": 0.005, "max_map50_drop": 0.01}}}

    failures = check_quantized_accuracy_report(
        report,
        thresholds,
        required_branches=["o2m-nms", "nms-free"],
    )

    assert "quantized COCO report missing branch nms-free" in failures


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
    thresholds = {"precisions": {"fp16": {"max_map50_95_drop": 0.005, "max_map50_drop": 0.01}}}

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

    assert failures == ["int8 o2m-nms max_raw_logit_abs_error 0.130000 exceeds 0.120000"]


def test_accuracy_report_must_include_actual_evaluation_ids() -> None:
    assert evaluation_image_ids_from_report({"dataset": {"image_ids": [3, 2, 2]}}) == {2, 3}


def test_accuracy_report_rejects_missing_evaluation_ids() -> None:
    with pytest.raises(ValueError, match="evaluation image IDs"):
        evaluation_image_ids_from_report({"dataset": {"disjoint_from": "train2017"}})


def test_accuracy_report_artifact_hashes_must_match_generated_release_artifacts() -> None:
    report = {
        "branches": {
            "o2m-nms": {
                "fp32": {
                    "checkpoint_sha256": "a" * 64,
                    "metadata_sha256": "b" * 64,
                },
                "fp16": {
                    "checkpoint_sha256": "c" * 64,
                    "metadata_sha256": "d" * 64,
                },
            }
        }
    }
    generated = {
        "o2m-nms": {
            "fp32": {
                "checkpoint_sha256": "a" * 64,
                "metadata_sha256": "b" * 64,
            },
            "fp16": {
                "checkpoint_sha256": "e" * 64,
                "metadata_sha256": "d" * 64,
            },
        }
    }

    failures = check_accuracy_artifact_bindings(report, generated)

    assert failures == [
        "o2m-nms fp16 checkpoint_sha256 "
        "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc "
        "does not match generated "
        "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
    ]


def test_release_accuracy_gate_requires_generated_artifact_bindings() -> None:
    report = {
        "reference_precision": "fp32",
        "candidate_precision": "fp16",
        "branches": {
            "o2m-nms": {
                "fp32": {
                    "precision": "fp32",
                    "branch": "o2m-nms",
                    "checkpoint_sha256": "a" * 64,
                    "metadata_sha256": "b" * 64,
                    "metrics": {"map50_95": 0.2, "map50": 0.32},
                },
                "fp16": {
                    "precision": "fp16",
                    "branch": "o2m-nms",
                    "checkpoint_sha256": "c" * 64,
                    "metadata_sha256": "d" * 64,
                    "metrics": {"map50_95": 0.199, "map50": 0.319},
                },
            }
        },
    }
    thresholds = {
        "release_policy": {
            "required_precisions": ["fp32", "fp16"],
            "require_artifact_bindings": True,
        },
        "precisions": {"fp16": {"max_map50_95_drop": 0.005, "max_map50_drop": 0.01}},
    }

    failures = check_quantized_accuracy_report(report, thresholds)

    assert failures == ["generated artifact bindings are required by release policy"]


def test_release_accuracy_gate_rejects_bogus_hashes_against_generated_artifacts() -> None:
    report = {
        "reference_precision": "fp32",
        "candidate_precision": "fp16",
        "branches": {
            "o2m-nms": {
                "fp32": {
                    "precision": "fp32",
                    "branch": "o2m-nms",
                    "checkpoint_sha256": "a" * 64,
                    "metadata_sha256": "b" * 64,
                    "metrics": {"map50_95": 0.2, "map50": 0.32},
                },
                "fp16": {
                    "precision": "fp16",
                    "branch": "o2m-nms",
                    "checkpoint_sha256": "c" * 64,
                    "metadata_sha256": "d" * 64,
                    "metrics": {"map50_95": 0.199, "map50": 0.319},
                },
            }
        },
    }
    thresholds = {
        "release_policy": {
            "required_precisions": ["fp32", "fp16"],
            "require_artifact_bindings": True,
        },
        "precisions": {"fp16": {"max_map50_95_drop": 0.005, "max_map50_drop": 0.01}},
    }
    generated = {
        "o2m-nms": {
            "fp32": {
                "checkpoint_sha256": "a" * 64,
                "metadata_sha256": "b" * 64,
            },
            "fp16": {
                "checkpoint_sha256": "e" * 64,
                "metadata_sha256": "d" * 64,
            },
        }
    }

    failures = check_quantized_accuracy_report(
        report,
        thresholds,
        expected_artifacts=generated,
    )

    assert any("o2m-nms fp16 checkpoint_sha256" in failure for failure in failures)
