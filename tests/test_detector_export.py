from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from complexity.generative.detection import (
    TRHashDetectorConfig,
    TRHashObjectDetector,
    load_detector_checkpoint,
)
from complexity.generative.detection.exporting import RawDetectorExport
from scripts.check_onnx_parity import (
    DECODED_BOX_GATE,
    DECODED_SCORE_GATE,
    DEFAULT_PARITY_TOLERANCE,
    RAW_GATE,
    V8_PARITY_TOLERANCES,
    V8_TOLERANCES,
    ParityTolerances,
    branch_from_sidecar,
    calibrated_parity_tolerance,
    calibrated_tolerances,
    check_parity,
    decode_context,
    evaluate_gates,
)
from scripts.export_onnx import export_onnx


def tiny_config(*, end_to_end: bool) -> TRHashDetectorConfig:
    return TRHashDetectorConfig(
        architecture_version=8,
        image_size=32,
        patch_size=8,
        vision_hidden_size=16,
        vision_layers=3,
        vision_heads=4,
        vision_num_experts=2,
        vision_top_k=1,
        vision_expert_width=8,
        vision_stage_depths=(1, 1, 1),
        vision_window_size=2,
        vision_precision="fp32",
        num_classes=3,
        reg_max=4,
        head_hidden_size=16,
        end_to_end=end_to_end,
    )


def write_checkpoint(path: Path, *, end_to_end: bool) -> TRHashObjectDetector:
    torch.manual_seed(7)
    model = TRHashObjectDetector(tiny_config(end_to_end=end_to_end)).eval()
    path.mkdir()
    (path / "config.json").write_text(json.dumps(model.config.to_dict()))
    save_file(model.state_dict(), str(path / "model.safetensors"))
    return model


def test_public_loader_accepts_v8_export_architecture(tmp_path: Path) -> None:
    checkpoint = tmp_path / "detector-v8"
    expected = write_checkpoint(checkpoint, end_to_end=False)
    loaded = load_detector_checkpoint(checkpoint, device="cpu").eval()

    assert loaded.config.architecture_version == 8
    values = torch.randn(1, 3, 32, 32)
    torch.testing.assert_close(
        RawDetectorExport(loaded, "o2m")(values),
        RawDetectorExport(expected, "o2m")(values),
    )


def test_auto_export_selects_the_production_branch() -> None:
    values = torch.randn(2, 3, 32, 32)
    end_to_end = TRHashObjectDetector(tiny_config(end_to_end=True)).eval()
    one_to_many, one_to_one = end_to_end.forward_branches(values)

    assert one_to_one is not None
    assert RawDetectorExport(end_to_end).branch == "nms-free"
    torch.testing.assert_close(RawDetectorExport(end_to_end)(values), one_to_one)
    torch.testing.assert_close(RawDetectorExport(end_to_end, "o2m")(values), one_to_many)

    classic = TRHashObjectDetector(tiny_config(end_to_end=False)).eval()
    assert RawDetectorExport(classic).branch == "o2m"
    with pytest.raises(ValueError, match="end_to_end=True"):
        RawDetectorExport(classic, "nms-free")


def sidecar_mapping(config: TRHashDetectorConfig, branch: str) -> dict:
    """Mirror the metadata sidecar written by scripts/export_onnx.py."""

    return {
        "architecture_version": config.architecture_version,
        "image_size": config.image_size,
        "num_classes": config.num_classes,
        "num_cells": config.num_cells,
        "regression_width": config.regression_width,
        "reg_max": config.reg_max,
        "scale_factors": list(config.scale_factors),
        "grid_sizes": list(config.grid_sizes),
        "p2_head": config.p2_head,
        "branch": branch,
        "requires_nms": branch == "o2m",
        "output_semantics": "raw_ltrb_dfl_and_quality_class_logits",
    }


@pytest.mark.parametrize(
    ("branch", "raw", "decoded_box", "decoded_score"),
    (
        ("o2m", 6e-3, 1.3e-4, 8e-5),
        ("nms-free", 1e-2, 1.3e-4, 4e-5),
    ),
)
def test_v8_exports_use_branch_calibrated_parity_tolerances(
    branch: str,
    raw: float,
    decoded_box: float,
    decoded_score: float,
) -> None:
    metadata = {"architecture_version": 8, "branch": branch}

    assert V8_PARITY_TOLERANCES[branch] == raw
    assert calibrated_parity_tolerance(metadata, branch) == raw
    assert V8_TOLERANCES[branch] == ParityTolerances(
        raw=raw,
        decoded_box=decoded_box,
        decoded_score=decoded_score,
    )
    assert calibrated_tolerances(metadata, branch) == V8_TOLERANCES[branch]


def test_legacy_or_unlabelled_exports_keep_strict_parity_tolerance() -> None:
    assert calibrated_parity_tolerance({}, "o2m") == DEFAULT_PARITY_TOLERANCE
    assert (
        calibrated_parity_tolerance({"architecture_version": 7}, "nms-free")
        == DEFAULT_PARITY_TOLERANCE
    )


def test_legacy_exports_disable_the_decoded_gates() -> None:
    legacy = calibrated_tolerances({"architecture_version": 7}, "o2m")

    assert legacy.raw == DEFAULT_PARITY_TOLERANCE
    assert legacy.decoded_box is None
    assert legacy.decoded_score is None


@pytest.mark.parametrize(
    "metadata",
    (
        {},
        {"architecture_version": 7, "branch": "o2m"},
        {"architecture_version": 8, "branch": "o2m"},  # decode fields missing
    ),
)
def test_decode_context_is_unavailable_without_full_v8_metadata(metadata: dict) -> None:
    assert decode_context(metadata) is None


def test_decoded_gates_are_skipped_rather_than_failed_for_legacy_exports() -> None:
    predictions = np.zeros((1, 4, 8), dtype=np.float32)
    drifted = predictions + 5e-5

    results = evaluate_gates(
        predictions,
        drifted,
        calibrated_tolerances({}, "auto"),
        decode_context({}),
    )

    assert [result.name for result in results] == [RAW_GATE]
    assert results[0].passed


def test_decoded_box_gate_catches_drift_the_raw_gate_tolerates() -> None:
    """Amplification case: softmax turns a coherent logit tilt into box motion.

    Tilting the DFL logits by ``epsilon * bin_index`` shifts the decoded
    expectation by about ``epsilon * Var(bin)``, which is then scaled by the
    stride. The raw drift stays under its (deliberately coarse) threshold while
    the decoded box drift blows past its own.
    """

    config = tiny_config(end_to_end=False)
    metadata = sidecar_mapping(config, "o2m")
    context = decode_context(metadata)
    assert context is not None

    epsilon = 1e-3
    bins = config.dfl_bins
    tilt = epsilon * np.arange(bins, dtype=np.float32)

    baseline = np.zeros((1, config.num_cells, config.prediction_width), dtype=np.float32)
    drifted = baseline.copy()
    # Same tilt on each of the four LTRB distributions; class logits untouched.
    drifted[..., : config.regression_width] += np.tile(tilt, 4)

    tolerances = ParityTolerances(raw=5e-3, decoded_box=1.3e-4, decoded_score=4e-5)
    results = {
        result.name: result
        for result in evaluate_gates(baseline, drifted, tolerances, context)
    }

    assert results[RAW_GATE].passed, "raw drift must stay inside its tolerance"
    assert not results[DECODED_BOX_GATE].passed, "decoded box drift must be caught"
    assert results[DECODED_SCORE_GATE].passed, "gates must fail independently"
    assert results[RAW_GATE].max_difference == pytest.approx(epsilon * (bins - 1))


def test_sidecar_less_auto_branch_still_defers_to_model_resolution() -> None:
    assert branch_from_sidecar({}, "auto") == "auto"


@pytest.mark.parametrize("branch", ("nms-free", "o2m"))
def test_dynamic_onnx_export_matches_the_selected_branch(tmp_path: Path, branch: str) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    checkpoint = tmp_path / "checkpoint"
    write_checkpoint(checkpoint, end_to_end=True)
    output = tmp_path / f"detector-{branch}.onnx"

    export_onnx(
        checkpoint,
        output,
        dynamic_batch=True,
        check=True,
        branch=branch,
    )

    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["architecture_version"] == 8
    assert metadata["branch"] == branch
    assert metadata["requires_nms"] is (branch == "o2m")
    # The sidecar must be rich enough to drive the decoded gates.
    assert decode_context(metadata) is not None
    # Strict raw threshold, calibrated decoded thresholds: all three gates run.
    assert check_parity(
        checkpoint,
        output,
        num_tests=1,
        batch_size=2,
        tolerance=1e-4,
    )
    assert check_parity(
        checkpoint,
        output,
        num_tests=1,
        batch_size=2,
        tolerance=1e-4,
        skip_decoded=True,
    )
