from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.build_onnx_release import (
    MANIFEST_NAME,
    ReleaseConfig,
    ReleaseError,
    artifact_entry,
    build_manifest,
    config_from_mapping,
    load_config,
    output_contract,
    render_release_notes,
    sha256_file,
    toolchain_mismatches,
    verify_manifest,
)

CONFIG_PATH = Path("docs/onnx/release.json")

SIDECAR = {
    "architecture_version": 8,
    "image_size": 640,
    "num_classes": 80,
    "num_cells": 34000,
    "regression_width": 68,
    "reg_max": 16,
    "grid_sizes": [160, 80, 40, 20],
    "branch": "o2m",
    "requires_nms": True,
    "output_semantics": "raw_ltrb_dfl_and_quality_class_logits",
}


def config_mapping(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "checkpoint_repo": "AETHORIA-AI/TR-HASH-Vision-v8-2M-COCO-SFT",
        "checkpoint_revision": "f3b3e659612e543ca9ff91892c0662d38dc1a1d6",
        "opset": 17,
        "parity_num_tests": 5,
        "toolchain": {"torch": "2.13.0", "onnx": "1.21.0", "onnxruntime": "1.23.2"},
        "branches": [
            {
                "branch": "o2m",
                "checkpoint_subdir": None,
                "stem": "tr_hash_v8_o2m",
                "post_processing": "decode plus NMS",
            },
            {
                "branch": "nms-free",
                "checkpoint_subdir": "best_nms_free",
                "stem": "tr_hash_v8_nms_free",
                "post_processing": "decode plus confidence filtering",
            },
        ],
    }
    values.update(overrides)
    return values


def written_release(directory: Path) -> dict[str, Any]:
    """Write two fake artifacts and return their verified manifest."""

    config = config_from_mapping(config_mapping())
    artifacts = []
    for spec in config.branches:
        model = directory / spec.model_name
        model.write_bytes(spec.branch.encode() * 64)
        sidecar = directory / spec.sidecar_name
        sidecar.write_text(json.dumps({**SIDECAR, "branch": spec.branch}))
        artifacts.append(
            artifact_entry(
                model,
                kind="model",
                branch=spec.branch,
                requires_nms=spec.branch == "o2m",
                post_processing=spec.post_processing,
                contract=output_contract(SIDECAR),
            )
        )
        artifacts.append(artifact_entry(sidecar, kind="metadata", branch=spec.branch))

    return build_manifest(
        config,
        artifacts,
        commit="0" * 40,
        toolchain=config.toolchain,
    )


def test_committed_release_config_is_valid_and_pins_both_branches() -> None:
    config = load_config(CONFIG_PATH)

    assert isinstance(config, ReleaseConfig)
    assert {spec.branch for spec in config.branches} == {"o2m", "nms-free"}
    assert config.opset == 17
    assert set(config.toolchain) == {"torch", "onnx", "onnxruntime"}
    assert config.toolchain["onnxruntime"] == "1.23.2"
    assert config.quantization is not None
    assert config.quantization.enabled_precisions == ("fp16", "int8")
    assert config.quantization.calibration_manifest == Path(
        "artifacts/vision_v8_quantized_eval/calibration.json"
    )
    assert config.quantization.thresholds == Path("configs/vision_v8_quantization_thresholds.json")
    assert config.quantization.accuracy_report == Path(
        "artifacts/vision_v8_quantized_eval/accuracy.json"
    )
    assert config.quantization.accuracy_markdown == Path(
        "artifacts/vision_v8_quantized_eval/accuracy.md"
    )
    assert config.quantization.provider_gates == (
        ("CPUExecutionProvider", "fp32"),
        ("CUDAExecutionProvider", "fp16"),
        ("CPUExecutionProvider", "int8"),
    )
    # Five seeds underestimate the observed maxima (see the validation report),
    # so a release gate has to be deeper than the development default.
    assert config.parity_num_tests >= 20


def test_release_workflows_share_quantized_evidence_artifact_contract() -> None:
    release_workflow = Path(".github/workflows/onnx-release.yml").read_text(encoding="utf-8")
    coco_workflow = Path(".github/workflows/vision-v8-coco-accuracy.yml").read_text(
        encoding="utf-8"
    )

    assert "--name vision-v8-quantized-release-inputs" in release_workflow
    assert "onnxruntime-gpu" in release_workflow
    assert "runs-on: ${{ inputs.runner || 'self-hosted' }}" in release_workflow
    assert "name: vision-v8-quantized-release-inputs" in coco_workflow
    assert "calibration.json" in coco_workflow
    assert "calibration_images/**" in coco_workflow
    assert "accuracy.json" in coco_workflow
    assert "accuracy.md" in coco_workflow


def test_release_builder_runs_by_documented_file_path(tmp_path: Path) -> None:
    config_path = tmp_path / "release.json"
    config_path.write_text(
        json.dumps(
            config_mapping(
                quantization={
                    "enabled_precisions": ["int8"],
                    "calibration_manifest": str(tmp_path / "missing-calibration.json"),
                    "thresholds": "configs/vision_v8_quantization_thresholds.json",
                    "accuracy_report": str(tmp_path / "missing-accuracy.json"),
                    "accuracy_markdown": str(tmp_path / "missing-accuracy.md"),
                }
            )
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_onnx_release.py",
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "dist"),
            "--allow-toolchain-drift",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "No module named 'scripts." not in result.stderr


def test_a_moving_checkpoint_ref_is_rejected() -> None:
    with pytest.raises(ReleaseError, match="40-character commit sha"):
        config_from_mapping(config_mapping(checkpoint_revision="main"))


def test_a_duplicated_branch_is_rejected() -> None:
    branches = config_mapping()["branches"]
    with pytest.raises(ReleaseError, match="branch twice"):
        config_from_mapping(config_mapping(branches=[branches[0], branches[0]]))


def test_an_unpinned_toolchain_is_rejected() -> None:
    with pytest.raises(ReleaseError, match="pin a toolchain"):
        config_from_mapping(config_mapping(toolchain={}))


def test_unsupported_quantized_precision_is_rejected() -> None:
    with pytest.raises(ReleaseError, match="unsupported quantized precisions"):
        config_from_mapping(
            config_mapping(
                quantization={
                    "enabled_precisions": ["fp8"],
                    "calibration_manifest": "calibration.json",
                    "thresholds": "thresholds.json",
                    "accuracy_report": "accuracy.json",
                    "accuracy_markdown": "accuracy.md",
                }
            )
        )


def test_toolchain_drift_is_reported_per_package() -> None:
    pinned = {"torch": "2.13.0", "onnx": "1.21.0"}

    assert toolchain_mismatches(pinned, {"torch": "2.13.0", "onnx": "1.21.0"}) == []
    assert toolchain_mismatches(pinned, {"torch": "2.6.0", "onnx": "1.21.0"}) == [
        "torch: pinned 2.13.0, installed 2.6.0"
    ]
    assert toolchain_mismatches(pinned, {"torch": "2.13.0"}) == [
        "onnx: pinned 1.21.0, installed missing"
    ]


def test_output_contract_derives_the_prediction_width_from_the_sidecar() -> None:
    contract = output_contract(SIDECAR)

    assert contract["input_shape"] == [1, 3, 640, 640]
    assert contract["output_shape"] == [1, 34000, 148]
    assert contract["dtype"] == "float32"


def test_manifest_carries_every_field_the_release_must_document(
    tmp_path: Path,
) -> None:
    manifest = written_release(tmp_path)

    assert manifest["checkpoint_revision"] == ("f3b3e659612e543ca9ff91892c0662d38dc1a1d6")
    assert manifest["framework_commit"] == "0" * 40
    assert manifest["opset"] == 17
    assert manifest["toolchain"]["torch"] == "2.13.0"
    assert len(manifest["artifacts"]) == 4
    for artifact in manifest["artifacts"]:
        assert artifact["size_bytes"] > 0
        assert len(artifact["sha256"]) == 64


def test_verification_passes_on_an_untouched_release(tmp_path: Path) -> None:
    manifest = written_release(tmp_path)

    assert verify_manifest(manifest, tmp_path) == []


def test_verification_catches_an_altered_binary(tmp_path: Path) -> None:
    manifest = written_release(tmp_path)
    target = tmp_path / "tr_hash_v8_o2m.onnx"
    original = target.read_bytes()
    # Same length, different content: only the digest can catch this.
    target.write_bytes(b"x" + original[1:])

    problems = verify_manifest(manifest, tmp_path)

    assert len(problems) == 1
    assert "tr_hash_v8_o2m.onnx" in problems[0]
    assert "sha256" in problems[0]


def test_verification_catches_a_truncated_or_missing_artifact(tmp_path: Path) -> None:
    manifest = written_release(tmp_path)
    (tmp_path / "tr_hash_v8_o2m.onnx").write_bytes(b"short")
    (tmp_path / "tr_hash_v8_nms_free.json").unlink()

    problems = "\n".join(verify_manifest(manifest, tmp_path))

    assert "size" in problems
    assert "missing" in problems


def test_verification_binds_the_manifest_to_the_publishing_commit(
    tmp_path: Path,
) -> None:
    manifest = written_release(tmp_path)

    assert verify_manifest(manifest, tmp_path, expect_commit="0" * 40) == []

    problems = verify_manifest(manifest, tmp_path, expect_commit="1" * 40)

    assert len(problems) == 1
    assert "framework_commit" in problems[0]


def test_release_notes_distinguish_the_two_branches(tmp_path: Path) -> None:
    manifest = written_release(tmp_path)

    notes = render_release_notes(manifest)

    assert "decode plus NMS" in notes
    assert "decode plus confidence filtering" in notes
    assert "tr_hash_v8_o2m.onnx" in notes
    assert "tr_hash_v8_nms_free.onnx" in notes
    # Sidecars are published but are not models: they must not add table rows.
    assert notes.count("| o2m |") == 1
    assert MANIFEST_NAME in notes


def test_sha256_matches_a_known_digest(tmp_path: Path) -> None:
    target = tmp_path / "empty.bin"
    target.write_bytes(b"")

    assert sha256_file(target) == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
