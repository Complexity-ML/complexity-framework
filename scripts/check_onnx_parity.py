"""Verify numerical parity between a detector checkpoint and its ONNX export.

Three independent gates are evaluated on the same deterministic inputs:

* ``raw`` compares the exported logits directly. It is a coarse export-integrity
  check: raw drift concentrates in fine-grid regression logits and its observed
  maximum is unstable across seeds.
* ``decoded-box`` compares normalized xyxy boxes after LTRB/DFL decode.
* ``decoded-score`` compares sigmoid quality-class scores.

The decoded gates describe deployment behaviour and carry the real guarantee.
They are only available when the ONNX sidecar exposes v8 decode metadata;
legacy exports keep their strict raw-only behaviour.

Usage:
    python scripts/check_onnx_parity.py CHECKPOINT model.onnx
    python scripts/check_onnx_parity.py CHECKPOINT model.onnx --num-tests 10
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

os.environ["COMPLEXITY_DISABLE_KERNELS"] = "1"

from complexity.deploy.onnx_detector.dfl import decode_dfl_boxes
from complexity.deploy.onnx_detector.grid import GridGeometry, generate_grid_geometry
from complexity.deploy.onnx_detector.metadata import (
    OnnxDetectorMetadata,
    metadata_from_mapping,
    validate_output_shape,
)
from complexity.generative.detection.exporting import ExportBranch, RawDetectorExport
from complexity.generative.detection.hub import load_detector_checkpoint

RAW_GATE = "raw"
DECODED_BOX_GATE = "decoded-box"
DECODED_SCORE_GATE = "decoded-score"

DEFAULT_PARITY_TOLERANCE = 1e-4


@dataclass(frozen=True)
class ParityTolerances:
    """Per-gate absolute tolerances; ``None`` disables a decoded gate."""

    raw: float
    decoded_box: float | None = None
    decoded_score: float | None = None


# Calibrated on AETHORIA-AI/TR-HASH-Vision-v8-2M-COCO-SFT over 50 deterministic
# seeds, at twice the observed maximum. See docs/onnx/tr_hash_v8_validation_report.md.
V8_TOLERANCES = {
    "o2m": ParityTolerances(raw=6e-3, decoded_box=1.3e-4, decoded_score=8e-5),
    "nms-free": ParityTolerances(raw=1e-2, decoded_box=1.3e-4, decoded_score=4e-5),
}

# Raw-only view kept for callers that just need the legacy scalar threshold.
V8_PARITY_TOLERANCES = {
    branch: tolerances.raw for branch, tolerances in V8_TOLERANCES.items()
}


@dataclass(frozen=True)
class GateResult:
    """Outcome of one parity gate on one test input."""

    name: str
    tolerance: float
    max_difference: float
    mean_difference: float

    @property
    def passed(self) -> bool:
        return self.max_difference <= self.tolerance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="PyTorch checkpoint directory")
    parser.add_argument("onnx_model", type=Path, help="Exported ONNX model path")
    parser.add_argument(
        "--branch",
        choices=("auto", "nms-free", "o2m"),
        default="auto",
        help="Prediction branch; auto reads the ONNX sidecar when available",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help=(
            "Max allowed absolute raw-logit difference. Defaults to calibrated "
            "v8 branch thresholds when ONNX metadata is available, otherwise 1e-4."
        ),
    )
    parser.add_argument(
        "--decoded-box-tolerance",
        type=float,
        default=None,
        help="Max allowed absolute difference on normalized decoded boxes",
    )
    parser.add_argument(
        "--decoded-score-tolerance",
        type=float,
        default=None,
        help="Max allowed absolute difference on sigmoid quality-class scores",
    )
    parser.add_argument(
        "--skip-decoded",
        action="store_true",
        help="Only run the raw-logit gate, even when decode metadata is available",
    )
    parser.add_argument(
        "--num-tests",
        type=int,
        default=5,
        help="Number of random inputs to test (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Input batch size, including for dynamic-batch exports",
    )
    return parser.parse_args()


def sidecar_metadata(onnx_path: Path) -> dict:
    metadata_path = onnx_path.with_suffix(".json")
    if not metadata_path.is_file():
        return {}
    return json.loads(metadata_path.read_text())


def branch_from_sidecar(metadata: dict, requested: ExportBranch) -> ExportBranch:
    if requested != "auto":
        return requested
    if "branch" not in metadata:
        return "auto"
    branch = metadata.get("branch", "auto")
    if branch not in {"nms-free", "o2m"}:
        raise ValueError(f"invalid export branch in ONNX metadata: {branch}")
    return branch


def calibrated_parity_tolerance(metadata: dict, branch: ExportBranch) -> float:
    """Return the default raw-logit parity tolerance for an exported model."""

    return calibrated_tolerances(metadata, branch).raw


def calibrated_tolerances(metadata: dict, branch: ExportBranch) -> ParityTolerances:
    """Return per-gate defaults, falling back to strict raw-only for legacy exports."""

    if metadata.get("architecture_version") == 8 and branch in V8_TOLERANCES:
        return V8_TOLERANCES[branch]
    return ParityTolerances(raw=DEFAULT_PARITY_TOLERANCE)


def decode_context(metadata: dict) -> tuple[OnnxDetectorMetadata, GridGeometry] | None:
    """Build decode metadata and grid geometry, or None when unavailable.

    Legacy sidecars predate the v8 decode contract, so a rejected mapping is a
    supported outcome rather than an error: the decoded gates are skipped.
    """

    if not metadata:
        return None
    try:
        resolved = metadata_from_mapping(metadata)
    except (KeyError, TypeError, ValueError):
        return None
    geometry = generate_grid_geometry(resolved.image_size, resolved.grid_sizes)
    return resolved, geometry


def decoded_outputs(
    predictions: np.ndarray,
    metadata: OnnxDetectorMetadata,
    geometry: GridGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized xyxy boxes and sigmoid quality-class scores."""

    validate_output_shape(metadata, predictions.shape)
    regression = predictions[..., : metadata.regression_width]
    class_logits = predictions[..., metadata.regression_width :]
    boxes = decode_dfl_boxes(regression, metadata, geometry)
    boxes_norm = boxes / np.float32(metadata.image_size)
    return boxes_norm, _sigmoid(class_logits)


def evaluate_gates(
    torch_predictions: np.ndarray,
    onnx_predictions: np.ndarray,
    tolerances: ParityTolerances,
    context: tuple[OnnxDetectorMetadata, GridGeometry] | None = None,
) -> list[GateResult]:
    """Evaluate every enabled gate; each one reports independently."""

    results = [_gate(RAW_GATE, tolerances.raw, torch_predictions, onnx_predictions)]
    if context is None:
        return results

    metadata, geometry = context
    torch_boxes, torch_scores = decoded_outputs(torch_predictions, metadata, geometry)
    onnx_boxes, onnx_scores = decoded_outputs(onnx_predictions, metadata, geometry)

    if tolerances.decoded_box is not None:
        results.append(
            _gate(DECODED_BOX_GATE, tolerances.decoded_box, torch_boxes, onnx_boxes)
        )
    if tolerances.decoded_score is not None:
        results.append(
            _gate(DECODED_SCORE_GATE, tolerances.decoded_score, torch_scores, onnx_scores)
        )
    return results


def check_parity(
    checkpoint_path: Path,
    onnx_path: Path,
    *,
    branch: ExportBranch = "auto",
    tolerance: float | None = None,
    decoded_box_tolerance: float | None = None,
    decoded_score_tolerance: float | None = None,
    skip_decoded: bool = False,
    num_tests: int = 5,
    batch_size: int = 1,
) -> bool:
    """Compare PyTorch and ONNX raw and decoded outputs on random inputs."""

    if num_tests <= 0 or batch_size <= 0:
        raise ValueError("num_tests and batch_size must be positive")
    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime not installed. Run: pip install onnxruntime")
        return False

    print(f"Loading PyTorch model: {checkpoint_path}")
    detector = load_detector_checkpoint(checkpoint_path, device="cpu")
    metadata = sidecar_metadata(onnx_path)
    resolved_branch = branch_from_sidecar(metadata, branch)
    tolerances = _resolve_tolerances(
        metadata,
        resolved_branch,
        tolerance,
        decoded_box_tolerance,
        decoded_score_tolerance,
    )
    context = None if skip_decoded else decode_context(metadata)

    export_model = RawDetectorExport(detector, resolved_branch).eval()
    image_size = detector.config.image_size
    print(f"Prediction branch: {export_model.branch}")
    if context is None:
        print(
            "Decoded gates: skipped "
            f"({'--skip-decoded' if skip_decoded else 'no v8 decode metadata in sidecar'})"
        )

    print(f"Loading ONNX model: {onnx_path}")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    worst: dict[str, GateResult] = {}
    for test_index in range(num_tests):
        torch.manual_seed(test_index)
        values = torch.randn(batch_size, 3, image_size, image_size)
        with torch.no_grad():
            pytorch_output = export_model(values).numpy()
        onnx_output = session.run([output_name], {input_name: values.numpy()})[0]

        results = evaluate_gates(pytorch_output, onnx_output, tolerances, context)
        _print_test(test_index, num_tests, results)
        for result in results:
            current = worst.get(result.name)
            if current is None or result.max_difference > current.max_difference:
                worst[result.name] = result

    return _print_summary(export_model.branch, worst)


def main() -> None:
    args = parse_args()
    success = check_parity(
        args.checkpoint,
        args.onnx_model,
        branch=args.branch,
        tolerance=args.tolerance,
        decoded_box_tolerance=args.decoded_box_tolerance,
        decoded_score_tolerance=args.decoded_score_tolerance,
        skip_decoded=args.skip_decoded,
        num_tests=args.num_tests,
        batch_size=args.batch_size,
    )
    raise SystemExit(0 if success else 1)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _gate(
    name: str,
    tolerance: float,
    expected: np.ndarray,
    actual: np.ndarray,
) -> GateResult:
    difference = np.abs(expected - actual)
    return GateResult(
        name=name,
        tolerance=tolerance,
        max_difference=float(difference.max()),
        mean_difference=float(difference.mean()),
    )


def _resolve_tolerances(
    metadata: dict,
    branch: ExportBranch,
    raw_override: float | None,
    box_override: float | None,
    score_override: float | None,
) -> ParityTolerances:
    tolerances = calibrated_tolerances(metadata, branch)
    if raw_override is not None:
        tolerances = replace(tolerances, raw=raw_override)
    if box_override is not None:
        tolerances = replace(tolerances, decoded_box=box_override)
    if score_override is not None:
        tolerances = replace(tolerances, decoded_score=score_override)
    return tolerances


def _print_test(test_index: int, num_tests: int, results: list[GateResult]) -> None:
    label = f"  Test {test_index + 1}/{num_tests}"
    padding = " " * len(label)
    for position, result in enumerate(results):
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{label if position == 0 else padding}  {result.name:<13} "
            f"max={result.max_difference:.2e} mean={result.mean_difference:.2e} [{status}]"
        )


def _print_summary(branch: str, worst: dict[str, GateResult]) -> bool:
    failed = [result.name for result in worst.values() if not result.passed]
    summary = "FAILED" if failed else "PASSED"
    print(f"\nParity {summary}: branch={branch}")
    for result in worst.values():
        status = "PASS" if result.passed else "FAIL"
        print(
            f"  {result.name:<13} tol={result.tolerance:.2e} "
            f"worst_max={result.max_difference:.2e} [{status}]"
        )
    if failed:
        print(f"  failing gates: {', '.join(failed)}")
    return not failed


if __name__ == "__main__":
    main()
