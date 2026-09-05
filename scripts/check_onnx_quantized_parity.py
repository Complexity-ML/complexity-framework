"""Gate FP32-vs-quantized Vision v8 ONNX raw and decoded parity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp16", "int8"), required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument(
        "--provider",
        action="append",
        default=None,
        help="Provider alias or ORT provider name. May be repeated or comma-separated.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def decoded_parity_metrics(
    reference_detections: Sequence[object],
    candidate_detections: Sequence[object],
) -> dict[str, float | int]:
    """Compare decoded detections in stable score order."""

    count = min(len(reference_detections), len(candidate_detections))
    if count == 0:
        return {
            "reference_detection_count": len(reference_detections),
            "candidate_detection_count": len(candidate_detections),
            "class_mismatch_count": int(
                len(reference_detections) != len(candidate_detections)
            ),
            "max_decoded_box_px_error": 0.0,
            "max_score_abs_error": 0.0,
        }

    box_errors = []
    score_errors = []
    class_mismatches = abs(len(reference_detections) - len(candidate_detections))
    for reference, candidate in zip(
        reference_detections[:count],
        candidate_detections[:count],
    ):
        box_errors.append(
            float(
                np.max(
                    np.abs(
                        np.asarray(reference.box_pixel, dtype=np.float32)
                        - np.asarray(candidate.box_pixel, dtype=np.float32)
                    )
                )
            )
        )
        score_errors.append(abs(float(reference.score) - float(candidate.score)))
        class_mismatches += int(reference.class_id != candidate.class_id)

    return {
        "reference_detection_count": len(reference_detections),
        "candidate_detection_count": len(candidate_detections),
        "class_mismatch_count": class_mismatches,
        "max_decoded_box_px_error": max(box_errors),
        "max_score_abs_error": max(score_errors),
    }


def build_parity_report(
    *,
    reference_model: Path,
    candidate_model: Path,
    metadata_path: Path,
    image_path: Path,
    precision: str,
    providers: Sequence[str],
) -> dict[str, object]:
    from complexity.deploy.onnx_detector import OnnxDetectorPipeline
    from complexity.deploy.onnx_detector.metadata import load_metadata
    from complexity.deploy.onnx_detector.preprocess import preprocess_image

    metadata = load_metadata(metadata_path)
    preprocessed = preprocess_image(image_path, metadata.image_size)

    reference_session = OnnxDetectorPipeline.create_session(
        reference_model,
        providers=providers,
        warmup_runs=0,
    ).open()
    candidate_session = OnnxDetectorPipeline.create_session(
        candidate_model,
        providers=providers,
        warmup_runs=0,
    ).open()

    reference_raw = reference_session.run(preprocessed.pixel_values)
    candidate_raw = candidate_session.run(preprocessed.pixel_values)
    reference_pipeline = OnnxDetectorPipeline(
        metadata=metadata,
        session=reference_session,
    )
    candidate_pipeline = OnnxDetectorPipeline(
        metadata=metadata,
        session=candidate_session,
    )
    reference_result = reference_pipeline.predict(image_path)
    candidate_result = candidate_pipeline.predict(image_path)

    return {
        "schema_version": 1,
        "branch": metadata.branch,
        "precision": precision,
        "provider_used": candidate_session.provider_used,
        "max_raw_logit_abs_error": float(np.max(np.abs(reference_raw - candidate_raw))),
        **decoded_parity_metrics(
            reference_result.detections,
            candidate_result.detections,
        ),
    }


def main() -> None:
    from scripts.check_onnx_quantized_artifacts import (
        check_provider_precision_supported,
        check_quantized_parity_report,
        load_quantization_thresholds,
    )
    from scripts.onnx_detect import provider_names

    args = parse_args()
    thresholds = load_quantization_thresholds(args.thresholds)
    providers = provider_names(args.provider)
    report = build_parity_report(
        reference_model=args.reference_model,
        candidate_model=args.candidate_model,
        metadata_path=args.metadata,
        image_path=args.image,
        precision=args.precision,
        providers=providers,
    )
    check_provider_precision_supported(
        str(report["provider_used"]),
        args.precision,
        thresholds,
    )
    failures = check_quantized_parity_report(report, thresholds)
    if int(report["class_mismatch_count"]) > 0:
        failures.append(
            f"{report['branch']} {args.precision} decoded class/count mismatch: "
            f"{report['class_mismatch_count']}"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        print("Quantized parity FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
