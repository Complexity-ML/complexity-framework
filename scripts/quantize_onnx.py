"""Create reproducible FP16 and INT8 Vision v8 ONNX artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

Precision = Literal["fp32", "fp16", "int8"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32-model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp16", "int8"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sidecar",
        type=Path,
        help="Quantization provenance sidecar. Defaults to <output>.quantization.json.",
    )
    parser.add_argument(
        "--detector-metadata-output",
        type=Path,
        help=("Detector metadata sidecar copied from --metadata. Defaults to <output>.json."),
    )
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument("--checkpoint-revision", default="unknown")
    parser.add_argument(
        "--keep-fp32-op-type",
        action="append",
        default=[],
        help="FP16 conversion allowlist; may be repeated",
    )
    parser.add_argument("--disable-shape-infer", action="store_true")
    parser.add_argument("--repeat-output", type=Path)
    parser.add_argument("--require-identical-hash", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def framework_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def package_version(module_name: str) -> str | None:
    try:
        from importlib import metadata

        return metadata.version(module_name)
    except metadata.PackageNotFoundError:
        return None


def toolchain_versions() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "os": platform.platform(),
        "onnx": package_version("onnx"),
        "onnxruntime": package_version("onnxruntime"),
        "onnxconverter-common": package_version("onnxconverter-common"),
        "numpy": np.__version__,
    }


def assert_identical_artifact_hashes(first_sha256: str, second_sha256: str) -> None:
    if first_sha256 != second_sha256:
        raise ValueError(
            "quantization is not deterministic: "
            f"first SHA-256 {first_sha256} differs from repeat SHA-256 {second_sha256}"
        )


def write_quantization_sidecar(
    path: Path,
    *,
    precision: Precision,
    source_model_sha256: str,
    output_model_sha256: str,
    framework_commit: str,
    checkpoint_revision: str,
    settings: Mapping[str, Any],
    toolchain: Mapping[str, Any],
) -> None:
    payload = {
        "schema_version": 1,
        "artifact_type": "vision_v8_quantized_onnx",
        "precision": precision,
        "framework_commit": framework_commit,
        "checkpoint_revision": checkpoint_revision,
        "source_model_sha256": source_model_sha256,
        "output_model_sha256": output_model_sha256,
        "settings": dict(settings),
        "toolchain": dict(toolchain),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def default_quantization_sidecar(output_model: Path) -> Path:
    return output_model.with_name(f"{output_model.stem}.quantization.json")


def default_detector_metadata_output(output_model: Path) -> Path:
    return output_model.with_suffix(".json")


def copy_detector_metadata(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def quantize_fp32(input_model: Path, output_model: Path) -> None:
    output_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(input_model, output_model)


def quantize_fp16(
    input_model: Path,
    output_model: Path,
    *,
    keep_fp32_op_types: Sequence[str] = (),
    disable_shape_infer: bool = False,
) -> None:
    try:
        import onnx
        from onnxruntime.transformers import float16
    except ImportError as error:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "FP16 quantization requires ONNX Runtime transformer tools; "
            "install the export/quantization extra."
        ) from error

    output_model.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(input_model))
    converted = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=disable_shape_infer,
        op_block_list=list(keep_fp32_op_types),
    )
    onnx.save(converted, str(output_model))


class _CalibrationReader:
    def __init__(
        self,
        *,
        image_paths: Sequence[Path],
        metadata_path: Path,
        batch_size: int = 1,
        input_name: str = "pixel_values",
    ) -> None:
        from complexity.deploy.onnx_detector.metadata import load_metadata
        from complexity.deploy.onnx_detector.preprocess import preprocess_image

        metadata = load_metadata(metadata_path)
        self._input_name = input_name
        if batch_size <= 0:
            raise ValueError("calibration batch_size must be positive")
        self._batch_size = batch_size
        self._items = [
            preprocess_image(path, metadata.image_size).pixel_values for path in image_paths
        ]
        self._index = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._index >= len(self._items):
            return None
        batch = self._items[self._index : self._index + self._batch_size]
        self._index += self._batch_size
        return {self._input_name: np.concatenate(batch, axis=0)}


def _calibration_method(name: str) -> Any:
    from onnxruntime.quantization import CalibrationMethod

    normalized = name.lower()
    if normalized == "minmax":
        return CalibrationMethod.MinMax
    if normalized in {"entropy", "kl"}:
        return CalibrationMethod.Entropy
    if normalized == "percentile":
        return CalibrationMethod.Percentile
    raise ValueError(f"unsupported INT8 calibration method: {name}")


def _quant_type(name: str) -> Any:
    from onnxruntime.quantization import QuantType

    normalized = name.lower()
    if normalized == "qint8":
        return QuantType.QInt8
    if normalized == "quint8":
        return QuantType.QUInt8
    raise ValueError(f"unsupported ONNX Runtime quant type: {name}")


def _calibration_paths(manifest: Mapping[str, Any]) -> list[Path]:
    raw_paths = manifest.get("images", [])
    if not isinstance(raw_paths, Sequence) or isinstance(raw_paths, (str, bytes)):
        raise ValueError("calibration manifest images must be a sequence of paths")
    return [Path(str(path)) for path in sorted(raw_paths, key=str)]


def quantize_int8(
    input_model: Path,
    output_model: Path,
    *,
    metadata_path: Path,
    calibration_manifest: Mapping[str, Any],
) -> None:
    try:
        from onnxruntime.quantization import QuantFormat, quantize_static
    except ImportError as error:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "INT8 quantization requires onnxruntime.quantization and a calibration manifest."
        ) from error

    settings = calibration_manifest["quantization"]
    image_paths = _calibration_paths(calibration_manifest)
    if not image_paths:
        raise ValueError("INT8 quantization requires calibration manifest images")
    reader = _CalibrationReader(
        image_paths=image_paths,
        metadata_path=metadata_path,
        batch_size=int(settings["batch_size"]),
    )
    output_model.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        str(input_model),
        str(output_model),
        reader,
        quant_format=QuantFormat.QDQ,
        calibrate_method=_calibration_method(str(settings["calibration_method"])),
        per_channel=bool(settings["per_channel"]),
        activation_type=_quant_type(str(settings["activation_type"])),
        weight_type=_quant_type(str(settings["weight_type"])),
        extra_options={
            "ActivationSymmetric": bool(settings["symmetric_activations"]),
            "WeightSymmetric": bool(settings["symmetric_weights"]),
        },
    )


def quantize_once(
    *,
    fp32_model: Path,
    metadata_path: Path,
    precision: Precision,
    output_model: Path,
    calibration_manifest: Mapping[str, Any] | None,
    keep_fp32_op_types: Sequence[str],
    disable_shape_infer: bool,
) -> dict[str, Any]:
    if precision == "fp32":
        quantize_fp32(fp32_model, output_model)
        settings: dict[str, Any] = {}
    elif precision == "fp16":
        settings = {
            "keep_fp32_op_types": list(keep_fp32_op_types),
            "disable_shape_infer": disable_shape_infer,
        }
        quantize_fp16(
            fp32_model,
            output_model,
            keep_fp32_op_types=keep_fp32_op_types,
            disable_shape_infer=disable_shape_infer,
        )
    else:
        if calibration_manifest is None:
            raise ValueError("INT8 quantization requires --calibration-manifest")
        settings = dict(calibration_manifest["quantization"])
        settings = {
            key: settings[key]
            for key in (
                "calibration_method",
                "per_channel",
                "symmetric_activations",
                "symmetric_weights",
                "activation_type",
                "weight_type",
                "batch_size",
            )
        }
        quantize_int8(
            fp32_model,
            output_model,
            metadata_path=metadata_path,
            calibration_manifest=calibration_manifest,
        )
    return settings


def main() -> None:
    from scripts.check_onnx_quantized_artifacts import load_calibration_manifest

    args = parse_args()
    calibration_manifest = (
        load_calibration_manifest(args.calibration_manifest)
        if args.calibration_manifest is not None
        else None
    )
    settings = quantize_once(
        fp32_model=args.fp32_model,
        metadata_path=args.metadata,
        precision=args.precision,
        output_model=args.output,
        calibration_manifest=calibration_manifest,
        keep_fp32_op_types=tuple(args.keep_fp32_op_type),
        disable_shape_infer=args.disable_shape_infer,
    )
    output_sha256 = sha256_file(args.output)
    if args.repeat_output is not None:
        quantize_once(
            fp32_model=args.fp32_model,
            metadata_path=args.metadata,
            precision=args.precision,
            output_model=args.repeat_output,
            calibration_manifest=calibration_manifest,
            keep_fp32_op_types=tuple(args.keep_fp32_op_type),
            disable_shape_infer=args.disable_shape_infer,
        )
        repeat_sha256 = sha256_file(args.repeat_output)
        if args.require_identical_hash:
            assert_identical_artifact_hashes(output_sha256, repeat_sha256)

    detector_metadata_output = args.detector_metadata_output or default_detector_metadata_output(
        args.output
    )
    copy_detector_metadata(args.metadata, detector_metadata_output)

    sidecar = args.sidecar or default_quantization_sidecar(args.output)
    write_quantization_sidecar(
        sidecar,
        precision=args.precision,
        source_model_sha256=sha256_file(args.fp32_model),
        output_model_sha256=output_sha256,
        framework_commit=framework_commit(),
        checkpoint_revision=args.checkpoint_revision,
        settings=settings,
        toolchain=toolchain_versions(),
    )
    print(json.dumps({"output": str(args.output), "sha256": output_sha256}, indent=2))


if __name__ == "__main__":
    main()
