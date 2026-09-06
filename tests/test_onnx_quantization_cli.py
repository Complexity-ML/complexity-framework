import json
import sys
import types
from pathlib import Path

import pytest

from scripts import quantize_onnx
from scripts.quantize_onnx import (
    assert_identical_artifact_hashes,
    copy_detector_metadata,
    default_detector_metadata_output,
    default_quantization_sidecar,
    quantize_int8,
    write_quantization_sidecar,
)


def test_quantization_sidecar_binds_artifact_to_source_and_settings(
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "model.fp16.json"

    write_quantization_sidecar(
        sidecar,
        precision="fp16",
        source_model_sha256="source",
        output_model_sha256="output",
        framework_commit="commit",
        checkpoint_revision="checkpoint",
        settings={"keep_fp32_op_types": ["ReduceSum"]},
        toolchain={"onnx": "1.17.0"},
    )

    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["precision"] == "fp16"
    assert data["source_model_sha256"] == "source"
    assert data["output_model_sha256"] == "output"
    assert data["framework_commit"] == "commit"
    assert data["checkpoint_revision"] == "checkpoint"
    assert data["settings"]["keep_fp32_op_types"] == ["ReduceSum"]


def test_repeat_quantization_hash_mismatch_fails_loudly() -> None:
    with pytest.raises(ValueError, match="quantization is not deterministic"):
        assert_identical_artifact_hashes("first", "second")


def test_default_sidecars_keep_detector_metadata_separate(tmp_path: Path) -> None:
    output = tmp_path / "tr_hash_v8_o2m_fp16.onnx"
    source_metadata = tmp_path / "tr_hash_v8_o2m.json"
    source_metadata.write_text('{"branch": "o2m", "image_size": 640}', encoding="utf-8")

    detector_sidecar = default_detector_metadata_output(output)
    copy_detector_metadata(source_metadata, detector_sidecar)

    assert detector_sidecar == tmp_path / "tr_hash_v8_o2m_fp16.json"
    assert default_quantization_sidecar(output) == (
        tmp_path / "tr_hash_v8_o2m_fp16.quantization.json"
    )
    assert json.loads(detector_sidecar.read_text(encoding="utf-8"))["branch"] == "o2m"


def test_int8_quantization_passes_claimed_settings_to_ort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class QuantFormat:
        QDQ = "QDQ"

    class CalibrationMethod:
        MinMax = "MinMax"
        Entropy = "Entropy"
        Percentile = "Percentile"

    class QuantType:
        QInt8 = "QInt8"
        QUInt8 = "QUInt8"

    def quantize_static(*args: object, **kwargs: object) -> None:
        calls["args"] = args
        calls["kwargs"] = kwargs
        Path(args[1]).write_bytes(b"int8")

    fake_quantization = types.SimpleNamespace(
        CalibrationMethod=CalibrationMethod,
        QuantFormat=QuantFormat,
        QuantType=QuantType,
        quantize_static=quantize_static,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime.quantization", fake_quantization)
    monkeypatch.setattr(
        quantize_onnx,
        "_calibration_paths",
        lambda _manifest: [tmp_path / "a.jpg", tmp_path / "b.jpg"],
    )

    class Reader:
        def __init__(self, **kwargs: object) -> None:
            calls["reader_kwargs"] = kwargs

    monkeypatch.setattr(quantize_onnx, "_CalibrationReader", Reader)

    manifest = {
        "images": ["a.jpg", "b.jpg"],
        "quantization": {
            "calibration_method": "minmax",
            "per_channel": True,
            "symmetric_activations": False,
            "symmetric_weights": True,
            "activation_type": "quint8",
            "weight_type": "qint8",
            "batch_size": 2,
        },
    }

    quantize_int8(
        tmp_path / "fp32.onnx",
        tmp_path / "int8.onnx",
        metadata_path=tmp_path / "metadata.json",
        calibration_manifest=manifest,
    )

    assert calls["reader_kwargs"] == {
        "image_paths": [tmp_path / "a.jpg", tmp_path / "b.jpg"],
        "metadata_path": tmp_path / "metadata.json",
        "batch_size": 2,
    }
    assert calls["kwargs"]["per_channel"] is True
    assert calls["kwargs"]["activation_type"] == "QUInt8"
    assert calls["kwargs"]["weight_type"] == "QInt8"
    assert calls["kwargs"]["extra_options"] == {
        "ActivationSymmetric": False,
        "WeightSymmetric": True,
    }


def test_int8_sidecar_settings_only_include_applied_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(quantize_onnx, "quantize_int8", lambda *_, **__: None)
    manifest = {
        "quantization": {
            "calibration_method": "minmax",
            "per_channel": True,
            "symmetric_activations": False,
            "symmetric_weights": True,
            "activation_type": "quint8",
            "weight_type": "qint8",
            "batch_size": 2,
            "num_threads": 1,
        }
    }

    settings = quantize_onnx.quantize_once(
        fp32_model=tmp_path / "fp32.onnx",
        metadata_path=tmp_path / "metadata.json",
        precision="int8",
        output_model=tmp_path / "int8.onnx",
        calibration_manifest=manifest,
        keep_fp32_op_types=(),
        disable_shape_infer=False,
    )

    assert settings == {
        "calibration_method": "minmax",
        "per_channel": True,
        "symmetric_activations": False,
        "symmetric_weights": True,
        "activation_type": "quint8",
        "weight_type": "qint8",
        "batch_size": 2,
    }
