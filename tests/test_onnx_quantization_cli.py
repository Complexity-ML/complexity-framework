import json
from pathlib import Path

import pytest

from scripts.quantize_onnx import assert_identical_artifact_hashes, write_quantization_sidecar


def test_quantization_sidecar_binds_artifact_to_source_and_settings(tmp_path: Path) -> None:
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
