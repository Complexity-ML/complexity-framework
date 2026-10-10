import sys
from pathlib import Path

import pytest

from complexity.vision.detection import tensorrt_export


def test_tensorrt_help_does_not_require_a_gpu_runtime(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "tensorrt", None)
    monkeypatch.setattr(sys, "argv", ["tensorrt-export", "--help"])

    with pytest.raises(SystemExit) as exit_info:
        tensorrt_export.main()

    assert exit_info.value.code == 0
    assert "--max-batch" in capsys.readouterr().out


def test_missing_tensorrt_is_reported_before_artifact_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "tensorrt", None)

    with pytest.raises(ImportError, match="pip install tensorrt"):
        tensorrt_export.build_engine(tmp_path / "missing.onnx", tmp_path / "output.engine")

    assert not (tmp_path / "output.engine").exists()
