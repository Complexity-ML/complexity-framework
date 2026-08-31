"""ONNX Runtime session wrapper for detector exports."""

from __future__ import annotations

import os
import site
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class OrtSessionConfig:
    """Configuration for constructing an ONNX Runtime detector session."""

    model_path: Path | str
    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    warmup_runs: int = 1
    intra_op_num_threads: int | None = None
    inter_op_num_threads: int | None = None


class OnnxDetectorSession:
    """Thin wrapper around an ONNX Runtime inference session."""

    def __init__(self, config: OrtSessionConfig, session: object | None = None) -> None:
        self.config = config
        self._session = session
        self._dll_directory_handles: list[object] = []

    def open(self) -> "OnnxDetectorSession":
        if self._session is None:
            import onnxruntime as ort

            if _needs_cuda_dlls(self.config.providers) and hasattr(ort, "preload_dlls"):
                # Prefer NVIDIA site-package DLLs over an unrelated PyTorch CUDA build.
                ort.preload_dlls(directory="")
            self._add_tensorrt_dll_directories()
            session_options = ort.SessionOptions()
            if self.config.intra_op_num_threads is not None:
                session_options.intra_op_num_threads = self.config.intra_op_num_threads
            if self.config.inter_op_num_threads is not None:
                session_options.inter_op_num_threads = self.config.inter_op_num_threads
            self._session = ort.InferenceSession(
                str(self.config.model_path),
                sess_options=session_options,
                providers=list(self.config.providers),
            )
        return self

    @property
    def provider_used(self) -> str:
        providers = self._require_session().get_providers()
        return providers[0] if providers else ""

    @property
    def input_name(self) -> str:
        return self._require_session().get_inputs()[0].name

    @property
    def output_name(self) -> str:
        return self._require_session().get_outputs()[0].name

    def warmup(self, input_shape: Sequence[int]) -> None:
        dummy = np.zeros(tuple(input_shape), dtype=np.float32)
        for _ in range(self.config.warmup_runs):
            self.run(dummy)

    def run(self, pixel_values: np.ndarray) -> np.ndarray:
        output = self._require_session().run(
            [self.output_name],
            {self.input_name: pixel_values},
        )[0]
        return np.asarray(output, dtype=np.float32)

    def _add_tensorrt_dll_directories(self) -> None:
        if not _needs_tensorrt_dlls(self.config.providers):
            return
        for site_packages in _candidate_site_packages():
            tensorrt_libs = site_packages / "tensorrt_libs"
            if not tensorrt_libs.is_dir():
                continue
            if hasattr(os, "add_dll_directory"):
                self._dll_directory_handles.append(
                    os.add_dll_directory(str(tensorrt_libs))
                )
            os.environ["PATH"] = str(tensorrt_libs) + os.pathsep + os.environ.get(
                "PATH", ""
            )

    def _require_session(self):
        if self._session is None:
            raise RuntimeError("ONNX Runtime session is not open")
        return self._session


def _needs_cuda_dlls(providers: Sequence[str]) -> bool:
    cuda_providers = {"CUDAExecutionProvider", "TensorrtExecutionProvider"}
    return any(provider in cuda_providers for provider in providers)


def _needs_tensorrt_dlls(providers: Sequence[str]) -> bool:
    return any(provider == "TensorrtExecutionProvider" for provider in providers)


def _candidate_site_packages() -> tuple[Path, ...]:
    candidates: list[Path] = []
    for raw_path in [*site.getsitepackages(), site.getusersitepackages()]:
        path = Path(raw_path)
        if path not in candidates:
            candidates.append(path)
    return tuple(candidates)
