"""Tests for backend detection and acceleration policy."""

import torch


def test_cpu_backend_info(monkeypatch):
    from complexity.utils.device import get_backend_info, supports_custom_triton

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)

    info = get_backend_info()

    assert info.name == "cpu"
    assert info.device.type == "cpu"
    assert info.matmul == "CPU"
    assert info.distributed == "gloo"
    assert supports_custom_triton("auto") is False


def test_rocm_backend_maps_to_cuda_device(monkeypatch):
    from complexity.utils.device import get_backend, get_backend_info, select_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "AMD Instinct MI300X")
    monkeypatch.setattr(torch.version, "hip", "6.4.0", raising=False)

    info = get_backend_info("auto")

    assert get_backend("auto") == "rocm"
    assert select_device("rocm").type == "cuda"
    assert info.name == "rocm"
    assert info.hip_version == "6.4.0"
    assert info.matmul == "rocBLAS/hipBLASLt"
    assert info.distributed == "RCCL via torch.distributed nccl"
    assert info.custom_triton is False


def test_rocm_custom_triton_is_opt_in(monkeypatch):
    from complexity.utils.device import supports_custom_triton

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "6.4.0", raising=False)
    monkeypatch.delenv("COMPLEXITY_ALLOW_ROCM_TRITON", raising=False)

    assert supports_custom_triton("auto") is False
    assert supports_custom_triton(True) is True

    monkeypatch.setenv("COMPLEXITY_ALLOW_ROCM_TRITON", "1")
    assert supports_custom_triton("auto") is True


def test_nvidia_cuda_enables_custom_triton_in_auto(monkeypatch):
    from complexity.utils.device import get_backend_info, supports_custom_triton

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "NVIDIA H100")
    monkeypatch.setattr(torch.version, "hip", None, raising=False)

    info = get_backend_info("auto")

    assert info.name == "cuda"
    assert info.matmul == "cuBLAS/cuBLASLt"
    assert info.custom_triton is True
    assert supports_custom_triton("auto") is True


def test_sdpa_backend_env_override(monkeypatch):
    from complexity.utils.device import sdpa_kernel_backends

    monkeypatch.setenv("COMPLEXITY_SDPA_BACKENDS", "math")
    backends = sdpa_kernel_backends()

    assert backends is not None
    assert len(backends) == 1
    assert "MATH" in str(backends[0])
