from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


def _requirement_names(requirements: list[str]) -> set[str]:
    names = set()
    for requirement in requirements:
        name = requirement.split(";")[0].split(">=")[0].split("!=")[0]
        names.add(name)
    return names


def test_detection_extra_tracks_public_vision_runtime() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    requirements = [
        *project["project"]["dependencies"],
        *project["project"]["optional-dependencies"]["detection"],
    ]

    assert {
        "filelock",
        "matplotlib",
        "opencv-python",
        "Pillow",
        "requests",
        "torchvision",
        "psutil",
        "polars",
        "nvidia-ml-py",
        "albumentations",
        "faster-coco-eval",
        "pycocotools",
    } <= _requirement_names(requirements)


def test_export_extra_has_onnx_slim_and_runtime() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    requirements = "\n".join(project["project"]["optional-dependencies"]["export"])

    assert "onnxslim>=0.1.82" in requirements
    assert "onnxruntime>=1.20.0" in requirements


def test_backend_installer_keeps_torchvision_on_backend_index() -> None:
    installer = (ROOT / "scripts" / "install_backend.sh").read_text()

    assert "TORCH_PACKAGES+=(torchvision)" in installer
    assert '"${TORCH_PACKAGES[@]}" --index-url "$INDEX"' in installer
    assert 'PROFILE="${2:-core}"' in installer
    assert "onnxslim>=0.1.82" in installer
    assert "onnxruntime>=1.20.0" in installer
