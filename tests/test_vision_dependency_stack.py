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


def test_backend_installer_keeps_torchvision_on_backend_index(tmp_path, monkeypatch) -> None:
    """Execute the installer with a fake interpreter; never install packages."""
    import json
    import subprocess
    import sys

    trace = tmp_path / "commands.jsonl"
    interpreter = tmp_path / "fake-python"
    interpreter.write_text(f"#!{sys.executable}\n" + r'''
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
entry = {"args": args}
if args[:3] == ["-m", "pip", "install"] and "-c" in args:
    entry["constraints"] = Path(args[args.index("-c") + 1]).read_text()
with Path(os.environ["CF_INSTALL_TRACE"]).open("a") as stream:
    stream.write(json.dumps(entry) + "\n")
if args and args[0] == "-" and len(args) > 1:
    Path(args[1]).write_text("torch==9.0.0+cpu\ntorchvision==9.0.0+cpu\n")
''')
    interpreter.chmod(0o755)
    monkeypatch.setenv("PYTHON", str(interpreter))
    monkeypatch.setenv("CF_INSTALL_TRACE", str(trace))
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/install_backend.sh"), "cpu", "vision"],
        capture_output=True, text=True, check=True,
    )
    assert "backend=cpu profile=vision" in result.stdout
    entries = [json.loads(line) for line in trace.read_text().splitlines()]
    installs = [entry for entry in entries if entry["args"][:3] == ["-m", "pip", "install"]]
    assert len(installs) == 2
    assert installs[0]["args"][3:] == [
        "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu",
    ]
    assert installs[1]["args"][-2:] == ["-e", ".[image,detection,export]"]
    assert installs[1]["constraints"] == "torch==9.0.0+cpu\ntorchvision==9.0.0+cpu\n"
    constraints = Path(installs[1]["args"][installs[1]["args"].index("-c") + 1])
    assert not constraints.exists(), "temporary wheel constraints must be cleaned up"


def test_backend_installer_preview_does_not_execute_interpreter(tmp_path, monkeypatch):
    import subprocess
    missing = tmp_path / "must-not-execute"
    monkeypatch.setenv("PYTHON", str(missing))
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/install_backend.sh"), "cpu", "core", "--dry-run"],
        capture_output=True, text=True, check=True,
    )
    assert str(missing) in result.stdout
    assert "backend-constraints" in result.stdout
