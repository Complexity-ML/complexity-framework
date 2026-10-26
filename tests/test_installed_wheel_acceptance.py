"""Acceptance tests copied outside the checkout into a fresh wheel environment.

The ordinary repository suite skips these checks: their contract requires a
non-editable installed wheel and explicitly selected base/full/vision dependencies.
CI executes them in all profiles; no model or dataset is downloaded.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import unittest
from pathlib import Path

OPTIONAL_DISTRIBUTIONS = {
    "transformers": "transformers",
    "tokenizers": "tokenizers",
    "tiktoken": "tiktoken",
    "datasets": "datasets",
    "huggingface-hub": "huggingface_hub",
    "wandb": "wandb",
    "tensorboard": "tensorboard",
}


def _reports() -> Path:
    path = Path(os.environ["COMPLEXITY_ACCEPTANCE_OUTPUT"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def _console(*args: str) -> subprocess.CompletedProcess[str]:
    # Exercise the installed console entrypoint while rejecting even accidental
    # network access inside its Python process, including during the demo.
    console = shutil.which("complexity")
    assert console is not None
    code = """
import importlib.abc, runpy, socket, sys
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'scripts':
            raise AssertionError('Installed runtime depends on repository scripts: ' + fullname)
sys.meta_path.insert(0, Blocker())
def reject(*args, **kwargs):
    raise AssertionError('Installed-wheel acceptance must remain offline')
socket.socket.connect = reject
socket.create_connection = reject
console, *args = sys.argv[1:]
sys.argv = [console, *args]
runpy.run_path(console, run_name='__main__')
"""
    return subprocess.run(
        [sys.executable, "-I", "-c", code, console, *args],
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )


@unittest.skipUnless(
    os.environ.get("COMPLEXITY_WHEEL_ACCEPTANCE") == "1",
    "Requires the isolated installed-wheel environment, outside the checkout",
)
class InstalledWheelAcceptance(unittest.TestCase):
    def test_wheel_origin_isolated_dependencies_and_console_entrypoint(self):
        import complexity

        prefix = Path(sys.prefix).resolve()
        assert sys.prefix != sys.base_prefix, "Acceptance must use a dedicated venv"
        assert "include-system-site-packages = false" in (prefix / "pyvenv.cfg").read_text().lower()
        package = Path(complexity.__file__).resolve()
        purelib = Path(sysconfig.get_path("purelib")).resolve()
        assert package.is_relative_to(purelib), (package, purelib)
        assert not Path.cwd().resolve().is_relative_to(package.parent.parent)
        console = Path(shutil.which("complexity") or "").resolve()
        assert console.is_relative_to(prefix / "bin"), console
        distribution = importlib.metadata.distribution("complexity-framework")
        direct_url = distribution.read_text("direct_url.json")
        assert direct_url is not None
        source = json.loads(direct_url)
        assert source["url"].endswith(".whl"), source
        assert not source.get("dir_info", {}).get("editable", False)
        profile = os.environ["COMPLEXITY_ACCEPTANCE_PROFILE"]
        assert profile in {"base", "full", "vision"}
        dependencies = {}
        for name, module in OPTIONAL_DISTRIBUTIONS.items():
            installed = importlib.util.find_spec(module) is not None
            expected = profile == "full" or (profile == "vision" and name == "huggingface-hub")
            assert installed == expected, (profile, name, installed)
            if installed:
                dependencies[name] = importlib.metadata.version(name)
        import torch

        assert not torch.cuda.is_available(), "This gate qualifies the CPU backend only"
        evidence = {
            "profile": profile,
            "package": str(package),
            "console": str(console),
            "version": complexity.__version__,
            "python": sys.version,
            "torch": torch.__version__,
            "dependencies": dependencies,
            "system_site_packages": False,
            "cpu_backend": True,
            "mps_qualified": False,
            "mps_reason": "This gate uses CPU; no Apple MPS qualification is performed",
        }
        (_reports() / "installation.json").write_text(json.dumps(evidence, indent=2) + "\n")
        result = _console("--version")
        assert result.returncode == 0, result.stderr
        assert f"complexity {complexity.__version__}" in result.stdout

    def test_all_discovery_help_paths_import_no_runtime_and_remain_offline(self):
        code = """
import importlib.abc, socket, sys
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'torch', 'transformers', 'datasets', 'numpy', 'triton', 'flash_attn',
            'tiktoken', 'tokenizers', 'wandb', 'tensorboard', 'mlx', 'diffusers',
            'safetensors', 'cv2', 'PIL', 'onnx', 'onnxruntime', 'huggingface_hub',
        }:
            raise AssertionError('Unexpected runtime import: ' + fullname)
sys.meta_path.insert(0, Blocker())
def reject(*args, **kwargs):
    raise AssertionError('Unexpected network call')
socket.socket.connect = reject
socket.create_connection = reject
from typer.main import get_command
from typer.testing import CliRunner
from complexity.cli.app import app
from complexity.cli.core import ComplexityCLI
from complexity.data import ComplexityTemplate, ComplexityTokens, Message
from complexity.tokenizer import Tokenizer
runner = CliRunner()
paths = [[]]
def walk(command, prefix):
    for name, child in getattr(command, 'commands', {}).items():
        path = prefix + [name]
        paths.append(path)
        walk(child, path)
walk(get_command(app), [])
for path in paths:
    result = runner.invoke(app, path + ['--help'])
    assert result.exit_code == 0, (path, result.output, result.exception)
for args in (['--version'], ['info', 'version'], ['info', 'tokens'],
             ['tokenize', 'special-tokens'], ['vision', 'list']):
    result = runner.invoke(app, args)
    assert result.exit_code == 0, (args, result.output, result.exception)
assert ComplexityTokens().num_reserved == 2048
assert '<|user|>' in ComplexityCLI().format_tokens('bonjour')
print(len(paths))
"""
        result = subprocess.run(
            [sys.executable, "-I", "-c", code], text=True, capture_output=True, timeout=60
        )
        assert result.returncode == 0, result.stderr
        help_count = int(result.stdout.strip())
        assert help_count > 50, help_count
        (_reports() / "light-imports.json").write_text(
            json.dumps({"help_paths": help_count, "runtime_imports_blocked": True, "offline": True})
            + "\n"
        )

    def test_demo_refuses_existing_artifacts_before_writing(self):
        with tempfile.TemporaryDirectory(prefix="wheel-demo-existing-") as temporary:
            output = Path(temporary) / "existing"
            output.mkdir()
            sentinel = output / "sentinel.txt"
            sentinel.write_text("preserve existing artifacts\n")
            result = _console("train", "demo", "--output", str(output), "--device", "cpu")
            assert result.returncode != 0, result.stdout
            assert sentinel.read_text() == "preserve existing artifacts\n"
            assert list(output.iterdir()) == [sentinel]

    def test_installed_source_provenance_rejects_a_consumer_repository_commit(self):
        with tempfile.TemporaryDirectory(prefix="wheel-consumer-") as temporary:
            consumer = Path(temporary)
            (consumer / "fixture.txt").write_text("unrelated consumer repository\n")
            commands = (
                ["git", "init", "-q", str(consumer)],
                ["git", "-C", str(consumer), "add", "fixture.txt"],
                ["git", "-C", str(consumer), "-c", "user.name=Wheel fixture",
                 "-c", "user.email=fixture@example.invalid", "commit", "-q", "-m", "fixture"],
            )
            for command in commands:
                subprocess.run(command, check=True, capture_output=True, text=True)
            foreign = subprocess.check_output(
                ["git", "-C", str(consumer), "rev-parse", "HEAD"], text=True
            ).strip()
            environment = dict(os.environ, GITHUB_SHA=foreign)
            result = subprocess.run(
                [sys.executable, "-I", "-c", (
                    "from complexity.utils.source_provenance import framework_source_commit; "
                    "print(framework_source_commit())"
                )], cwd=consumer, env=environment, capture_output=True, text=True, timeout=30,
            )
            assert result.returncode == 0, result.stderr
            assert result.stdout.strip() == "unknown", (foreign, result.stdout)
            (_reports() / "source-provenance.json").write_text(json.dumps({
                "passed": True, "framework_commit": "unknown", "consumer_commit": foreign,
                "consumer_commit_rejected": True, "github_sha_rejected": True,
            }, indent=2) + "\n")

    def test_packaged_vision_tools_work_without_repository_scripts(self):
        import complexity

        assert importlib.util.find_spec("scripts") is None
        resource_root = Path(complexity.__file__).parent / "vision" / "detection"
        for resource in ("accuracy_gate.json", "release.json", "quantization_thresholds.json"):
            assert isinstance(json.loads((resource_root / resource).read_text()), dict)
        if os.environ["COMPLEXITY_ACCEPTANCE_PROFILE"] != "vision":
            return

        paths = (
            ("evaluate", "detector"), ("evaluate", "onnx"),
            ("export", "detector"), ("export", "tensorrt"),
            ("check", "coco"), ("check", "onnx-parity"), ("check", "quantized-parity"),
            ("merge", "coco"), ("infer", "onnx"), ("benchmark", "onnx"),
            ("quantize", "onnx"), ("release", "onnx"),
        )
        for operation, task in paths:
            result = _console("vision", operation, task, "--", "--help")
            assert result.returncode == 0, (operation, task, result.stderr, result.stdout)
            assert "usage:" in result.stdout.lower(), result.stdout

        output = _reports() / "vision"
        output.mkdir()
        invalid = output / "invalid.json"
        invalid.write_text("{}")
        result = _console("vision", "check", "coco", str(invalid))
        assert result.returncode == 1, (result.stdout, result.stderr)
        assert "malformed_report: report has no branch results" in result.stdout
        assert "missing JSON file" not in result.stdout + result.stderr

        metrics = {key: 0.1 for key in (
            "map50_95", "map50", "map75", "ap_small", "ap_medium", "ap_large", "ar_100"
        )}
        report = {
            "schema_version": 1, "backend": "onnx", "framework_commit": "synthetic-fixture",
            "checkpoint": "fixture.onnx", "checkpoint_sha256": "c" * 64, "dataset": {
                "name": "synthetic", "split": "fixture", "evaluated_images": 2,
                "annotations_sha256": "a" * 64, "image_list_sha256": "b" * 64,
            },
            "protocol": {"seed": 0, "release_eligible": False},
            "environment": {"python": sys.version, "os": sys.platform,
                "torch": "fixture", "onnxruntime": "fixture", "cuda_available": False,
                "torch_cuda": None, "tensorrt": None},
        }
        sources = []
        for branch in ("o2m-nms", "nms-free"):
            path = output / (branch + ".json")
            path.write_text(json.dumps({**report, "branches": {branch: {"metrics": metrics}}}))
            sources.append(str(path))
        merged = output / "merged"
        result = _console("vision", "merge", "coco", *sources, "--output", str(merged))
        assert result.returncode == 0, (result.stdout, result.stderr)
        payload = json.loads((merged / "evaluation.json").read_text())
        assert set(payload["branches"]) == {"o2m-nms", "nms-free"}
        assert payload["protocol"]["release_eligible"] is False
        assert (merged / "evaluation.md").is_file()
        assert payload["branches"]["o2m-nms"]["metrics"] == metrics
        (output / "acceptance.json").write_text(json.dumps({
            "passed": True, "runtime_help_paths": len(paths), "repository_scripts": False,
            "offline": True, "synthetic_data": True, "release_eligible": False,
        }, indent=2) + "\n")

    def test_offline_cpu_demo_trains_resumes_evaluates_and_reloads_export(self):
        output = _reports() / "demo"
        result = _console("train", "demo", "--output", str(output), "--device", "cpu")
        (_reports() / "demo.stderr.txt").write_text(result.stderr)
        (_reports() / "demo.stdout.json").write_text(result.stdout)
        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        assert report["format"] == "complexity-local-training-demo-v1"
        assert report["device"] == "cpu" and report["precision"] == "fp32"
        assert report["passed"] is True
        assert report["synthetic_data"] is True and report["downloaded_assets"] is False
        assert report["release_eligible"] is False
        assert report["planned_updates"] == report["final_update"] == report["data_cursor"] == 4
        assert report["pause_update"] == report["resumed_from_update"] == 2
        assert report["model_parameters"] > 0
        for field in ("initial_eval_loss", "final_eval_loss"):
            assert math.isfinite(report[field]) and report[field] > 0, (field, report[field])
        assert set(report["checks"]) >= {
            "model",
            "optimizer",
            "scheduler",
            "rng",
            "logits",
            "cursor",
            "sample_order",
            "parameters_updated",
            "export_reload",
        }
        assert all(check["passed"] is True for check in report["checks"].values())
        for field in ("checkpoint", "export", "report"):
            assert Path(report[field]).resolve().is_relative_to(output.resolve()), report[field]
        assert json.loads(Path(report["report"]).read_text()) == report
        assert (Path(report["checkpoint"]) / "complete.json").is_file()
        assert (Path(report["checkpoint"]) / "checkpoint.pt").is_file()
        assert (Path(report["export"]) / "config.json").is_file()
        assert (Path(report["export"]) / "model.safetensors").is_file()


if __name__ == "__main__":
    suite = unittest.main(exit=False)
    if os.environ.get("COMPLEXITY_WHEEL_ACCEPTANCE") == "1":
        (_reports() / "acceptance.json").write_text(
            json.dumps(
                {
                    "passed": suite.result.wasSuccessful() and not suite.result.skipped,
                    "tests_run": suite.result.testsRun,
                    "failures": len(suite.result.failures),
                    "errors": len(suite.result.errors),
                    "skipped": len(suite.result.skipped),
                    "device": "cpu",
                    "mps_qualified": False,
                    "profile": os.environ["COMPLEXITY_ACCEPTANCE_PROFILE"],
                },
                indent=2,
            )
            + "\n"
        )
    sys.exit(0 if suite.result.wasSuccessful() and not suite.result.skipped else 1)
