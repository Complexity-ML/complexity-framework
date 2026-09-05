"""Build and verify reproducible Vision v8 ONNX release artifacts.

The release is pinned by ``docs/onnx/release.json``: checkpoint repository and
revision, opset, and the exact export toolchain. Pinning the toolchain is what
makes the artifacts reproducible from a repository commit — the ONNX graph, and
therefore the SHA-256 of the binary, depends on the PyTorch version that traced
it.

Publication is fail-closed: a toolchain mismatch, a failed export, a failed
parity gate, or a checksum mismatch aborts before anything is uploaded.

Usage:
    python scripts/build_onnx_release.py --output-dir dist/onnx
    python scripts/build_onnx_release.py --verify dist/onnx/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("COMPLEXITY_DISABLE_KERNELS", "1")

MANIFEST_NAME = "manifest.json"
RELEASE_NOTES_NAME = "RELEASE_NOTES.md"
MANIFEST_VERSION = 1
DEFAULT_CONFIG_PATH = Path("docs/onnx/release.json")


@dataclass(frozen=True)
class BranchSpec:
    """One exported prediction branch."""

    branch: str
    checkpoint_subdir: str | None
    stem: str
    post_processing: str

    @property
    def model_name(self) -> str:
        return f"{self.stem}.onnx"

    @property
    def sidecar_name(self) -> str:
        return f"{self.stem}.json"


@dataclass(frozen=True)
class ReleaseConfig:
    """Pinned inputs for a reproducible release build."""

    checkpoint_repo: str
    checkpoint_revision: str
    opset: int
    parity_num_tests: int
    toolchain: Mapping[str, str]
    branches: tuple[BranchSpec, ...]
    quantization: QuantizationConfig | None = None


@dataclass(frozen=True)
class QuantizationConfig:
    """Pinned inputs for optional quantized release artifacts."""

    enabled_precisions: tuple[str, ...]
    calibration_manifest: Path
    thresholds: Path
    accuracy_report: Path
    accuracy_markdown: Path
    fp32_op_allowlist: tuple[str, ...]
    provider_gates: tuple[tuple[str, str], ...]


class ReleaseError(RuntimeError):
    """Any condition that must prevent publication."""


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> ReleaseConfig:
    """Load and validate the pinned release configuration."""

    return config_from_mapping(json.loads(Path(path).read_text()))


def config_from_mapping(values: Mapping[str, Any]) -> ReleaseConfig:
    branches = tuple(
        BranchSpec(
            branch=str(entry["branch"]),
            checkpoint_subdir=(
                None if entry.get("checkpoint_subdir") is None else str(entry["checkpoint_subdir"])
            ),
            stem=str(entry["stem"]),
            post_processing=str(entry["post_processing"]),
        )
        for entry in values["branches"]
    )
    if not branches:
        raise ReleaseError("release config must declare at least one branch")
    if len({spec.branch for spec in branches}) != len(branches):
        raise ReleaseError("release config declares a branch twice")

    revision = str(values["checkpoint_revision"])
    if len(revision) != 40 or not all(c in "0123456789abcdef" for c in revision):
        raise ReleaseError(
            "checkpoint_revision must be a full 40-character commit sha, "
            f"got {revision!r}; a moving ref would break reproducibility"
        )

    toolchain = values["toolchain"]
    if not isinstance(toolchain, Mapping) or not toolchain:
        raise ReleaseError("release config must pin a toolchain")

    quantization = None
    if isinstance(values.get("quantization"), Mapping):
        raw_quantization = values["quantization"]
        enabled_precisions = tuple(
            str(precision) for precision in raw_quantization.get("enabled_precisions", ())
        )
        unsupported = sorted(set(enabled_precisions) - {"fp16", "int8"})
        if unsupported:
            raise ReleaseError(f"unsupported quantized precisions: {unsupported}")
        provider_gates = tuple(
            (str(gate["provider"]), str(gate["precision"]))
            for gate in raw_quantization.get("provider_gates", ())
        )
        quantization = QuantizationConfig(
            enabled_precisions=enabled_precisions,
            calibration_manifest=Path(str(raw_quantization["calibration_manifest"])),
            thresholds=Path(str(raw_quantization["thresholds"])),
            accuracy_report=Path(str(raw_quantization["accuracy_report"])),
            accuracy_markdown=Path(str(raw_quantization["accuracy_markdown"])),
            fp32_op_allowlist=tuple(
                str(op_type) for op_type in raw_quantization.get("fp32_op_allowlist", ())
            ),
            provider_gates=provider_gates,
        )

    return ReleaseConfig(
        checkpoint_repo=str(values["checkpoint_repo"]),
        checkpoint_revision=revision,
        opset=int(values["opset"]),
        parity_num_tests=int(values.get("parity_num_tests", 5)),
        toolchain={str(k): str(v) for k, v in toolchain.items()},
        branches=branches,
        quantization=quantization,
    )


def installed_toolchain() -> dict[str, str]:
    """Return the installed versions of the packages that shape the export."""

    import onnx
    import onnxruntime
    import torch

    # Local version tags (``+cu130``) identify the wheel build, not the graph.
    return {
        "torch": torch.__version__.split("+")[0],
        "onnx": onnx.__version__,
        "onnxruntime": onnxruntime.__version__,
    }


def toolchain_mismatches(
    pinned: Mapping[str, str],
    installed: Mapping[str, str],
) -> list[str]:
    """Return one message per package whose version departs from the pin."""

    return [
        f"{package}: pinned {version}, installed {installed.get(package, 'missing')}"
        for package, version in pinned.items()
        if installed.get(package) != version
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def framework_commit() -> str:
    """Return the commit the release is built from."""

    from_ci = os.environ.get("GITHUB_SHA")
    if from_ci:
        return from_ci
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def artifact_entry(path: Path, **fields: Any) -> dict[str, Any]:
    """Describe one published file by name, size and digest."""

    resolved = Path(path)
    return {
        "name": resolved.name,
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        **fields,
    }


def provider_chain(provider: str) -> tuple[str, ...]:
    if provider == "TensorrtExecutionProvider":
        return (
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        )
    if provider == "CUDAExecutionProvider":
        return ("CUDAExecutionProvider", "CPUExecutionProvider")
    return (provider,)


def coco_report_branch(branch: str) -> str:
    return "o2m-nms" if branch == "o2m" else branch


def output_contract(sidecar: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the ONNX input/output contract from an export sidecar."""

    image_size = int(sidecar["image_size"])
    prediction_width = int(sidecar["regression_width"]) + int(sidecar["num_classes"])
    return {
        "input_name": "pixel_values",
        "input_shape": [1, 3, image_size, image_size],
        "output_name": "predictions",
        "output_shape": [1, int(sidecar["num_cells"]), prediction_width],
        "dtype": "float32",
        "output_semantics": sidecar.get("output_semantics", ""),
        "regression_width": int(sidecar["regression_width"]),
        "num_classes": int(sidecar["num_classes"]),
        "grid_sizes": list(sidecar["grid_sizes"]),
    }


def build_manifest(
    config: ReleaseConfig,
    artifacts: Sequence[Mapping[str, Any]],
    *,
    commit: str,
    toolchain: Mapping[str, str],
) -> dict[str, Any]:
    """Assemble the machine-readable release manifest."""

    return {
        "manifest_version": MANIFEST_VERSION,
        "framework_commit": commit,
        "checkpoint_repo": config.checkpoint_repo,
        "checkpoint_revision": config.checkpoint_revision,
        "opset": config.opset,
        "parity_num_tests": config.parity_num_tests,
        "toolchain": dict(toolchain),
        "artifacts": [dict(artifact) for artifact in artifacts],
    }


def verify_manifest(
    manifest: Mapping[str, Any],
    directory: Path,
    *,
    expect_commit: str | None = None,
) -> list[str]:
    """Return one message per artifact that is missing, resized or altered.

    ``expect_commit`` additionally binds the manifest to the commit publication
    happens from: a release whose tag resolves elsewhere would document digests
    that its own source tree cannot reproduce.
    """

    problems: list[str] = []
    if expect_commit is not None:
        recorded = str(manifest.get("framework_commit", ""))
        if recorded != expect_commit:
            problems.append(f"framework_commit {recorded or 'missing'}, expected {expect_commit}")
    for artifact in manifest["artifacts"]:
        path = Path(directory) / str(artifact["name"])
        if not path.is_file():
            problems.append(f"{artifact['name']}: missing")
            continue
        actual_size = path.stat().st_size
        if actual_size != int(artifact["size_bytes"]):
            problems.append(
                f"{artifact['name']}: size {actual_size}, manifest {artifact['size_bytes']}"
            )
        actual_digest = sha256_file(path)
        if actual_digest != str(artifact["sha256"]):
            problems.append(
                f"{artifact['name']}: sha256 {actual_digest}, manifest {artifact['sha256']}"
            )
    return problems


def render_release_notes(manifest: Mapping[str, Any]) -> str:
    """Render release notes that keep the two branches distinguishable."""

    lines = [
        "# TR-HASH Vision v8 ONNX artifacts",
        "",
        f"Built from framework commit `{manifest['framework_commit']}` and "
        f"checkpoint `{manifest['checkpoint_repo']}` at revision "
        f"`{manifest['checkpoint_revision']}`, opset `{manifest['opset']}`.",
        "",
        "Both models expose raw detector logits only; decode and post-processing "
        "run outside the graph.",
        "",
        "| Branch | Precision | Model | Post-processing | Size | SHA-256 |",
        "|---|---|---|---|---:|---|",
    ]
    for artifact in manifest["artifacts"]:
        if artifact.get("kind") != "model":
            continue
        lines.append(
            f"| {artifact['branch']} | {artifact.get('precision', 'unknown')} | "
            f"`{artifact['name']}` | "
            f"{artifact['post_processing']} | {artifact['size_bytes']:,} bytes | "
            f"`{artifact['sha256']}` |"
        )

    lines += [
        "",
        "## Verifying a download",
        "",
        "```bash",
        "sha256sum -c <(python - <<'PY'",
        "import json",
        f"manifest = json.load(open('{MANIFEST_NAME}'))",
        "for a in manifest['artifacts']:",
        "    print(f\"{a['sha256']}  {a['name']}\")",
        "PY",
        ")",
        "```",
        "",
        "## Toolchain",
        "",
        "The exported graph depends on the tracing toolchain, so reproducing "
        "these digests requires the pinned versions:",
        "",
        "| Package | Version |",
        "|---|---|",
    ]
    for package, version in manifest["toolchain"].items():
        lines.append(f"| `{package}` | `{version}` |")
    lines.append("")
    return "\n".join(lines)


def render_benchmark_report(report: Mapping[str, Any]) -> str:
    """Render benchmark evidence into a release-friendly Markdown table."""

    lines = [
        "# Vision v8 ONNX Quantization Benchmarks",
        "",
        "| Branch | Precision | Provider | Median ms | P95 ms | Throughput | Peak MB |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for branch, branch_report in _mapping(report.get("branches")).items():
        for precision, precision_report in _mapping(branch_report).items():
            data = _mapping(precision_report)
            latency = _mapping(data.get("latency"))
            lines.append(
                f"| {branch} | {precision} | {data.get('actual_provider', '')} | "
                f"{float(latency.get('median_ms', 0.0)):.3f} | "
                f"{float(latency.get('p95_ms', 0.0)):.3f} | "
                f"{float(data.get('throughput_images_per_second', 0.0)):.3f} | "
                f"{float(data.get('peak_memory_mb', 0.0)):.3f} |"
            )
    lines.append("")
    return "\n".join(lines)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def build_release(
    config: ReleaseConfig,
    output_dir: Path,
    *,
    allow_toolchain_drift: bool = False,
) -> dict[str, Any]:
    """Download, export, gate and describe the release. Raises on any failure."""

    from huggingface_hub import snapshot_download

    from scripts.benchmark_onnx_artifacts import benchmark_onnx_artifact
    from scripts.check_onnx_parity import check_parity
    from scripts.check_onnx_quantized_artifacts import (
        assert_disjoint_image_ids,
        check_provider_precision_supported,
        check_quantized_accuracy_report,
        check_quantized_benchmark_report,
        check_quantized_parity_report,
        check_unexpected_fp32_nodes,
        evaluation_image_ids_from_report,
        inspect_onnx_node_dtypes,
        load_calibration_manifest,
        load_quantization_thresholds,
    )
    from scripts.check_onnx_quantized_parity import build_parity_report
    from scripts.export_onnx import export_onnx
    from scripts.quantize_onnx import (
        assert_identical_artifact_hashes,
        copy_detector_metadata,
        default_quantization_sidecar,
        quantize_once,
        write_quantization_sidecar,
    )
    from scripts.quantize_onnx import (
        toolchain_versions as quantization_toolchain_versions,
    )

    installed = installed_toolchain()
    mismatches = toolchain_mismatches(config.toolchain, installed)
    if mismatches:
        message = "toolchain does not match the pinned release toolchain:\n  " + "\n  ".join(
            mismatches
        )
        if not allow_toolchain_drift:
            raise ReleaseError(
                f"{message}\nDigests would not be reproducible. "
                "Pass --allow-toolchain-drift for a local dry run."
            )
        print(f"WARNING: {message}")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    commit = framework_commit()
    artifacts: list[dict[str, Any]] = []

    quantization_thresholds: Mapping[str, Any] | None = None
    calibration_manifest: Mapping[str, Any] | None = None
    accuracy_report: Mapping[str, Any] | None = None
    provider_by_precision: dict[str, str] = {}
    benchmark_report: dict[str, Any] | None = None
    benchmark_settings: Mapping[str, Any] = {}
    expected_accuracy_artifacts: dict[str, dict[str, dict[str, str]]] = {}
    if config.quantization is not None:
        quantization_thresholds = load_quantization_thresholds(config.quantization.thresholds)
        for provider, precision in config.quantization.provider_gates:
            check_provider_precision_supported(
                provider,
                precision,
                quantization_thresholds,
            )
            provider_by_precision[precision] = provider
        if config.quantization.enabled_precisions:
            calibration_manifest = load_calibration_manifest(
                config.quantization.calibration_manifest
            )
            batch_size = int(calibration_manifest["quantization"]["batch_size"])
            if batch_size != 1:
                raise ReleaseError(
                    "default ONNX release exports fixed batch size 1, "
                    f"but calibration batch_size is {batch_size}"
                )
        if config.quantization.enabled_precisions and calibration_manifest is None:
            raise ReleaseError("quantized release parity requires calibration images")
        parity_image = Path(str(calibration_manifest["images"][0]))
        if not config.quantization.accuracy_report.is_file():
            raise ReleaseError(
                "quantized release requires an accuracy comparison report: "
                f"{config.quantization.accuracy_report}"
            )
        if not config.quantization.accuracy_markdown.is_file():
            raise ReleaseError(
                "quantized release requires a Markdown accuracy report: "
                f"{config.quantization.accuracy_markdown}"
            )
        accuracy_report = json.loads(
            config.quantization.accuracy_report.read_text(encoding="utf-8")
        )
        assert_disjoint_image_ids(
            {int(image_id) for image_id in calibration_manifest["image_ids"]},
            evaluation_image_ids_from_report(accuracy_report),
        )
        benchmark_report = {"schema_version": 1, "branches": {}}
        benchmark_settings = _mapping(quantization_thresholds.get("benchmark"))

    print(f"Downloading {config.checkpoint_repo}@{config.checkpoint_revision}")
    checkpoint_root = Path(
        snapshot_download(
            repo_id=config.checkpoint_repo,
            revision=config.checkpoint_revision,
        )
    )

    for spec in config.branches:
        checkpoint = (
            checkpoint_root
            if spec.checkpoint_subdir is None
            else checkpoint_root / spec.checkpoint_subdir
        )
        model_path = destination / spec.model_name

        print(f"\n=== {spec.branch} ===")
        export_onnx(
            checkpoint,
            model_path,
            opset_version=config.opset,
            check=True,
            branch=spec.branch,
        )

        if not check_parity(
            checkpoint,
            model_path,
            branch=spec.branch,
            num_tests=config.parity_num_tests,
        ):
            raise ReleaseError(f"parity gates failed for branch {spec.branch}")

        sidecar_path = model_path.with_suffix(".json")
        sidecar = json.loads(sidecar_path.read_text())
        artifacts.append(
            artifact_entry(
                model_path,
                kind="model",
                branch=spec.branch,
                precision="fp32",
                requires_nms=bool(sidecar["requires_nms"]),
                post_processing=spec.post_processing,
                contract=output_contract(sidecar),
            )
        )
        artifacts.append(
            artifact_entry(
                sidecar_path,
                kind="metadata",
                branch=spec.branch,
                precision="fp32",
            )
        )
        if config.quantization is not None and quantization_thresholds is not None:
            expected_accuracy_artifacts.setdefault(coco_report_branch(spec.branch), {})["fp32"] = {
                "checkpoint_sha256": sha256_file(model_path),
                "metadata_sha256": sha256_file(sidecar_path),
            }
            fp32_benchmark = benchmark_onnx_artifact(
                model_path=model_path,
                metadata_path=sidecar_path,
                providers=provider_chain(provider_by_precision.get("fp32", "CPUExecutionProvider")),
                batch_size=1,
                warmup_iterations=int(benchmark_settings["warmup_iterations"]),
                measured_iterations=int(benchmark_settings["measured_iterations"]),
                ort_intra_op_threads=1,
                ort_inter_op_threads=1,
            )
            check_provider_precision_supported(
                str(fp32_benchmark["actual_provider"]),
                "fp32",
                quantization_thresholds,
            )
            benchmark_report["branches"].setdefault(spec.branch, {})["fp32"] = fp32_benchmark
        if config.quantization is not None and quantization_thresholds is not None:
            for precision in config.quantization.enabled_precisions:
                quantized_model_path = destination / f"{spec.stem}_{precision}.onnx"
                repeat_model_path = destination / f"{spec.stem}_{precision}_repeat.onnx"
                print(f"Quantizing {spec.branch} to {precision}")
                settings = quantize_once(
                    fp32_model=model_path,
                    metadata_path=sidecar_path,
                    precision=precision,
                    output_model=quantized_model_path,
                    calibration_manifest=calibration_manifest,
                    keep_fp32_op_types=config.quantization.fp32_op_allowlist,
                    disable_shape_infer=False,
                )
                quantize_once(
                    fp32_model=model_path,
                    metadata_path=sidecar_path,
                    precision=precision,
                    output_model=repeat_model_path,
                    calibration_manifest=calibration_manifest,
                    keep_fp32_op_types=config.quantization.fp32_op_allowlist,
                    disable_shape_infer=False,
                )
                assert_identical_artifact_hashes(
                    sha256_file(quantized_model_path),
                    sha256_file(repeat_model_path),
                )
                repeat_model_path.unlink()

                precision_thresholds = quantization_thresholds["precisions"][precision]
                if precision_thresholds.get("unexpected_fp32_nodes") == "fail":
                    dtype_report = inspect_onnx_node_dtypes(quantized_model_path)
                    unexpected = check_unexpected_fp32_nodes(
                        dtype_report,
                        allowlist=config.quantization.fp32_op_allowlist,
                    )
                    if unexpected:
                        raise ReleaseError(
                            f"{quantized_model_path.name} retained unexpected "
                            "FP32 nodes: " + ", ".join(unexpected)
                        )

                quantized_metadata_path = quantized_model_path.with_suffix(".json")
                copy_detector_metadata(sidecar_path, quantized_metadata_path)
                quantized_sidecar_path = default_quantization_sidecar(quantized_model_path)
                write_quantization_sidecar(
                    quantized_sidecar_path,
                    precision=precision,
                    source_model_sha256=sha256_file(model_path),
                    output_model_sha256=sha256_file(quantized_model_path),
                    framework_commit=commit,
                    checkpoint_revision=config.checkpoint_revision,
                    settings=settings,
                    toolchain=quantization_toolchain_versions(),
                )
                parity_report = build_parity_report(
                    reference_model=model_path,
                    candidate_model=quantized_model_path,
                    metadata_path=sidecar_path,
                    image_path=parity_image,
                    precision=precision,
                    providers=provider_chain(
                        provider_by_precision.get(precision, "CPUExecutionProvider")
                    ),
                )
                provider_used = str(parity_report["provider_used"])
                check_provider_precision_supported(
                    provider_used,
                    precision,
                    quantization_thresholds,
                )
                parity_failures = check_quantized_parity_report(
                    parity_report,
                    quantization_thresholds,
                )
                if int(parity_report["class_mismatch_count"]) > 0:
                    parity_failures.append(
                        f"{spec.branch} {precision} decoded class/count mismatch: "
                        f"{parity_report['class_mismatch_count']}"
                    )
                if parity_failures:
                    raise ReleaseError(
                        f"{quantized_model_path.name} failed quantized parity:\n  "
                        + "\n  ".join(parity_failures)
                    )
                parity_report_path = destination / f"{spec.stem}_{precision}_parity.json"
                parity_report_path.write_text(
                    json.dumps(parity_report, indent=2) + "\n",
                    encoding="utf-8",
                )
                artifacts.append(
                    artifact_entry(
                        quantized_model_path,
                        kind="model",
                        branch=spec.branch,
                        precision=precision,
                        source_model=Path(model_path).name,
                        requires_nms=bool(sidecar["requires_nms"]),
                        post_processing=spec.post_processing,
                        contract=output_contract(sidecar),
                    )
                )
                artifacts.append(
                    artifact_entry(
                        quantized_metadata_path,
                        kind="metadata",
                        branch=spec.branch,
                        precision=precision,
                    )
                )
                artifacts.append(
                    artifact_entry(
                        quantized_sidecar_path,
                        kind="quantization_metadata",
                        branch=spec.branch,
                        precision=precision,
                        source_model=Path(model_path).name,
                    )
                )
                artifacts.append(
                    artifact_entry(
                        parity_report_path,
                        kind="parity_report",
                        branch=spec.branch,
                        precision=precision,
                    )
                )
                expected_accuracy_artifacts.setdefault(
                    coco_report_branch(spec.branch),
                    {},
                )[precision] = {
                    "checkpoint_sha256": sha256_file(quantized_model_path),
                    "metadata_sha256": sha256_file(quantized_metadata_path),
                }
                quantized_benchmark = benchmark_onnx_artifact(
                    model_path=quantized_model_path,
                    metadata_path=quantized_metadata_path,
                    providers=provider_chain(
                        provider_by_precision.get(precision, "CPUExecutionProvider")
                    ),
                    batch_size=1,
                    warmup_iterations=int(benchmark_settings["warmup_iterations"]),
                    measured_iterations=int(benchmark_settings["measured_iterations"]),
                    ort_intra_op_threads=1,
                    ort_inter_op_threads=1,
                )
                check_provider_precision_supported(
                    str(quantized_benchmark["actual_provider"]),
                    precision,
                    quantization_thresholds,
                )
                benchmark_report["branches"].setdefault(spec.branch, {})[precision] = (
                    quantized_benchmark
                )

    if (
        config.quantization is not None
        and quantization_thresholds is not None
        and benchmark_report is not None
    ):
        benchmark_failures = check_quantized_benchmark_report(
            benchmark_report,
            quantization_thresholds,
            required_branches=[spec.branch for spec in config.branches],
        )
        if benchmark_failures:
            raise ReleaseError(
                "quantized benchmark gate failed:\n  " + "\n  ".join(benchmark_failures)
            )
        assert accuracy_report is not None
        accuracy_failures = check_quantized_accuracy_report(
            accuracy_report,
            quantization_thresholds,
            required_branches=[coco_report_branch(spec.branch) for spec in config.branches],
            expected_artifacts=expected_accuracy_artifacts,
        )
        if accuracy_failures:
            raise ReleaseError(
                "quantized COCO accuracy gate failed:\n  " + "\n  ".join(accuracy_failures)
            )
        accuracy_report_path = destination / "quantized_accuracy.json"
        accuracy_markdown_path = destination / "quantized_accuracy.md"
        accuracy_report_path.write_text(
            json.dumps(accuracy_report, indent=2) + "\n",
            encoding="utf-8",
        )
        accuracy_markdown_path.write_text(
            config.quantization.accuracy_markdown.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        artifacts.append(
            artifact_entry(
                accuracy_report_path,
                kind="accuracy_report",
                precision="mixed",
            )
        )
        artifacts.append(
            artifact_entry(
                accuracy_markdown_path,
                kind="accuracy_report_markdown",
                precision="mixed",
            )
        )
        benchmark_json_path = destination / "quantized_benchmarks.json"
        benchmark_markdown_path = destination / "quantized_benchmarks.md"
        benchmark_json_path.write_text(
            json.dumps(benchmark_report, indent=2) + "\n",
            encoding="utf-8",
        )
        benchmark_markdown_path.write_text(
            render_benchmark_report(benchmark_report),
            encoding="utf-8",
        )
        artifacts.append(
            artifact_entry(
                benchmark_json_path,
                kind="benchmark_report",
                precision="mixed",
            )
        )
        artifacts.append(
            artifact_entry(
                benchmark_markdown_path,
                kind="benchmark_report_markdown",
                precision="mixed",
            )
        )

    manifest = build_manifest(
        config,
        artifacts,
        commit=commit,
        toolchain=installed,
    )
    manifest_path = destination / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    problems = verify_manifest(manifest, destination)
    if problems:
        raise ReleaseError("checksum verification failed:\n  " + "\n  ".join(problems))

    (destination / RELEASE_NOTES_NAME).write_text(render_release_notes(manifest))
    print(f"\nManifest: {manifest_path}")
    print(f"Verified {len(manifest['artifacts'])} artifacts")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Pinned release configuration (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dist/onnx"),
        help="Directory receiving the release artifacts (default: %(default)s)",
    )
    parser.add_argument(
        "--verify",
        type=Path,
        default=None,
        help="Verify an existing manifest against the files beside it, then exit",
    )
    parser.add_argument(
        "--expect-commit",
        default=None,
        help="With --verify, require the manifest to record this framework commit",
    )
    parser.add_argument(
        "--allow-toolchain-drift",
        action="store_true",
        help="Warn instead of failing when versions differ from the pin",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.verify is not None:
        manifest = json.loads(args.verify.read_text())
        problems = verify_manifest(
            manifest,
            args.verify.parent,
            expect_commit=args.expect_commit,
        )
        if problems:
            print("Verification FAILED:")
            for problem in problems:
                print(f"  {problem}")
            raise SystemExit(1)
        print(f"Verification PASSED: {len(manifest['artifacts'])} artifacts")
        return

    try:
        build_release(
            load_config(args.config),
            args.output_dir,
            allow_toolchain_drift=args.allow_toolchain_drift,
        )
    except ReleaseError as error:
        print(f"Release ABORTED: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
