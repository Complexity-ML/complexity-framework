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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

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
                None if entry.get("checkpoint_subdir") is None
                else str(entry["checkpoint_subdir"])
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

    return ReleaseConfig(
        checkpoint_repo=str(values["checkpoint_repo"]),
        checkpoint_revision=revision,
        opset=int(values["opset"]),
        parity_num_tests=int(values.get("parity_num_tests", 5)),
        toolchain={str(k): str(v) for k, v in toolchain.items()},
        branches=branches,
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
            problems.append(
                f"framework_commit {recorded or 'missing'}, expected {expect_commit}"
            )
    for artifact in manifest["artifacts"]:
        path = Path(directory) / str(artifact["name"])
        if not path.is_file():
            problems.append(f"{artifact['name']}: missing")
            continue
        actual_size = path.stat().st_size
        if actual_size != int(artifact["size_bytes"]):
            problems.append(
                f"{artifact['name']}: size {actual_size}, "
                f"manifest {artifact['size_bytes']}"
            )
        actual_digest = sha256_file(path)
        if actual_digest != str(artifact["sha256"]):
            problems.append(
                f"{artifact['name']}: sha256 {actual_digest}, "
                f"manifest {artifact['sha256']}"
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
        "| Branch | Model | Post-processing | Size | SHA-256 |",
        "|---|---|---|---:|---|",
    ]
    for artifact in manifest["artifacts"]:
        if artifact.get("kind") != "model":
            continue
        lines.append(
            f"| {artifact['branch']} | `{artifact['name']}` | "
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


def build_release(
    config: ReleaseConfig,
    output_dir: Path,
    *,
    allow_toolchain_drift: bool = False,
) -> dict[str, Any]:
    """Download, export, gate and describe the release. Raises on any failure."""

    from huggingface_hub import snapshot_download

    from scripts.check_onnx_parity import check_parity
    from scripts.export_onnx import export_onnx

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

    print(f"Downloading {config.checkpoint_repo}@{config.checkpoint_revision}")
    checkpoint_root = Path(
        snapshot_download(
            repo_id=config.checkpoint_repo,
            revision=config.checkpoint_revision,
        )
    )

    artifacts: list[dict[str, Any]] = []
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
                requires_nms=bool(sidecar["requires_nms"]),
                post_processing=spec.post_processing,
                contract=output_contract(sidecar),
            )
        )
        artifacts.append(
            artifact_entry(sidecar_path, kind="metadata", branch=spec.branch)
        )

    manifest = build_manifest(
        config,
        artifacts,
        commit=framework_commit(),
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
