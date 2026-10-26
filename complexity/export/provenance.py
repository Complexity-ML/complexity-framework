"""Collect model-only artifacts with checkpoint and run provenance.

Inputs are local and explicit. This operation neither publishes artifacts nor
reconstructs distributed checkpoints. A recorded optimizer checkpoint remains
outside the collected model bundle and is required separately for exact resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .transformers import TOKENIZER_FILES

DEFAULT_RUN_FILES = ("run_config.json", "metrics.csv")
DEFAULT_DATA_FILES = ("dataset_manifest.json", "train/tokens.idx.json", "eval/tokens.idx.json")
METADATA_SUFFIXES = {".json", ".jsonl", ".csv", ".log", ".yaml", ".yml", ".toml"}


def sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _relative_file(root: Path, name: str, *, metadata: bool = False) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Expected a contained relative file path: {name!r}")
    if metadata and relative.suffix not in METADATA_SUFFIXES:
        raise ValueError(f"Expected text metadata, not a checkpoint/token binary: {name!r}")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"File escapes its source directory: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _metadata_files(root: Path, names: Sequence[str]) -> list[tuple[Path, Path]]:
    if (
        not isinstance(names, Sequence)
        or isinstance(names, (str, bytes))
        or not names
        or any(not isinstance(name, str) for name in names)
        or len(names) != len(set(names))
    ):
        raise ValueError("Metadata files must be a nonempty list of unique relative names")
    return [(_relative_file(root, name, metadata=True), Path(name)) for name in names]


def _checkpoint_directory(path: Path) -> Path:
    if path.name != "latest":
        return path
    root = path.parent.resolve()
    if path.is_symlink():
        resolved = path.resolve()
        if not resolved.is_relative_to(root) or not resolved.is_dir():
            raise ValueError("latest must point to a checkpoint inside its root")
        return resolved
    if path.is_file():
        target = path.read_text(encoding="utf-8").strip()
        relative = Path(target)
        if (
            not target
            or relative.is_absolute()
            or len(relative.parts) != 1
            or target in {".", ".."}
        ):
            raise ValueError("latest must name one checkpoint directory inside its root")
        resolved = root / relative
        if not resolved.resolve().is_relative_to(root) or not resolved.is_dir():
            raise ValueError("latest points outside its root or to a missing checkpoint")
        return resolved
    raise FileNotFoundError(path)


def _source_model(checkpoint: Path) -> tuple[Path, dict, int | None, dict | None]:
    """Restrict PyTorch deserialization; native SafeTensors are copied unchanged."""
    state_file = checkpoint / "checkpoint.pt"
    if state_file.is_file():
        state_file = _relative_file(checkpoint, "checkpoint.pt")
        import torch

        try:
            state = torch.load(state_file, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError as exc:
            raise ValueError(
                "Checkpoint is not readable with restricted weights-only loading"
            ) from exc
        if not isinstance(state, Mapping):
            raise ValueError("Checkpoint must contain a model state dictionary")
        model = state.get("model")
        if not isinstance(model, Mapping) or not model:
            raise ValueError("Checkpoint contains no model state dictionary")
        if any(
            not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
            for name, tensor in model.items()
        ):
            raise ValueError("Model state must map tensor names to tensors only")
        config = state.get("model_config", state.get("config"))
        config_file = checkpoint / "config.json"
        controls_file = checkpoint / "runtime_mlp_controls.json"
        if "runtime_mlp_controls" in state or controls_file.is_file():
            # Validate the saved canonical recipe before copying its config.
            # Removed route weights/counts/scales fail strict validation;
            # collection never translates them into another architecture.
            if controls_file.is_file():
                _relative_file(checkpoint, controls_file.name)
            from complexity.utils.checkpointing import _read_controls

            controls = _read_controls(checkpoint, dict(state))
            config = controls["model_config"]
        elif config_file.is_file():
            config = _json_object(_relative_file(checkpoint, "config.json"))
        elif config is None:
            # Native SFT keeps the architecture in its immutable run identity.
            # Only accept the exact mapping path, never infer a shape from weights.
            identity = state.get("identity")
            if isinstance(identity, Mapping):
                plan = identity.get("plan")
                if isinstance(plan, Mapping):
                    config = plan.get("model_config")
        if not isinstance(config, Mapping) or not config:
            raise ValueError("Checkpoint must provide its model configuration")
        step = state.get("step")
        if step is None and checkpoint.name.startswith("step_"):
            try:
                step = int(checkpoint.name.removeprefix("step_"))
            except ValueError:
                pass
        if step is not None and (type(step) is not int or step < 0):
            raise ValueError("Checkpoint step must be a nonnegative integer")
        return state_file, dict(config), step, dict(model)

    weights = checkpoint / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(
            f"Expected checkpoint.pt or native model.safetensors in {checkpoint}"
        )
    weights = _relative_file(checkpoint, "model.safetensors")
    from safetensors import safe_open

    with safe_open(weights, framework="pt", device="cpu") as handle:
        if not list(handle.keys()):
            raise ValueError("Native weights contain no tensors")
    config = _json_object(_relative_file(checkpoint, "config.json"))
    if not config:
        raise ValueError("Native weights must provide their model configuration")
    return weights, config, None, None


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def collect_run_artifacts(
    *,
    run_name: str,
    checkpoint_root: str | Path,
    run_dir: str | Path,
    data_root: str | Path,
    tokenizer: str | Path,
    output_dir: str | Path,
    log: str | Path | None = None,
    run_files: Sequence[str] = DEFAULT_RUN_FILES,
    data_files: Sequence[str] = DEFAULT_DATA_FILES,
) -> Path:
    """Write a fresh local model bundle and an integrity manifest.

    Metadata lists allow different runner/dataset layouts. Only contained text
    metadata and allowlisted tokenizer assets are copied. Legacy checkpoints
    use restricted ``torch.load(weights_only=True)`` with no pickle fallback;
    optimizer/RNG state is never included. Native SafeTensors bytes are kept.
    """
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("run_name must be nonempty")
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Artifact output already exists: {output}")
    output = output.resolve()
    checkpoint = _checkpoint_directory(Path(checkpoint_root)).resolve()
    source_weights = checkpoint / (
        "checkpoint.pt" if (checkpoint / "checkpoint.pt").is_file() else "model.safetensors"
    )
    source_weights = _relative_file(checkpoint, source_weights.name)
    source_hash = sha256(source_weights)
    source_bytes = source_weights.stat().st_size
    source_weights, config, step, model = _source_model(checkpoint)
    native = model is None
    # Ensure metadata is JSON-serializable before creating any output.
    json.dumps(config, allow_nan=False)
    copies = _metadata_files(Path(run_dir), run_files)
    copies.extend(
        (source, Path("dataset") / name)
        for source, name in _metadata_files(Path(data_root), data_files)
    )
    tokenizer_root = Path(tokenizer)
    tokenizer_names = sorted(TOKENIZER_FILES | {"chat_template.json"})
    tokenizer_assets = [
        (_relative_file(tokenizer_root, name), Path("tokenizer") / name)
        for name in tokenizer_names
        if (tokenizer_root / name).exists() or (tokenizer_root / name).is_symlink()
    ]
    if not tokenizer_assets:
        raise FileNotFoundError(f"No known tokenizer assets found in {tokenizer_root}")
    copies.extend(tokenizer_assets)
    if log is not None:
        if not Path(log).is_file():
            raise FileNotFoundError(log)
        copies.append((Path(log), Path("training.log")))
    reserved = {
        Path("config.json"),
        Path("model_config.json"),
        Path("model.safetensors"),
        Path("artifact_manifest.json"),
    }
    destinations = [destination for _, destination in copies]
    if reserved.intersection(destinations) or len(destinations) != len(set(destinations)):
        raise ValueError("Metadata destinations collide with exported model or manifest files")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        if model is None:
            shutil.copy2(source_weights, staging / "model.safetensors")
        else:
            from safetensors.torch import save_file

            tensors = {
                name: value.detach().cpu().contiguous().clone() for name, value in model.items()
            }
            save_file(tensors, staging / "model.safetensors", metadata={"format": "pt"})
            del tensors, model
        _write_json(staging / "config.json", config)
        _write_json(
            staging / "model_config.json",
            {"run_name": run_name, "checkpoint_step": step, "model": config},
        )
        for source, destination in copies:
            target = staging / destination
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        if sha256(source_weights) != source_hash:
            raise RuntimeError("Source checkpoint changed during collection")
        manifest = {
            "schema_version": 1,
            "format": "complexity-run-artifacts-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_name": run_name,
            "model_only": True,
            "resume_supported": False,
            "source_format": "native-weights" if native else "local-training-checkpoint",
            "checkpoint": {
                "path": str(source_weights.resolve()),
                "bytes": source_bytes,
                "sha256": source_hash,
                "included_in_release": native,
                "note": (
                    "Original native weights have no optimizer-state resume data."
                    if native
                    else "Keep training checkpoint separately for exact optimizer-state resume."
                ),
            },
            "files": [
                {
                    "path": str(path.relative_to(staging)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in sorted(staging.rglob("*"))
                if path.is_file()
            ],
        }
        _write_json(staging / "artifact_manifest.json", manifest)
        # Reserve the destination exclusively, then publish the complete stage.
        output.mkdir()
        try:
            os.replace(staging, output)
        except BaseException:
            output.rmdir()
            raise
    return output / "artifact_manifest.json"


def collect_from_config(path: str | Path) -> Path:
    """Resolve an explicit JSON collection config relative to its own file."""
    path = Path(path).resolve()
    config = _json_object(path)
    if config.pop("format", None) != "complexity-artifact-collection-v1":
        raise ValueError("Collection config requires format='complexity-artifact-collection-v1'")
    required = {"run_name", "checkpoint_root", "run_dir", "data_root", "tokenizer", "output_dir"}
    allowed = required | {"log", "run_files", "data_files"}
    if missing := required - config.keys():
        raise ValueError(f"Missing collection keys: {sorted(missing)}")
    if unknown := config.keys() - allowed:
        raise ValueError(f"Unknown collection keys: {sorted(unknown)}")
    for name in (required - {"run_name"}) | {"log"}:
        if name in config:
            if not isinstance(config[name], str) or not config[name]:
                raise ValueError(f"{name} must be a nonempty local path")
            config[name] = path.parent / config[name]
    return collect_run_artifacts(**config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    for name in (
        "run-name",
        "checkpoint-root",
        "run-dir",
        "data-root",
        "tokenizer",
        "output-dir",
        "log",
    ):
        parser.add_argument(f"--{name}", type=str if name == "run-name" else Path)
    args = vars(parser.parse_args())
    config = args.pop("config")
    try:
        if config is not None:
            if any(value is not None for value in args.values()):
                raise ValueError("Choose --config or explicit collection arguments")
            manifest = collect_from_config(config)
        else:
            required = set(args) - {"log"}
            if missing := {key for key in required if args[key] is None}:
                raise ValueError(f"Missing collection arguments: {sorted(missing)}")
            manifest = collect_run_artifacts(**args)
    except (ValueError, TypeError, RuntimeError, OSError) as exc:
        parser.exit(2, f"Collection failed: {exc}\n")
    print(manifest)
    print(manifest.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
