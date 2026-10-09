"""Read-only configuration and checkpoint inspection for the CLI.

These helpers never construct a model, load tensors, or start training. A
checkpoint integrity report is deliberately separate from a successful restore.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


def load_mapping(path: Path) -> dict[str, Any]:
    """Read a JSON/YAML mapping with errors suitable for command-line users."""
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read configuration {path}: {exc}") from exc
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Configuration must be a non-empty mapping: {path}")
    if not all(isinstance(key, str) for key in data):
        raise ValueError(f"Configuration keys must be strings: {path}")
    return data


def _aliases(data: dict[str, Any], aliases: dict[str, str], warnings: list[str]) -> dict[str, Any]:
    normalized = dict(data)
    for old, new in aliases.items():
        if old not in normalized:
            continue
        value = normalized.pop(old)
        if new in normalized and normalized[new] != value:
            raise ValueError(f"Conflicting fields: {old} and {new}")
        normalized[new] = value
        warnings.append(f"Legacy field {old}; use {new} in new configurations")
    return normalized


def inspect_config(path: Path) -> dict[str, Any]:
    """Validate authored model/training configs while accepting old CLI aliases."""
    from complexity.config import ModelConfig

    raw = load_mapping(path)
    model_fields = {field.name for field in dataclasses.fields(ModelConfig)}
    if "run" in raw or ("model" not in raw and not set(raw).intersection(model_fields)):
        raise ValueError(
            "This is not a ModelConfig or a model/training template. "
            "Run configurations belong to their training script's --config parser; "
            "FSDP recipes can be inspected with complexity train inspect recipe.json."
        )
    warnings: list[str] = []
    model = raw.get("model", raw)
    if not isinstance(model, dict):
        raise ValueError("model section must be a mapping")
    model = _aliases(
        model,
        {
            "num_layers": "num_hidden_layers",
            "num_heads": "num_attention_heads",
            "num_kv_heads": "num_key_value_heads",
            "max_seq_length": "max_position_embeddings",
        },
        warnings,
    )
    # The old CLI template used a label and an FFN multiplier rather than the
    # actual ModelConfig fields. Keep those templates inspectable, explicitly.
    model.pop("name", None)
    if "ff_multiplier" in model:
        multiplier = model.pop("ff_multiplier")
        if (
            not isinstance(multiplier, (int, float))
            or isinstance(multiplier, bool)
            or not math.isfinite(multiplier)
            or multiplier <= 0
        ):
            raise ValueError("ff_multiplier must be a positive number")
        intermediate = model.get("hidden_size", 0) * multiplier
        if "intermediate_size" in model and model["intermediate_size"] != intermediate:
            raise ValueError("Conflicting fields: ff_multiplier and intermediate_size")
        if intermediate != int(intermediate):
            raise ValueError("ff_multiplier must produce an integer intermediate_size")
        model["intermediate_size"] = int(intermediate)
        warnings.append("Legacy field ff_multiplier; use intermediate_size in new configurations")
    if "use_sliding_window" in model:
        enabled = model.pop("use_sliding_window")
        if type(enabled) is not bool:
            raise ValueError("use_sliding_window must be a boolean")
        if enabled and not model.get("sliding_window"):
            raise ValueError("use_sliding_window requires a positive sliding_window")
        if not enabled and model.get("sliding_window") is not None:
            raise ValueError("Conflicting fields: use_sliding_window and sliding_window")
        warnings.append("Legacy field use_sliding_window; use sliding_window in new configurations")
    config = ModelConfig.from_dict(model, strict=True)
    report: dict[str, Any] = {
        "format": "training_template" if "training" in raw else "model_config",
        "path": str(path),
        "model": config.to_dict(),
        "warnings": warnings,
        "training_started": False,
        "validation_scope": "ModelConfig and optional TrainingConfig fields",
    }
    if "model" in raw:
        unknown = set(raw) - {"model", "training", "data", "checkpoint", "logging"}
        if unknown:
            raise ValueError(f"Unknown configuration sections: {', '.join(sorted(unknown))}")
        for section in ("training", "data", "checkpoint", "logging"):
            if section in raw and not isinstance(raw[section], dict):
                raise ValueError(f"{section} section must be a mapping")
    if "training" in raw:
        from complexity.training.config import TrainingConfig

        training = _aliases(
            raw["training"],
            {
                "mixed_precision": "precision",
                "gradient_clip": "grad_clip",
            },
            warnings,
        )
        valid = {field.name for field in dataclasses.fields(TrainingConfig)}
        unknown = set(training) - valid
        if unknown:
            raise ValueError(f"Unknown training fields: {', '.join(sorted(unknown))}")
        for field in dataclasses.fields(TrainingConfig):
            if field.name not in training:
                continue
            value = training[field.name]
            if field.type is int and type(value) is not int:
                raise ValueError(f"training.{field.name} must be an integer")
            if field.type is float and (
                type(value) not in (int, float) or not math.isfinite(value)
            ):
                raise ValueError(f"training.{field.name} must be a finite number")
            if field.type is bool and type(value) is not bool:
                raise ValueError(f"training.{field.name} must be a boolean")
        training_config = TrainingConfig(**training)
        report["training"] = dataclasses.asdict(training_config)
        warnings.append(
            "Template validation does not construct dataloaders. Train with an explicit "
            "Trainer or the script that owns your experiment."
        )
    if any(section in raw for section in ("data", "checkpoint", "logging")):
        warnings.append(
            "data, checkpoint and logging mappings are script-owned metadata; their "
            "paths and behavior were not validated."
        )
    return report


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(data: dict[str, Any], key: str, *, allow_zero: bool = False) -> int:
    value = data.get(key)
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(f"{key} must be {'non-negative' if allow_zero else 'positive'} integer")
    return value


def inspect_recipe(path: Path) -> dict[str, Any]:
    """Describe the pinned FSDP recipe and the evidence available locally."""
    path = path / "recipe.json" if path.is_dir() else path
    recipe = load_mapping(path)
    world = _positive_int(recipe, "world_size")
    batch = _positive_int(recipe, "microbatch")
    context = _positive_int(recipe, "context")
    updates = _positive_int(recipe, "max_updates")
    tokens_per_update = world * batch * context
    warnings = [
        "Recipe inspection does not validate hardware, restore a checkpoint, or start training."
    ]
    assets: dict[str, Any] = {}
    for filename, key in (("model.json", "model_sha256"), ("data-plan.json", "data_plan_sha256")):
        asset = path.parent / filename
        expected = recipe.get(key)
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected)
        ):
            raise ValueError(f"{key} must be a lowercase SHA-256 digest")
        if not asset.is_file():
            assets[filename] = {"present": False, "sha256_matches": None}
            warnings.append(f"Missing linked asset: {asset}; asset identity remains unverified")
            continue
        matches = sha256(asset) == expected
        assets[filename] = {"present": True, "sha256_matches": matches}
        if not matches:
            raise ValueError(f"Recipe hash mismatch: {asset}")
    plan_path = path.parent / "data-plan.json"
    target_tokens = None
    if plan_path.is_file():
        plan = load_mapping(plan_path)
        if plan.get("tokens_per_update") != tokens_per_update or plan.get("updates") != updates:
            raise ValueError("Recipe and data-plan disagree on update count or tokens per update")
        if plan.get("scheduled_tokens") != updates * tokens_per_update:
            raise ValueError("data-plan scheduled_tokens does not match recipe arithmetic")
        target_tokens = plan.get("target_tokens_requested")
    return {
        "format": "fsdp_recipe",
        "path": str(path),
        "model_repo": recipe.get("model_repo"),
        "world_size": world,
        "microbatch": batch,
        "context": context,
        "max_updates": updates,
        "tokens_per_update": tokens_per_update,
        "scheduled_tokens": updates * tokens_per_update,
        "target_tokens_requested": target_tokens,
        "recipe_sha256": sha256(path),
        "assets": assets,
        "warnings": warnings,
        "training_started": False,
    }


def inspect_checkpoint(
    path: Path, *, verify: bool = False, recipe_path: Path | None = None
) -> dict[str, Any]:
    """Inspect a TR-HASH FSDP checkpoint using only JSON and ordinary files."""
    if not path.is_dir():
        raise ValueError(f"Checkpoint directory not found: {path}")
    manifest = load_mapping(path / "complete.json")
    state = load_mapping(path / "state.json")
    step = _positive_int(state, "step", allow_zero=True)
    world = _positive_int(state, "world_size")
    next_update = _positive_int(state, "next_update", allow_zero=True)
    tokens = _positive_int(state, "tokens_seen", allow_zero=True)
    if manifest.get("step") != step or next_update != step:
        raise ValueError("Checkpoint manifest/state disagree on step or next_update")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Checkpoint manifest files must be a non-empty mapping")
    required = {"state.json", "config.json"} | {f"rng_rank{rank}.pt" for rank in range(world)}
    if not required.issubset(files):
        raise ValueError(f"Missing manifest entries: {', '.join(sorted(required - files.keys()))}")
    if not any(name in files for name in ("distributed/.metadata", "distributed/metadata.json")):
        raise ValueError("Checkpoint manifest is missing distributed metadata")
    if not any(isinstance(name, str) and name.endswith(".distcp") for name in files):
        raise ValueError("Checkpoint manifest is missing distributed tensor shards")
    total_bytes = 0
    root = path.resolve()
    for name, identity in files.items():
        if not isinstance(name, str) or not name or "\\" in name:
            raise ValueError(f"Unsafe checkpoint manifest path: {name!r}")
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
            raise ValueError(f"Unsafe checkpoint manifest path: {name!r}")
        file = path / name
        if not file.resolve().is_relative_to(root):
            raise ValueError(f"Checkpoint manifest path escapes directory: {name}")
        if not isinstance(identity, dict):
            raise ValueError(f"Invalid manifest identity: {name}")
        expected_bytes = _positive_int(identity, "bytes", allow_zero=True)
        expected_sha = identity.get("sha256")
        if (
            not isinstance(expected_sha, str)
            or len(expected_sha) != 64
            or any(c not in "0123456789abcdef" for c in expected_sha)
        ):
            raise ValueError(f"Invalid SHA-256 in manifest: {name}")
        if not file.is_file() or file.stat().st_size != expected_bytes:
            raise ValueError(f"Missing file or byte-size mismatch: {name}")
        if verify and sha256(file) != expected_sha:
            raise ValueError(f"Checksum mismatch: {name}")
        total_bytes += expected_bytes
    recipe_matches = None
    if recipe_path is not None:
        recipe_path = recipe_path / "recipe.json" if recipe_path.is_dir() else recipe_path
        recipe = load_mapping(recipe_path)
        recipe_matches = sha256(recipe_path) == state.get("recipe_sha256")
        if not recipe_matches:
            raise ValueError("Checkpoint recipe_sha256 differs from the supplied recipe")
        if recipe.get("world_size") != world:
            raise ValueError("Checkpoint world_size differs from the supplied recipe")
        unit = (
            _positive_int(recipe, "world_size")
            * _positive_int(recipe, "microbatch")
            * _positive_int(recipe, "context")
        )
        if tokens != step * unit or step > _positive_int(recipe, "max_updates"):
            raise ValueError("Checkpoint progress is inconsistent with the supplied recipe")
        if state.get("data_plan_sha256") != recipe.get("data_plan_sha256"):
            raise ValueError("Checkpoint data-plan identity differs from the supplied recipe")
    return {
        "format": "tr_hash_fsdp_checkpoint",
        "path": str(path),
        "step": step,
        "next_update": next_update,
        "tokens_seen": tokens,
        "world_size": world,
        "manifest_files": len(files),
        "manifest_bytes": total_bytes,
        "file_sizes_verified": True,
        "sha256_verified": verify,
        "recipe_matches": recipe_matches,
        "checkpoint_restored": False,
        "verification_scope": "manifest-listed file sizes"
        + (" and SHA-256 hashes" if verify else ""),
        "warnings": [
            "No tensors or RNG state were loaded. Resume must use the original FSDP training script.",
            "A successful restore, optimizer/RNG reload and bounded advancement remain unverified.",
        ],
    }
