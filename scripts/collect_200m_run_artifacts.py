#!/usr/bin/env python3
"""Collect a finished 200M run into a small, verifiable release directory.

The full training checkpoint (including optimizer state) stays under the
checkpoint root so it can be downloaded separately for an exact resume. This
script exports the model weights to safetensors and copies the run metrics,
configuration, tokenizer, and immutable dataset metadata needed for evaluation
and provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import save_file

from complexity.utils.local_checkpoint import CHECKPOINT_FILE, resolve_checkpoint_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    return parser


def sha256(path: Path, chunk_size: int = 64 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def copy_required(source: Path, destination: Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def main() -> None:
    args = build_parser().parse_args()
    checkpoint_dir = resolve_checkpoint_path(args.checkpoint_root)
    checkpoint_path = checkpoint_dir / CHECKPOINT_FILE
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Artifact output already exists: {output_dir}. Remove it explicitly to rebuild."
        )
    output_dir.mkdir(parents=True)

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = state.get("model")
    if not isinstance(model_state, dict) or not model_state:
        raise ValueError(f"No model state dictionary found in {checkpoint_path}")
    model_config = state.get("config")
    checkpoint_step = int(state.get("step", checkpoint_dir.name.split("_", 1)[-1]))

    # The model ties its embedding and LM-head weights. Safetensors requires
    # independent storage for duplicate keys, so clone each exported tensor.
    export_state = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in model_state.items()
    }
    weights_path = output_dir / "model.safetensors"
    save_file(export_state, weights_path)
    del export_state
    del state

    model_config_path = output_dir / "model_config.json"
    model_config_path.write_text(
        json.dumps(
            {
                "run_name": args.run_name,
                "checkpoint_step": checkpoint_step,
                "model": model_config,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    copy_required(args.run_dir / "run_config.json", output_dir / "run_config.json")
    copy_required(args.run_dir / "metrics.csv", output_dir / "metrics.csv")
    copy_required(
        args.data_root / "dataset_manifest.json",
        output_dir / "dataset" / "dataset_manifest.json",
    )
    for partition in ("train", "eval"):
        copy_required(
            args.data_root / partition / "tokens.idx.json",
            output_dir / "dataset" / partition / "tokens.idx.json",
        )
    for tokenizer_file in sorted(args.tokenizer.glob("*")):
        if tokenizer_file.is_file():
            copy_required(
                tokenizer_file,
                output_dir / "tokenizer" / tokenizer_file.name,
            )
    if args.log is not None:
        copy_required(args.log, output_dir / "training.log")

    release_files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": sha256(checkpoint_path),
            "included_in_release": False,
            "note": "Download separately only when exact optimizer-state resume is required.",
        },
        "files": [
            {
                "path": str(path.relative_to(output_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in release_files
        ],
    }
    manifest_path = output_dir / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
