#!/usr/bin/env python3
"""Publish and remotely verify the selected reasoning-SFT root bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo, hf_hub_download


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remote_lfs_sha(api: HfApi, repo_id: str, token: str) -> str | None:
    info = api.model_info(repo_id, files_metadata=True, token=token)
    for sibling in info.siblings or []:
        if sibling.rfilename == "model.safetensors" and sibling.lfs is not None:
            lfs = sibling.lfs
            return lfs.sha256 if hasattr(lfs, "sha256") else lfs.get("sha256")
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--repo-id",
        default="AETHORIA-AI/TR-HASH-MoE-200M-160B-Reasoning-SFT",
    )
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required")

    manifest = json.loads((args.bundle / "release_manifest.json").read_text(encoding="utf-8"))
    weights = args.bundle / "model.safetensors"
    local_sha = sha256(weights)
    if manifest.get("weights_dtype") != "float32":
        raise RuntimeError("Reasoning root release must use F32 SafeTensors")
    if local_sha != manifest.get("weights_sha256"):
        raise RuntimeError("Local model.safetensors does not match release manifest")

    api = HfApi(token=token)
    create_repo(args.repo_id, repo_type="model", private=False, token=token, exist_ok=True)
    api.upload_folder(
        folder_path=str(args.bundle),
        repo_id=args.repo_id,
        repo_type="model",
        path_in_repo=".",
        token=token,
        commit_message=f"Promote reasoning SFT step {manifest['selected_step']}",
    )
    required = {
        "README.md",
        "config.json",
        "model.safetensors",
        "model_config.yaml",
        "configuration_tr_hash_moe.py",
        "modeling_tr_hash_moe.py",
        "chat_template.jinja",
        "tokenizer.json",
        "release_manifest.json",
    }
    files = set(api.list_repo_files(args.repo_id, repo_type="model", token=token))
    missing = sorted(required - files)
    if missing:
        raise RuntimeError(f"Remote reasoning release is incomplete: {missing}")
    remote_sha = remote_lfs_sha(api, args.repo_id, token)
    if remote_sha != local_sha:
        raise RuntimeError(f"Remote weights SHA mismatch: {remote_sha} != {local_sha}")

    config_path = Path(
        hf_hub_download(
            args.repo_id,
            "config.json",
            repo_type="model",
            token=token,
            force_download=True,
        )
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "tr_hash_moe":
        raise RuntimeError("Remote config lost the TR-HASH Transformers adapter")
    if config.get("num_experts_per_tok") != 2 or config.get("top_k") != 2:
        raise RuntimeError("Remote config lost deterministic top-2 routing metadata")
    if config.get("torch_dtype") != "float32":
        raise RuntimeError("Remote config does not declare F32 root weights")
    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "selected_step": manifest["selected_step"],
                "remote_weights_sha256": remote_sha,
                "verified_files": sorted(required),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
