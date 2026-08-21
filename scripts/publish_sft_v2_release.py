#!/usr/bin/env python3
"""Publish and remotely verify the promoted clean-SFT v2 root bundle."""

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


def remote_lfs_sha(api: HfApi, repo_id: str, filename: str, token: str) -> str | None:
    info = api.model_info(repo_id, files_metadata=True, token=token)
    for sibling in info.siblings or []:
        if sibling.rfilename != filename or sibling.lfs is None:
            continue
        lfs = sibling.lfs
        return lfs.sha256 if hasattr(lfs, "sha256") else lfs.get("sha256")
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required")
    manifest_path = args.bundle / "release_manifest.json"
    weights_path = args.bundle / "model.safetensors"
    config_path = args.bundle / "config.json"
    for path in (manifest_path, weights_path, config_path, args.bundle / "README.md"):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("weights_dtype") not in {"float32", "bfloat16"}:
        raise RuntimeError("Root release must declare float32 or bfloat16 weights")
    local_weights_sha = sha256(weights_path)
    if local_weights_sha != manifest["weights_sha256"]:
        raise RuntimeError("Local release manifest does not match model.safetensors")

    api = HfApi(token=token)
    create_repo(args.repo_id, repo_type="model", private=False, token=token, exist_ok=True)
    api.upload_folder(
        folder_path=str(args.bundle),
        repo_id=args.repo_id,
        repo_type="model",
        path_in_repo=".",
        token=token,
        commit_message=(
            f"Promote clean full-SFT v2 epoch {manifest['selected_epoch']} "
            f"step {manifest['selected_step']}"
        ),
    )

    files = set(api.list_repo_files(args.repo_id, repo_type="model", token=token))
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
    missing = sorted(required - files)
    if missing:
        raise RuntimeError(f"Remote root release is incomplete: {missing}")
    remote_sha = remote_lfs_sha(api, args.repo_id, "model.safetensors", token)
    if remote_sha != local_weights_sha:
        raise RuntimeError(f"Remote weights SHA256 mismatch: {remote_sha} != {local_weights_sha}")

    remote_config = Path(
        hf_hub_download(
            args.repo_id,
            "config.json",
            repo_type="model",
            token=token,
            force_download=True,
        )
    )
    config = json.loads(remote_config.read_text(encoding="utf-8"))
    if config.get("model_type") != "tr_hash_moe":
        raise RuntimeError("Remote config is not Transformers-recognizable TR-HASH")
    if config.get("top_k") != 2 or config.get("num_experts_per_tok") != 2:
        raise RuntimeError("Remote config lost native/Transformers top-2 routing metadata")
    if config.get("torch_dtype") != manifest["weights_dtype"]:
        raise RuntimeError(
            "Remote torch_dtype does not match the promoted SafeTensors precision"
        )
    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "selected_epoch": manifest["selected_epoch"],
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
