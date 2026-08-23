#!/usr/bin/env python3
"""Upload and remotely verify the audited 500M-token reasoning dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo, hf_hub_download

REPO_ID = "AETHORIA-AI/TR-HASH-MoE-200M-Reasoning-SFT-500M"
TOKENIZED = Path("tokenized/tr-hash-32k-v2-2048")
HASHED_FILES = (
    Path("train.jsonl"),
    Path("eval.jsonl"),
    TOKENIZED / "train/input_ids.bin",
    TOKENIZED / "train/labels.bin",
    TOKENIZED / "train/examples.jsonl",
    TOKENIZED / "eval/input_ids.bin",
    TOKENIZED / "eval/labels.bin",
    TOKENIZED / "eval/examples.jsonl",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remote_blob_sha(api: HfApi, repo_id: str, filename: str, token: str) -> str | None:
    info = api.dataset_info(repo_id, files_metadata=True, token=token)
    for sibling in info.siblings or []:
        if sibling.rfilename != filename:
            continue
        if sibling.lfs is not None:
            lfs = sibling.lfs
            return lfs.sha256 if hasattr(lfs, "sha256") else lfs.get("sha256")
        return None
    return None


def validate_local(dataset: Path) -> dict:
    audit = json.loads((dataset / "metadata/release-audit.json").read_text(encoding="utf-8"))
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    tokenized = json.loads((dataset / TOKENIZED / "manifest.json").read_text(encoding="utf-8"))
    if audit.get("status") != "passed":
        raise ValueError("dataset release audit did not pass")
    actual = int(manifest["actual_unique_formatted_tokens"])
    if not 500_000_000 <= actual < 500_020_000:
        raise ValueError(f"invalid unique token count: {actual}")
    if int(tokenized["partitions"]["train"]["num_tokens"]) != actual:
        raise ValueError("raw/tokenized train-token parity failed")
    missing = [str(path) for path in HASHED_FILES if not (dataset / path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing release files: {missing}")
    return {"manifest": manifest, "tokenized": tokenized, "audit": audit}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--repo-id", default=REPO_ID)
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required")
    validation = validate_local(args.dataset)
    local_hashes = {str(path): sha256(args.dataset / path) for path in HASHED_FILES}

    api = HfApi(token=token)
    create_repo(args.repo_id, repo_type="dataset", private=False, token=token, exist_ok=True)
    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(args.dataset),
        ignore_patterns=[
            ".DS_Store",
            "**/.DS_Store",
            "**/*.partial",
            "**/.tokenize_state.json",
            ".build_state.json",
        ],
    )

    files = set(api.list_repo_files(args.repo_id, repo_type="dataset", token=token))
    required = {
        "README.md",
        "manifest.json",
        "metadata/recipe.json",
        "metadata/release-audit.json",
        str(TOKENIZED / "manifest.json"),
        str(TOKENIZED / "tokenizer/tokenizer.json"),
        *(str(path) for path in HASHED_FILES),
    }
    missing = sorted(required - files)
    if missing:
        raise RuntimeError(f"remote reasoning dataset is incomplete: {missing}")

    remote_manifest = Path(
        hf_hub_download(
            args.repo_id,
            "manifest.json",
            repo_type="dataset",
            token=token,
            force_download=True,
        )
    )
    if sha256(remote_manifest) != sha256(args.dataset / "manifest.json"):
        raise RuntimeError("remote raw manifest hash mismatch")
    mismatches = {}
    for filename, local_hash in local_hashes.items():
        remote_hash = remote_blob_sha(api, args.repo_id, filename, token)
        if remote_hash is None:
            remote_path = Path(
                hf_hub_download(
                    args.repo_id,
                    filename,
                    repo_type="dataset",
                    token=token,
                    force_download=True,
                )
            )
            remote_hash = sha256(remote_path)
        if remote_hash != local_hash:
            mismatches[filename] = {"local": local_hash, "remote": remote_hash}
    if mismatches:
        raise RuntimeError(f"remote dataset hashes differ: {mismatches}")

    info = api.dataset_info(args.repo_id, token=token)
    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "revision": info.sha,
                "unique_formatted_tokens": validation["manifest"]["actual_unique_formatted_tokens"],
                "verified_files": sorted(required),
                "verified_sha256": local_hashes,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
