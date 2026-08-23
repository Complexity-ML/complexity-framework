#!/usr/bin/env python3
"""Publish a packaged TR-HASH 32,004 SFT dataset and verify remote hashes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from huggingface_hub import (
    CommitOperationDelete,
    HfApi,
    create_repo,
    get_token,
    hf_hub_download,
)

from scripts.package_tr_hash_sft_32004_release import TOKENIZED_SUBDIR
from scripts.tokenize_tr_hash_200m_clean_sft_v2 import sha256

LARGE_RELEASE_FILES = (
    Path("train.jsonl"),
    Path("eval.jsonl"),
    TOKENIZED_SUBDIR / "train/input_ids.bin",
    TOKENIZED_SUBDIR / "train/labels.bin",
    TOKENIZED_SUBDIR / "train/examples.jsonl",
    TOKENIZED_SUBDIR / "eval/input_ids.bin",
    TOKENIZED_SUBDIR / "eval/labels.bin",
    TOKENIZED_SUBDIR / "eval/examples.jsonl",
)
SMALL_RELEASE_FILES = (
    Path("README.md"),
    Path("manifest.json"),
    Path("metadata/recompile-recipe.json"),
    Path("metadata/source-manifest.json"),
    Path("metadata/release-audit.json"),
    TOKENIZED_SUBDIR / "manifest.json",
    TOKENIZED_SUBDIR / "chat_template.json",
    TOKENIZED_SUBDIR / "train/sft.idx.json",
    TOKENIZED_SUBDIR / "eval/sft.idx.json",
    TOKENIZED_SUBDIR / "tokenizer/tokenizer.json",
    TOKENIZED_SUBDIR / "tokenizer/tokenizer_config.json",
    TOKENIZED_SUBDIR / "tokenizer/special_tokens_map.json",
    TOKENIZED_SUBDIR / "tokenizer/config.json",
    TOKENIZED_SUBDIR / "tokenizer/chat_template.jinja",
    Path("tokenizer/tokenizer.json"),
    Path("tokenizer/tokenizer_config.json"),
    Path("tokenizer/special_tokens_map.json"),
    Path("tokenizer/config.json"),
    Path("tokenizer/chat_template.jinja"),
)
STALE_RELEASE_PATHS = (
    "tokenized/tr-hash-32k-v2-2048",
    "recipe.json",
    "metadata/recipe.json",
    "metadata/quality-audit.json",
)


def validate_local_release(dataset: Path) -> dict[str, Any]:
    audit = json.loads((dataset / "metadata/release-audit.json").read_text(encoding="utf-8"))
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if audit.get("status") != "passed":
        raise ValueError("local release audit did not pass")
    if int(audit.get("tokenizer_vocab_size", 0)) != 32_004:
        raise ValueError("local release does not use vocab 32,004")
    missing = [
        str(path)
        for path in (*LARGE_RELEASE_FILES, *SMALL_RELEASE_FILES)
        if not (dataset / path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing release files: {missing}")
    return {"audit": audit, "manifest": manifest}


def _remote_blob_sha(api: HfApi, repo_id: str, filename: str, token: str) -> str | None:
    info = api.dataset_info(repo_id, files_metadata=True, token=token)
    for sibling in info.siblings or []:
        if sibling.rfilename != filename:
            continue
        lfs = sibling.lfs
        if lfs is None:
            return None
        return lfs.sha256 if hasattr(lfs, "sha256") else lfs.get("sha256")
    return None


def _verify_remote_file(
    api: HfApi,
    *,
    repo_id: str,
    dataset: Path,
    relative: Path,
    token: str,
) -> tuple[str, str]:
    local_hash = sha256(dataset / relative)
    remote_hash = _remote_blob_sha(api, repo_id, str(relative), token)
    if remote_hash is None:
        downloaded = Path(
            hf_hub_download(
                repo_id,
                str(relative),
                repo_type="dataset",
                token=token,
                force_download=True,
            )
        )
        remote_hash = sha256(downloaded)
    if remote_hash != local_hash:
        raise RuntimeError(
            f"remote hash mismatch for {relative}: local={local_hash} remote={remote_hash}"
        )
    return local_hash, remote_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args()
    validation = validate_local_release(args.dataset)
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise SystemExit("Hugging Face authentication is required")
    api = HfApi(token=token)
    create_repo(
        args.repo_id,
        repo_type="dataset",
        private=False,
        exist_ok=True,
        token=token,
    )
    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(args.dataset),
        ignore_patterns=[
            ".DS_Store",
            "**/.DS_Store",
            "**/*.partial",
            "**/.tokenize_state.json",
            "**/.cache/**",
        ],
    )
    remote_files = set(api.list_repo_files(args.repo_id, repo_type="dataset", token=token))
    stale = [
        path
        for path in STALE_RELEASE_PATHS
        if path in remote_files
        or any(filename.startswith(path.rstrip("/") + "/") for filename in remote_files)
    ]
    if stale:
        api.create_commit(
            repo_id=args.repo_id,
            repo_type="dataset",
            operations=[CommitOperationDelete(path_in_repo=path) for path in stale],
            commit_message="Remove superseded 32,000-token SFT artifacts",
            token=token,
        )
    required = {str(path) for path in (*LARGE_RELEASE_FILES, *SMALL_RELEASE_FILES)}
    remote_files = set(api.list_repo_files(args.repo_id, repo_type="dataset", token=token))
    missing = sorted(required - remote_files)
    if missing:
        raise RuntimeError(f"remote 32,004 release is incomplete: {missing}")
    forbidden = sorted(
        filename
        for filename in remote_files
        if filename.startswith("tokenized/tr-hash-32k-v2-2048/")
    )
    if forbidden:
        raise RuntimeError(f"legacy 32,000 tokenized files remain: {forbidden}")
    verified: dict[str, str] = {}
    for relative in (*LARGE_RELEASE_FILES, *SMALL_RELEASE_FILES):
        local_hash, _ = _verify_remote_file(
            api,
            repo_id=args.repo_id,
            dataset=args.dataset,
            relative=relative,
            token=token,
        )
        verified[str(relative)] = local_hash
    info = api.dataset_info(args.repo_id, token=token)
    report = {
        "repo_id": args.repo_id,
        "revision": info.sha,
        "tokenizer_vocab_size": 32_004,
        "train_examples": validation["manifest"]["train_examples"],
        "eval_examples": validation["manifest"]["eval_examples"],
        "verified_sha256": verified,
        "legacy_tokenized_folder_removed": True,
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
