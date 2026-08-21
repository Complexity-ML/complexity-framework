#!/usr/bin/env python3
"""Publish clean-SFT training metrics and per-epoch evaluation reports."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo

PREFIX = "training/sft-v2-300k"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required")
    for path in (args.evaluation_root, args.metrics, args.panel):
        if not path.exists():
            raise FileNotFoundError(path)
    api = HfApi(token=token)
    create_repo(args.repo_id, repo_type="model", private=False, token=token, exist_ok=True)
    api.upload_folder(
        folder_path=str(args.evaluation_root),
        repo_id=args.repo_id,
        repo_type="model",
        path_in_repo=f"{PREFIX}/evaluations",
        token=token,
        commit_message="Upload SFT v2 per-epoch PIQA and regression evaluations",
    )
    api.upload_file(
        path_or_fileobj=str(args.metrics),
        path_in_repo=f"{PREFIX}/metrics.csv",
        repo_id=args.repo_id,
        repo_type="model",
        token=token,
        commit_message="Upload SFT v2 training metrics",
    )
    api.upload_file(
        path_or_fileobj=str(args.panel),
        path_in_repo=f"{PREFIX}/regression_panel.json",
        repo_id=args.repo_id,
        repo_type="model",
        token=token,
        commit_message="Upload SFT v2 promotion panel",
    )
    files = set(api.list_repo_files(args.repo_id, repo_type="model", token=token))
    required = {
        f"{PREFIX}/evaluations/summary.json",
        f"{PREFIX}/metrics.csv",
        f"{PREFIX}/regression_panel.json",
    }
    missing = sorted(required - files)
    if missing:
        raise RuntimeError(f"Remote evaluation verification failed; missing={missing}")
    print(f"Verified SFT v2 evaluation artifacts in {args.repo_id}", flush=True)


if __name__ == "__main__":
    main()
