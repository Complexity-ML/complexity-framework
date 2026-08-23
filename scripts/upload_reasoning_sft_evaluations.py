#!/usr/bin/env python3
"""Publish audited reasoning-SFT evaluation reports to the model repository."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument(
        "--repo-id",
        default="AETHORIA-AI/TR-HASH-MoE-200M-160B-Reasoning-SFT",
    )
    parser.add_argument("--path-in-repo", default="evaluation/reasoning-sft-500m")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required")
    if not (args.evaluation_root / "summary.json").is_file():
        raise SystemExit("selection summary is missing")

    HfApi(token=token).upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(args.evaluation_root),
        path_in_repo=args.path_in_repo,
        commit_message="Upload reasoning SFT checkpoint evaluations",
        ignore_patterns=[".evaluation_complete"],
    )


if __name__ == "__main__":
    main()
