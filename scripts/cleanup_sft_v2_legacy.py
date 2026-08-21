#!/usr/bin/env python3
"""Remove superseded SFT-v1 artifacts after a verified SFT-v2 promotion."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import CommitOperationDelete, HfApi


LEGACY_PREFIXES = (
    "step_000463/",
    "step_000926/",
    "step_001389/",
    "reports/piqa/",
    "reports/training/",
)
ROOT_REQUIRED = {
    "README.md",
    "config.json",
    "model.safetensors",
    "release_manifest.json",
    "tokenizer.json",
}


def cleanup_plan(files: set[str], selected_step: int) -> list[str]:
    checkpoint = f"training/sft-v2-300k/checkpoints/step_{selected_step:06d}/checkpoint.pt"
    missing = sorted((ROOT_REQUIRED | {checkpoint}) - files)
    if missing:
        raise ValueError(f"Refusing cleanup before verified SFT-v2 publication: {missing}")
    return sorted(
        path for path in files if any(path.startswith(prefix) for prefix in LEGACY_PREFIXES)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    selected = summary.get("selected")
    if not summary.get("release_ready") or not selected:
        raise ValueError("Selection summary does not authorize repository cleanup")
    selected_step = int(selected["step"])

    api = HfApi(token=token)
    before = set(api.list_repo_files(args.repo_id, repo_type="model", token=token))
    planned = cleanup_plan(before, selected_step)
    result = {
        "repo_id": args.repo_id,
        "selected_step": selected_step,
        "execute": args.execute,
        "deleted": planned if args.execute else [],
        "planned": planned,
    }
    if args.execute and planned:
        api.create_commit(
            repo_id=args.repo_id,
            repo_type="model",
            token=token,
            operations=[CommitOperationDelete(path_in_repo=path) for path in planned],
            commit_message="Remove superseded SFT-v1 checkpoints and reports",
        )
        after = set(api.list_repo_files(args.repo_id, repo_type="model", token=token))
        survivors = sorted(set(planned) & after)
        if survivors:
            raise RuntimeError(f"Legacy cleanup verification failed: {survivors}")
        cleanup_plan(after, selected_step)
        result["verified_retained_files"] = sorted(
            ROOT_REQUIRED
            | {
                f"training/sft-v2-300k/checkpoints/step_{selected_step:06d}/checkpoint.pt"
            }
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
