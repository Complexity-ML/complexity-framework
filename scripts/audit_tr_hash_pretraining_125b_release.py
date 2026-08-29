#!/usr/bin/env python3
"""Audit a completed TR-HASH 125B dataset release from manifests and LFS metadata."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from scripts.build_agentic_pretraining_50b import (
    config_sha256,
    validate_config,
    validate_curriculum,
)


def audit_release_documents(
    *,
    config: Mapping[str, Any],
    curriculum: Mapping[str, Any],
    mixture: Mapping[str, Any],
    plan: Mapping[str, Any],
    state: Mapping[str, Any],
    source_manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    validate_config(config)
    validate_curriculum(config, curriculum)
    config_digest = config_sha256(config)
    curriculum_digest = config_sha256(curriculum)
    if mixture.get("config_sha256") != config_digest or plan.get("config_sha256") != config_digest:
        raise ValueError("config SHA-256 disagrees with the published runtime metadata")
    if plan.get("curriculum_sha256") != curriculum_digest:
        raise ValueError("curriculum SHA-256 disagrees with the runtime plan")
    if state.get("config_sha256") != config_digest:
        raise ValueError("build state config SHA-256 disagrees with the published config")
    if state.get("exact_document_deduplication") is not True:
        raise ValueError("build state does not prove global exact-document deduplication")
    protected_count = int(state.get("protected_prompt_count", 0))
    if config.get("protected_benchmark_sources") and protected_count <= 0:
        raise ValueError("build state does not prove a populated benchmark decontamination index")
    protected_digest = str(state.get("protected_index_sha256", ""))
    if len(protected_digest) != 64:
        raise ValueError("build state has no valid protected-index SHA-256")

    actual_tokens = int(mixture["actual_tokens"])
    if int(plan["unique_tokens"]) != actual_tokens or int(plan["trained_tokens"]) != actual_tokens:
        raise ValueError("the runtime plan replays or omits packed tokens")
    contract = config["tokenizer_contract"]
    if contract.get("status") != "validated":
        raise ValueError("published corpus tokenizer contract is not validated")
    if mixture.get("tokenizer_sha256") != contract.get("tokenizer_sha256"):
        raise ValueError("mixture tokenizer SHA-256 disagrees with the pinned tokenizer")
    if state.get("tokenizer_sha256") != mixture.get("tokenizer_sha256"):
        raise ValueError("build state tokenizer SHA-256 disagrees with the mixture")

    expected_sources = {str(source["name"]): source for source in mixture["sources"]}
    if set(source_manifests) != set(expected_sources):
        raise ValueError("source manifest set disagrees with the mixture manifest")
    if set(plan.get("source_passes", {})) != set(expected_sources) or any(
        int(value) != 1 for value in plan["source_passes"].values()
    ):
        raise ValueError("every source must have exactly one training pass")
    state_sources = {str(source["name"]): source for source in state.get("sources", ())}
    if set(state_sources) != set(expected_sources):
        raise ValueError("build state source set disagrees with the mixture manifest")

    expected_shards: dict[str, dict[str, Any]] = {}
    source_rows: dict[str, int] = {}
    for source_name, entry in expected_sources.items():
        manifest = source_manifests[source_name]
        if manifest.get("source") != source_name:
            raise ValueError(f"source manifest name mismatch: {source_name}")
        if int(manifest["seq_len"]) != int(mixture["seq_len"]) or manifest["dtype"] != "uint16":
            raise ValueError(f"source layout mismatch: {source_name}")
        rows = sum(int(shard["rows"]) for shard in manifest["shards"])
        if rows != int(manifest["rows"]) or rows != int(entry["rows"]):
            raise ValueError(f"source row mismatch: {source_name}")
        if int(manifest["trained_tokens"]) != rows * int(mixture["seq_len"]):
            raise ValueError(f"source token mismatch: {source_name}")
        source_rows[source_name] = rows
        if int(state_sources[source_name]["rows_done"]) != rows:
            raise ValueError(f"build state is incomplete for source: {source_name}")
        manifest_parent = PurePosixPath(str(entry["manifest"])).parent
        for shard in manifest["shards"]:
            relative = str(manifest_parent / str(shard["file"]))
            if relative in expected_shards:
                raise ValueError(f"duplicate shard path: {relative}")
            if int(shard["bytes"]) != int(shard["tokens"]) * 2:
                raise ValueError(f"uint16 byte count mismatch: {relative}")
            expected_shards[relative] = {
                "bytes": int(shard["bytes"]),
                "sha256": str(shard["sha256"]),
                "source": source_name,
                "file": str(shard["file"]),
                "rows": int(shard["rows"]),
            }

    if sum(int(entry["trained_tokens"]) for entry in expected_sources.values()) != actual_tokens:
        raise ValueError("source tokens do not sum to actual_tokens")

    planned: list[tuple[str, str]] = []
    planned_rows = {name: 0 for name in expected_sources}
    for phase in plan["phases"]:
        if int(phase.get("passes", 0)) != 1:
            raise ValueError(f"phase {phase.get('name')} does not use one pass")
        for source_name, shards in phase["sources"].items():
            if source_name not in expected_sources:
                raise ValueError(f"unknown source in runtime plan: {source_name}")
            for shard in shards:
                planned.append((source_name, str(shard["file"])))
                planned_rows[source_name] += int(shard["rows"])
    expected_pairs = {
        (metadata["source"], metadata["file"]) for metadata in expected_shards.values()
    }
    if len(planned) != len(set(planned)) or set(planned) != expected_pairs:
        raise ValueError("runtime plan does not consume every shard exactly once")
    if planned_rows != source_rows:
        raise ValueError("runtime plan row totals disagree with source manifests")

    return {
        "requested_tokens": int(mixture["requested_tokens"]),
        "actual_tokens": actual_tokens,
        "alignment_overhead_tokens": actual_tokens - int(mixture["requested_tokens"]),
        "source_count": len(expected_sources),
        "shard_count": len(expected_shards),
        "tokenizer_sha256": mixture["tokenizer_sha256"],
        "config_sha256": config_digest,
        "curriculum_sha256": curriculum_digest,
        "protected_prompt_count": protected_count,
        "protected_index_sha256": protected_digest,
        "expected_shards": expected_shards,
    }


def _download_json(repo_id: str, filename: str, token: str | None) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", token=token)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit_hub_release(repo_id: str, *, token: str | None, require_private: bool) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    repo = api.repo_info(repo_id=repo_id, repo_type="dataset")
    if require_private and not bool(repo.private):
        raise ValueError("dataset must remain private until the release audit passes")
    config = _download_json(repo_id, "_metadata/config.json", token)
    curriculum = _download_json(repo_id, "_metadata/curriculum.json", token)
    mixture = _download_json(repo_id, "mixture_manifest.json", token)
    plan = _download_json(repo_id, "pretrain_plan.json", token)
    state = _download_json(repo_id, "_state/state.json", token)
    source_manifests = {
        str(entry["name"]): _download_json(repo_id, str(entry["manifest"]), token)
        for entry in mixture["sources"]
    }
    report = audit_release_documents(
        config=config,
        curriculum=curriculum,
        mixture=mixture,
        plan=plan,
        state=state,
        source_manifests=source_manifests,
    )

    expected = report.pop("expected_shards")
    paths = sorted(expected)
    for start in range(0, len(paths), 100):
        batch = paths[start : start + 100]
        remote = {
            str(info.path): info
            for info in api.get_paths_info(
                repo_id=repo_id,
                paths=batch,
                repo_type="dataset",
                expand=True,
            )
        }
        if set(remote) != set(batch):
            raise ValueError(f"missing remote shards: {sorted(set(batch) - set(remote))}")
        for path in batch:
            info = remote[path]
            wanted = expected[path]
            if int(info.size) != wanted["bytes"]:
                raise ValueError(f"remote size mismatch: {path}")
            lfs = getattr(info, "lfs", None)
            if lfs is None or getattr(lfs, "sha256", None) != wanted["sha256"]:
                raise ValueError(f"remote LFS SHA-256 mismatch: {path}")
    report["remote_shards_verified"] = len(paths)
    report["private"] = bool(repo.private)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="AETHORIA-AI/TR-HASH-Pretraining-125B-Agentic-32K")
    parser.add_argument("--allow-public", action="store_true")
    args = parser.parse_args()
    import os

    report = audit_hub_release(
        args.repo,
        token=os.environ.get("HF_TOKEN"),
        require_private=not args.allow_public,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
