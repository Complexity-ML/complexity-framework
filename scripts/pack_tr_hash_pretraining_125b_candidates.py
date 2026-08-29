#!/usr/bin/env python3
"""Globally deduplicate and tokenize verified 125B candidate shards."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
from collections.abc import Iterator, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from scripts.build_agentic_pretraining_50b import (
    HubPublisher,
    build,
    sha256_file,
    validate_config,
    validate_curriculum,
)

LOGGER = logging.getLogger("tr_hash_125b_pack")


def load_candidate_manifests(path: Path, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("complete"):
        raise ValueError("candidate manifest is not complete")
    manifests = {str(item["source"]): item for item in payload.get("sources", ())}
    expected = [str(source["name"]) for source in config["sources"]]
    if list(manifests) != expected:
        raise ValueError("candidate manifest source order differs from the canonical config")
    for source in config["sources"]:
        name = str(source["name"])
        manifest = manifests[name]
        if not manifest.get("complete"):
            raise ValueError(f"candidate source is incomplete: {name}")
        if int(manifest["retained_tokens"]) < int(manifest["candidate_target_tokens"]):
            raise ValueError(f"candidate source is below its staged quota: {name}")
        indexes = [int(shard["shard_index"]) for shard in manifest["shards"]]
        if indexes != list(range(len(indexes))):
            raise ValueError(f"candidate shard sequence is not contiguous: {name}")
    return manifests


def staged_config(
    config: Mapping[str, Any],
    manifests: Mapping[str, Mapping[str, Any]],
    *,
    candidate_manifest_sha256: str,
) -> dict[str, Any]:
    """Create the audited final-pack config without any raw-source re-filtering."""

    derived = deepcopy(dict(config))
    derived["parallel_sources"] = 1
    derived["protected_benchmarks"] = []
    derived["protected_benchmark_sources"] = []
    derived["candidate_manifest_sha256"] = candidate_manifest_sha256
    normalized_sources = []
    for original in config["sources"]:
        source = {
            key: value
            for key, value in dict(original).items()
            if key
            not in {
                "dataset_id",
                "config_name",
                "revision",
                "split",
                "path",
                "path_env",
                "source_type",
                "messages_field",
                "content_fields",
                "text_field",
            }
        }
        source["selection"] = "staged"
        source["text_field"] = "text"
        source["candidate_shards"] = list(manifests[str(source["name"])]["shards"])
        normalized_sources.append(source)
    derived["sources"] = normalized_sources
    validate_config(derived)
    return derived


class CandidateShardReader:
    """Download, verify, consume, then evict one candidate shard at a time."""

    def __init__(self, *, repo_id: str, cache_dir: Path, token: str | None = None) -> None:
        self.repo_id = repo_id
        self.cache_dir = cache_dir
        self.token = token
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _download(self, shard: Mapping[str, Any]) -> Path:
        local = shard.get("local_path")
        if local:
            path = Path(str(local))
        else:
            from huggingface_hub import hf_hub_download

            path = Path(
                hf_hub_download(
                    repo_id=self.repo_id,
                    filename=str(shard["repo_path"]),
                    repo_type="dataset",
                    token=self.token,
                    local_dir=self.cache_dir,
                )
            )
        if sha256_file(path) != str(shard["sha256"]):
            raise RuntimeError(f"candidate SHA-256 mismatch: {shard['repo_path']}")
        return path

    def __call__(self, source: Mapping[str, Any], seed: int) -> Iterator[Mapping[str, Any]]:
        del seed
        for shard in source["candidate_shards"]:
            path = self._download(shard)
            evict = not shard.get("local_path")
            try:
                with gzip.open(path, "rt", encoding="utf-8") as stream:
                    for line_number, line in enumerate(stream, 1):
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if row.get("source") != source["name"]:
                            raise ValueError(
                                f"candidate source mismatch at {shard['repo_path']}:{line_number}"
                            )
                        yield row
            finally:
                if evict:
                    path.unlink(missing_ok=True)
                    LOGGER.info("candidate consumed and evicted: %s", shard["repo_path"])


def pack_candidates(
    *,
    config: Mapping[str, Any],
    curriculum: Mapping[str, Any],
    candidate_manifest_path: Path,
    tokenizer_path: Path,
    work_dir: Path,
    repo_id: str,
    repo_prefix: str,
    candidate_cache_dir: Path,
    token: str | None = None,
    publisher: Any | None = None,
    source_iterator_factory: Any | None = None,
    max_shards: int | None = None,
) -> dict[str, Any]:
    manifests = load_candidate_manifests(candidate_manifest_path, config)
    derived = staged_config(
        config,
        manifests,
        candidate_manifest_sha256=sha256_file(candidate_manifest_path),
    )
    validate_curriculum(derived, curriculum)
    actual_publisher = publisher or HubPublisher(repo_id, repo_prefix, token)
    reader = source_iterator_factory or CandidateShardReader(
        repo_id=repo_id,
        cache_dir=candidate_cache_dir,
        token=token,
    )
    return build(
        config=derived,
        tokenizer_path=tokenizer_path,
        work_dir=work_dir,
        publisher=actual_publisher,
        curriculum=curriculum,
        max_shards=max_shards,
        source_iterator_factory=reader,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/agentic_pretraining/tr_hash_pretraining_125b.json"
    )
    parser.add_argument(
        "--curriculum",
        default="configs/agentic_pretraining/tr_hash_pretraining_125b_curriculum.json",
    )
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--candidate-cache-dir", required=True)
    parser.add_argument("--hf-repo", default="AETHORIA-AI/TR-HASH-Pretraining-125B-Agentic-32K")
    parser.add_argument("--repo-prefix", default="production")
    parser.add_argument("--max-shards", type=int)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    curriculum = json.loads(Path(args.curriculum).read_text(encoding="utf-8"))
    pack_candidates(
        config=config,
        curriculum=curriculum,
        candidate_manifest_path=Path(args.candidate_manifest),
        tokenizer_path=Path(args.tokenizer),
        work_dir=Path(args.work_dir),
        repo_id=args.hf_repo,
        repo_prefix=args.repo_prefix,
        candidate_cache_dir=Path(args.candidate_cache_dir),
        token=os.environ.get("HF_TOKEN"),
        max_shards=args.max_shards,
    )


if __name__ == "__main__":
    main()
