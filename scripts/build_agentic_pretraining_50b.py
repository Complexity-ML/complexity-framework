#!/usr/bin/env python3
"""Build, publish, verify, and locally evict a restart-safe token corpus.

The same implementation builds both the filtered corpora and explicitly
source-curated direct corpora.  Direct mode keeps the source-level curation and
fixed token budgets but intentionally skips per-document filtering, benchmark
decontamination, and global exact deduplication for high-throughput builds.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import queue
import re
import sqlite3
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from scripts.build_agentic_pretraining_corpus import (
    DEFAULT_PROTECTED_BENCHMARKS,
    benchmark_match,
    build_benchmark_index,
    content_sha256,
    is_agentic_candidate,
    iter_source,
    normalize_text,
    quality_rejection,
)

LOGGER = logging.getLogger("tr_hash_agentic_50b")
SCHEMA = "tr-hash-token-production-v2"
ALLOWED_SELECTIONS = {"quality", "agentic", "agentic_trajectory", "staged", "direct"}


def make_direct_source_curated_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return an explicit high-throughput source-curated production config."""

    direct = copy.deepcopy(dict(config))
    direct["schema"] = "tr-hash-pretraining-125b-source-curated-v1"
    direct["description"] = (
        "Production 125B source-curated pretraining corpus for TR-HASH Agentic 32K. "
        "Documents are tokenized directly from pinned, curated upstream sources."
    )
    direct["direct_materialization"] = True
    direct["tokenization_batch_size"] = 4096
    direct["producer_candidate_batch_size"] = 4096
    direct["producer_scan_batch_size"] = 4096
    direct["parallel_sources"] = 3
    direct["producer_queue_depth"] = 4
    direct["protected_benchmarks"] = []
    direct["protected_benchmark_sources"] = []
    direct["release_gate"] = (
        "Keep private until source licenses, shard hashes, token budgets, and the "
        "source-curated direct-materialization disclosure have been audited."
    )
    for source in direct["sources"]:
        source["selection"] = "direct"
        source["shuffle_buffer"] = 1
    return direct


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_sha256(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def allocate_rows(
    *,
    target_tokens: int,
    seq_len: int,
    global_batch_sequences: int,
    sources: Sequence[Mapping[str, Any]],
) -> tuple[int, int, dict[str, int]]:
    if min(target_tokens, seq_len, global_batch_sequences) < 1:
        raise ValueError("layout values must be positive")
    total_rows = math.ceil(target_tokens / (seq_len * global_batch_sequences))
    total_rows *= global_batch_sequences
    total_units = total_rows // global_batch_sequences
    exact = [total_units * float(source["weight"]) for source in sources]
    units = [math.floor(value) for value in exact]
    remainder = total_units - sum(units)
    order = sorted(range(len(sources)), key=lambda index: exact[index] - units[index], reverse=True)
    for index in order[:remainder]:
        units[index] += 1
    rows = [value * global_batch_sequences for value in units]
    return (
        total_rows,
        total_rows * seq_len,
        {str(source["name"]): rows[index] for index, source in enumerate(sources)},
    )


def allocate_phase_rows(
    *,
    config: Mapping[str, Any],
    curriculum: Mapping[str, Any],
    rows_by_source: Mapping[str, int],
) -> dict[str, dict[str, int]]:
    """Allocate aligned source rows to phases without replay or shard overlap."""

    alignment = int(config["global_batch_sequences"])
    phases = list(curriculum["phases"])
    result = {str(phase["name"]): {} for phase in phases}
    sources_by_bucket: dict[str, list[Mapping[str, Any]]] = {}
    for source in config["sources"]:
        sources_by_bucket.setdefault(str(source["bucket"]), []).append(source)

    for bucket, sources in sources_by_bucket.items():
        source_units = {
            str(source["name"]): int(rows_by_source[str(source["name"])]) // alignment
            for source in sources
        }
        total_units = sum(source_units.values())
        requested_bucket = int(config["bucket_targets"][bucket])
        phase_exact = [
            total_units * int(phase["bucket_tokens"][bucket]) / requested_bucket for phase in phases
        ]
        phase_units = [math.floor(value) for value in phase_exact]
        remainder = total_units - sum(phase_units)
        order = sorted(
            range(len(phases)),
            key=lambda index: phase_exact[index] - phase_units[index],
            reverse=True,
        )
        for index in order[:remainder]:
            phase_units[index] += 1

        remaining = dict(source_units)
        for phase_index, phase in enumerate(phases):
            phase_name = str(phase["name"])
            if phase_index == len(phases) - 1:
                allocation = remaining
            else:
                share = int(phase["bucket_tokens"][bucket]) / requested_bucket
                exact = {name: source_units[name] * share for name in source_units}
                allocation = {
                    name: min(remaining[name], math.floor(value)) for name, value in exact.items()
                }
                needed = phase_units[phase_index] - sum(allocation.values())
                candidates = sorted(
                    source_units,
                    key=lambda name: (exact[name] - allocation[name], remaining[name]),
                    reverse=True,
                )
                for name in candidates:
                    if needed == 0:
                        break
                    if allocation[name] < remaining[name]:
                        allocation[name] += 1
                        needed -= 1
                if needed:
                    raise ValueError(
                        f"cannot align curriculum allocation for {bucket}/{phase_name}"
                    )
                remaining = {name: remaining[name] - allocation[name] for name in remaining}
            for name, units in allocation.items():
                if units:
                    result[phase_name][name] = units * alignment

    for source_name, rows in rows_by_source.items():
        if sum(phase.get(source_name, 0) for phase in result.values()) != rows:
            raise ValueError(f"curriculum does not consume source exactly once: {source_name}")
    return result


def validate_config(config: Mapping[str, Any]) -> None:
    sources = list(config.get("sources", ()))
    if not sources:
        raise ValueError("corpus config must contain sources")
    names = [str(source.get("name", "")) for source in sources]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("source names must be non-empty and unique")
    if abs(sum(float(source.get("weight", 0.0)) for source in sources) - 1.0) > 1e-9:
        raise ValueError("source weights must sum to 1")
    target_tokens = int(config["target_tokens"])
    explicit_targets = [source.get("target_tokens") for source in sources]
    if any(value is not None for value in explicit_targets):
        if not all(value is not None for value in explicit_targets):
            raise ValueError("either every source or no source must define target_tokens")
        if sum(int(value) for value in explicit_targets) != target_tokens:
            raise ValueError("source target_tokens must sum to the corpus target")
        for source in sources:
            expected_weight = int(source["target_tokens"]) / target_tokens
            if abs(float(source["weight"]) - expected_weight) > 1e-12:
                raise ValueError(f"source weight disagrees with target_tokens: {source['name']}")
    buckets = Counter(str(source.get("bucket", "")) for source in sources)
    if "" in buckets:
        raise ValueError("every source must define a bucket")
    for source in sources:
        selection = str(source.get("selection", "agentic"))
        if selection not in ALLOWED_SELECTIONS:
            raise ValueError(f"unsupported selection {selection!r}: {source['name']}")
        if (
            "dataset_id" in source
            and re.fullmatch(r"[0-9a-f]{40}", str(source.get("revision", ""))) is None
        ):
            raise ValueError(
                f"remote source revision is not an immutable git SHA: {source['name']}"
            )
        if not source.get("license_audit"):
            raise ValueError(f"source has no license audit note: {source['name']}")
    direct_materialization = bool(config.get("direct_materialization", False))
    direct_sources = [
        str(source["name"])
        for source in sources
        if str(source.get("selection", "agentic")) == "direct"
    ]
    if direct_sources and not direct_materialization:
        raise ValueError("direct source selection requires direct_materialization=true")
    if direct_materialization:
        if len(direct_sources) != len(sources):
            raise ValueError("direct materialization requires every source selection to be direct")
        if config.get("protected_benchmarks") or config.get("protected_benchmark_sources"):
            raise ValueError("direct materialization cannot claim benchmark decontamination")
    for benchmark in config.get("protected_benchmark_sources", ()):
        if (
            "dataset_id" in benchmark
            and re.fullmatch(r"[0-9a-f]{40}", str(benchmark.get("revision", ""))) is None
        ):
            raise ValueError(
                f"protected benchmark revision is not immutable: {benchmark.get('name')}"
            )
        if (
            "archive_url" in benchmark
            and re.fullmatch(r"[0-9a-f]{64}", str(benchmark.get("archive_sha256", ""))) is None
        ):
            raise ValueError(
                f"protected benchmark archive SHA-256 is invalid: {benchmark.get('name')}"
            )
    bucket_targets = config.get("bucket_targets")
    if bucket_targets:
        if sum(int(value) for value in bucket_targets.values()) != target_tokens:
            raise ValueError("bucket targets must sum to the corpus target")
        actual = Counter()
        for source in sources:
            actual[str(source["bucket"])] += int(source["target_tokens"])
        if dict(actual) != {str(key): int(value) for key, value in bucket_targets.items()}:
            raise ValueError(f"source targets disagree with bucket targets: {dict(actual)}")


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def row_text(row: Mapping[str, Any], source: Mapping[str, Any]) -> str:
    """Serialize raw documents or tool trajectories with canonical 32K markers."""

    messages_field = source.get("messages_field")
    if messages_field:
        messages = row.get(str(messages_field), ())
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            return ""
        parts: list[str] = []
        tools_field = source.get("tools_field")
        tools = row.get(str(tools_field), ()) if tools_field else ()
        if tools:
            parts.extend(("<|system|>", "Available tools: ", _json_text(tools), "<|end_of_turn|>"))
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            role = str(message.get("role", "")).casefold()
            content = message.get("content", "")
            if not isinstance(content, str):
                content = _json_text(content)
            if role in {"tool", "function", "observation"}:
                parts.extend(
                    ("<|tool_result_start|>", content, "<|tool_result_end|>", "<|end_of_turn|>")
                )
                continue
            marker = role if role in {"system", "user", "assistant"} else "assistant"
            parts.extend((f"<|{marker}|>", content))
            tool_calls = message.get("tool_calls") or message.get("function_call")
            if tool_calls:
                parts.extend(("<|tool_call_start|>", _json_text(tool_calls), "<|tool_call_end|>"))
            parts.append("<|end_of_turn|>")
        return "".join(parts)
    value = row.get(source.get("text_field", "text"), "")
    return value if isinstance(value, str) else ""


def validate_tokenizer_contract(config: Mapping[str, Any], tokenizer_path: Path) -> None:
    contract = config.get("tokenizer_contract")
    if not contract:
        return
    if contract.get("status") != "validated":
        raise ValueError("125B build is gated until tokenizer_contract.status is validated")
    revision = contract.get("revision")
    manifest_digest = contract.get("manifest_sha256")
    tokenizer_digest = contract.get("tokenizer_sha256")
    if not revision or not manifest_digest or not tokenizer_digest:
        raise ValueError(
            "validated tokenizer contract requires revision, manifest_sha256, and tokenizer_sha256"
        )
    manifest = tokenizer_path / str(
        contract.get("required_manifest", "agentic_tokenizer_manifest.json")
    )
    if not manifest.is_file() or sha256_file(manifest) != manifest_digest:
        raise ValueError("local tokenizer manifest does not match the pinned contract")
    tokenizer_json = (
        tokenizer_path / "tokenizer.json" if tokenizer_path.is_dir() else tokenizer_path
    )
    if not tokenizer_json.is_file() or sha256_file(tokenizer_json) != tokenizer_digest:
        raise ValueError("local tokenizer.json does not match the pinned contract")


def validate_curriculum(config: Mapping[str, Any], curriculum: Mapping[str, Any]) -> None:
    phases = list(curriculum.get("phases", ()))
    if not phases:
        raise ValueError("curriculum must contain phases")
    if int(curriculum.get("total_tokens", -1)) != int(config["target_tokens"]):
        raise ValueError("curriculum total does not match corpus target")
    if sum(int(phase["target_tokens"]) for phase in phases) != int(config["target_tokens"]):
        raise ValueError("phase targets do not sum to corpus target")
    aggregate: Counter[str] = Counter()
    names: set[str] = set()
    for phase in phases:
        name = str(phase.get("name", ""))
        if not name or name in names:
            raise ValueError("curriculum phase names must be non-empty and unique")
        names.add(name)
        bucket_tokens = {str(key): int(value) for key, value in phase["bucket_tokens"].items()}
        if sum(bucket_tokens.values()) != int(phase["target_tokens"]):
            raise ValueError(f"bucket tokens do not sum within phase {name}")
        shares = {str(key): float(value) for key, value in phase["bucket_shares"].items()}
        if set(shares) != set(bucket_tokens) or abs(sum(shares.values()) - 1.0) > 1e-9:
            raise ValueError(f"invalid bucket shares in phase {name}")
        for bucket, tokens in bucket_tokens.items():
            if abs(shares[bucket] - tokens / int(phase["target_tokens"])) > 1e-12:
                raise ValueError(f"bucket share disagrees with tokens in phase {name}")
            aggregate[bucket] += tokens
    expected = {str(key): int(value) for key, value in config["bucket_targets"].items()}
    if dict(aggregate) != expected:
        raise ValueError(f"curriculum bucket totals disagree with corpus: {dict(aggregate)}")
    invariants = curriculum.get("invariants", {})
    if (
        invariants.get("replay") is not False
        or invariants.get("each_packed_row_consumed_once") is not True
    ):
        raise ValueError("125B curriculum must be no-replay and consume every row once")


class StateStore:
    """Exact content deduplication and shard progress in one SQLite transaction."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=120)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS seen (
                digest BLOB PRIMARY KEY
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS progress (
                source TEXT PRIMARY KEY,
                scanned INTEGER NOT NULL,
                rows_done INTEGER NOT NULL,
                source_tokens INTEGER NOT NULL,
                last_token INTEGER,
                carry BLOB NOT NULL,
                counters TEXT NOT NULL,
                signals TEXT NOT NULL,
                partial_position INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS shards (
                source TEXT NOT NULL,
                shard_index INTEGER NOT NULL,
                repo_path TEXT NOT NULL,
                rows INTEGER NOT NULL,
                tokens INTEGER NOT NULL,
                bytes INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                PRIMARY KEY (source, shard_index)
            );
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        progress_columns = {
            str(row[1]) for row in self.connection.execute("PRAGMA table_info(progress)")
        }
        if "partial_position" not in progress_columns:
            self.connection.execute(
                "ALTER TABLE progress ADD COLUMN partial_position INTEGER NOT NULL DEFAULT 0"
            )
        self.connection.commit()

    def bind_run(self, *, config_digest: str, tokenizer_digest: str) -> None:
        expected = {"config_sha256": config_digest, "tokenizer_sha256": tokenizer_digest}
        existing = dict(self.connection.execute("SELECT key, value FROM metadata"))
        conflicts = {
            key: (existing[key], value)
            for key, value in expected.items()
            if key in existing and existing[key] != value
        }
        if conflicts:
            raise ValueError(f"work directory belongs to an incompatible build: {conflicts}")
        self.connection.executemany(
            "INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)", expected.items()
        )
        self.connection.commit()

    def progress(self, source: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT scanned, rows_done, source_tokens, last_token, carry, counters, signals, "
            "partial_position "
            "FROM progress WHERE source=?",
            (source,),
        ).fetchone()
        if row is None:
            return {
                "scanned": 0,
                "rows_done": 0,
                "source_tokens": 0,
                "last_token": None,
                "carry": np.empty(0, dtype=np.uint16),
                "counters": Counter(),
                "signals": Counter(),
                "partial_position": 0,
            }
        return {
            "scanned": int(row[0]),
            "rows_done": int(row[1]),
            "source_tokens": int(row[2]),
            "last_token": row[3],
            "carry": np.frombuffer(row[4], dtype=np.uint16).copy(),
            "counters": Counter(json.loads(row[5])),
            "signals": Counter(json.loads(row[6])),
            "partial_position": int(row[7]),
        }

    def shards(self, source: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT source, shard_index, repo_path, rows, tokens, bytes, sha256 FROM shards"
        params: tuple[Any, ...] = ()
        if source is not None:
            sql += " WHERE source=?"
            params = (source,)
        sql += " ORDER BY source, shard_index"
        return [
            dict(
                zip(
                    ("source", "shard_index", "repo_path", "rows", "tokens", "bytes", "sha256"), row
                )
            )
            for row in self.connection.execute(sql, params)
        ]

    def begin(self) -> None:
        self.connection.execute("BEGIN IMMEDIATE")

    def reserve_digest(self, digest: str) -> bool:
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO seen(digest) VALUES (?)", (bytes.fromhex(digest),)
        )
        return cursor.rowcount == 1

    def save_progress(self, *, source: str, progress: Mapping[str, Any]) -> None:
        """Commit a restart point after its partial memmap has been flushed."""

        carry = np.asarray(progress["carry"], dtype=np.uint16)
        self.connection.execute(
            """
            INSERT INTO progress(
                source, scanned, rows_done, source_tokens, last_token, carry,
                counters, signals, partial_position
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source) DO UPDATE SET
                scanned=excluded.scanned, rows_done=excluded.rows_done,
                source_tokens=excluded.source_tokens, last_token=excluded.last_token,
                carry=excluded.carry, counters=excluded.counters, signals=excluded.signals,
                partial_position=excluded.partial_position
            """,
            (
                source,
                int(progress["scanned"]),
                int(progress["rows_done"]),
                int(progress["source_tokens"]),
                progress["last_token"],
                carry.tobytes(),
                json.dumps(dict(progress["counters"]), sort_keys=True),
                json.dumps(dict(progress["signals"]), sort_keys=True),
                int(progress.get("partial_position", 0)),
            ),
        )
        self.connection.commit()

    def commit_shard(
        self,
        *,
        source: str,
        scanned: int,
        rows_done: int,
        source_tokens: int,
        last_token: int,
        carry: np.ndarray,
        counters: Counter[str],
        signals: Counter[str],
        shard: Mapping[str, Any],
        partial_position: int = 0,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO progress(
                source, scanned, rows_done, source_tokens, last_token, carry,
                counters, signals, partial_position
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source) DO UPDATE SET
                scanned=excluded.scanned, rows_done=excluded.rows_done,
                source_tokens=excluded.source_tokens, last_token=excluded.last_token,
                carry=excluded.carry, counters=excluded.counters, signals=excluded.signals,
                partial_position=excluded.partial_position
            """,
            (
                source,
                scanned,
                rows_done,
                source_tokens,
                last_token,
                carry.astype(np.uint16, copy=False).tobytes(),
                json.dumps(dict(counters), sort_keys=True),
                json.dumps(dict(signals), sort_keys=True),
                partial_position,
            ),
        )
        self.connection.execute(
            "INSERT INTO shards(source, shard_index, repo_path, rows, tokens, bytes, sha256) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                source,
                shard["shard_index"],
                shard["repo_path"],
                shard["rows"],
                shard["tokens"],
                shard["bytes"],
                shard["sha256"],
            ),
        )
        self.connection.commit()

    def rollback(self) -> None:
        self.connection.rollback()


class HubPublisher:
    def __init__(
        self,
        repo_id: str,
        prefix: str,
        token: str | None = None,
        *,
        create_private_repo: bool = False,
    ) -> None:
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.prefix = prefix.strip("/")
        self.api = HfApi(token=token)
        if create_private_repo:
            self.api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=True,
                exist_ok=True,
            )
        self.api.repo_info(repo_id=repo_id, repo_type="dataset")

    def repo_path(self, relative: str) -> str:
        return f"{self.prefix}/{relative}" if self.prefix else relative

    def _publish_file_at(self, local_path: Path, repo_path: str) -> dict[str, Any]:
        expected_size = local_path.stat().st_size
        expected_sha = sha256_file(local_path)
        last_error: BaseException | None = None
        for attempt in range(6):
            try:
                self.api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    commit_message=f"Upload verified corpus artifact {repo_path}",
                )
                last_error = None
                break
            except Exception as error:
                last_error = error
                if attempt == 5:
                    break
                delay = 2**attempt
                LOGGER.warning(
                    "upload failed for %s (attempt %d/6); retrying in %ds: %s",
                    repo_path,
                    attempt + 1,
                    delay,
                    error,
                )
                time.sleep(delay)
        if last_error is not None:
            raise RuntimeError(f"upload failed after 6 attempts: {repo_path}") from last_error
        info = self.api.get_paths_info(
            repo_id=self.repo_id,
            paths=[repo_path],
            repo_type="dataset",
            expand=True,
        )[0]
        if int(info.size) != expected_size:
            raise RuntimeError(f"remote size mismatch for {repo_path}")
        lfs = getattr(info, "lfs", None)
        remote_sha = getattr(lfs, "sha256", None) if lfs is not None else None
        if remote_sha is None:
            from huggingface_hub import hf_hub_download

            verification = Path(
                hf_hub_download(
                    repo_id=self.repo_id,
                    filename=repo_path,
                    repo_type="dataset",
                    token=self.api.token,
                    force_download=True,
                )
            )
            remote_sha = sha256_file(verification)
        if remote_sha != expected_sha:
            raise RuntimeError(f"remote sha256 mismatch for {repo_path}")
        return {"repo_path": repo_path, "bytes": expected_size, "sha256": expected_sha}

    def publish_file(self, local_path: Path, relative: str) -> dict[str, Any]:
        return self._publish_file_at(local_path, self.repo_path(relative))

    def publish_root_file(self, local_path: Path, relative: str) -> dict[str, Any]:
        return self._publish_file_at(local_path, relative.strip("/"))

    def publish_json(self, payload: Mapping[str, Any], relative: str, work_dir: Path) -> None:
        path = work_dir / ".metadata" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        self.publish_file(path, relative)


def _tokenizer(path: Path) -> tuple[Tokenizer, int | None]:
    tokenizer_json = path / "tokenizer.json" if path.is_dir() else path
    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    config_path = tokenizer_json.parent / "tokenizer_config.json"
    eos_token_id = None
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        eos = config.get("eos_token")
        if isinstance(eos, str):
            eos_token_id = tokenizer.token_to_id(eos)
    if tokenizer.get_vocab_size(with_added_tokens=True) != 32_000:
        raise ValueError("production tokenizer must contain exactly 32,000 IDs")
    return tokenizer, eos_token_id


def _state_payload(
    store: StateStore,
    *,
    config: Mapping[str, Any],
    config_digest: str,
    actual_tokens: int,
    rows_by_source: Mapping[str, int],
    tokenizer_digest: str,
    protected_prompt_count: int,
    protected_index_digest: str,
) -> dict[str, Any]:
    direct_materialization = bool(config.get("direct_materialization", False))
    sources = []
    for source in config["sources"]:
        name = str(source["name"])
        progress = store.progress(name)
        sources.append(
            {
                "name": name,
                "bucket": source["bucket"],
                "selection": source.get("selection", "agentic"),
                "weight": source["weight"],
                "requested_source_tokens": source.get("target_tokens"),
                "license_audit": source["license_audit"],
                "input": {
                    key: source[key]
                    for key in ("source_type", "dataset_id", "config_name", "split", "revision")
                    if key in source
                },
                "target_rows": rows_by_source[name],
                "rows_done": progress["rows_done"],
                "scanned": progress["scanned"],
                "source_tokens": progress["source_tokens"],
                "counters": dict(progress["counters"]),
                "signal_counts": dict(progress["signals"]),
                "shards": store.shards(name),
            }
        )
    return {
        "schema": SCHEMA,
        "config_sha256": config_digest,
        "tokenizer_sha256": tokenizer_digest,
        "requested_tokens": int(config["target_tokens"]),
        "actual_tokens": actual_tokens,
        "seq_len": int(config["seq_len"]),
        "materialization_mode": (
            "direct_source_curated" if direct_materialization else "filtered_deduplicated"
        ),
        "exact_document_deduplication": not direct_materialization,
        "benchmark_decontamination": not direct_materialization,
        "agentic_signal_filtering": not direct_materialization,
        "protected_prompt_count": protected_prompt_count,
        "protected_index_sha256": protected_index_digest,
        "sources": sources,
    }


def _source_manifest(
    store: StateStore,
    *,
    source: Mapping[str, Any],
    seq_len: int,
    expected_rows: int,
) -> dict[str, Any]:
    shards = store.shards(str(source["name"]))
    rows = sum(int(shard["rows"]) for shard in shards)
    if rows != expected_rows:
        raise ValueError(f"source manifest incomplete: {source['name']}={rows}/{expected_rows}")
    return {
        "format": "tr-hash-token-mixture-v1",
        "source": source["name"],
        "bucket": source["bucket"],
        "selection": source.get("selection", "agentic"),
        "weight": source["weight"],
        "input": {
            key: source[key]
            for key in ("source_type", "dataset_id", "config_name", "split", "revision")
            if key in source
        },
        "license_audit": source["license_audit"],
        "seq_len": seq_len,
        "dtype": "uint16",
        "rows": rows,
        "trained_tokens": rows * seq_len,
        "shards": [
            {
                "file": Path(str(shard["repo_path"])).name,
                "rows": int(shard["rows"]),
                "tokens": int(shard["tokens"]),
                "bytes": int(shard["bytes"]),
                "sha256": shard["sha256"],
            }
            for shard in shards
        ],
    }


def _mixture_manifest(
    *,
    config: Mapping[str, Any],
    rows_by_source: Mapping[str, int],
    actual_tokens: int,
    tokenizer_sha256: str,
) -> dict[str, Any]:
    seq_len = int(config["seq_len"])
    direct_materialization = bool(config.get("direct_materialization", False))
    return {
        "format": "tr-hash-token-mixture-v1",
        "schema": config.get("schema", SCHEMA),
        "dtype": "uint16",
        "seq_len": seq_len,
        "requested_tokens": int(config["target_tokens"]),
        "actual_tokens": actual_tokens,
        "global_batch_sequences": int(config["global_batch_sequences"]),
        "bucket_targets": config.get("bucket_targets"),
        "config_sha256": config_sha256(config),
        "tokenizer_sha256": tokenizer_sha256,
        "materialization_mode": (
            "direct_source_curated" if direct_materialization else "filtered_deduplicated"
        ),
        "exact_document_deduplication": not direct_materialization,
        "benchmark_decontamination": not direct_materialization,
        "agentic_signal_filtering": not direct_materialization,
        "sources": [
            {
                "name": source["name"],
                "bucket": source["bucket"],
                "weight": source["weight"],
                "rows": rows_by_source[str(source["name"])],
                "trained_tokens": rows_by_source[str(source["name"])] * seq_len,
                "manifest": f"corpora/{source['name']}/manifest.json",
            }
            for source in config["sources"]
        ],
    }


def _replay_plan(
    store: StateStore,
    *,
    config: Mapping[str, Any],
    curriculum: Mapping[str, Any],
    rows_by_source: Mapping[str, int],
    actual_tokens: int,
) -> dict[str, Any]:
    phase_rows = allocate_phase_rows(
        config=config,
        curriculum=curriculum,
        rows_by_source=rows_by_source,
    )
    cursors = {str(source["name"]): 0 for source in config["sources"]}
    source_shards = {
        str(source["name"]): store.shards(str(source["name"])) for source in config["sources"]
    }
    phases = []
    for phase in curriculum["phases"]:
        name = str(phase["name"])
        selections: dict[str, list[dict[str, Any]]] = {}
        for source_name, wanted_rows in phase_rows[name].items():
            consumed = 0
            selected = []
            shards = source_shards[source_name]
            while consumed < wanted_rows:
                cursor = cursors[source_name]
                if cursor >= len(shards):
                    raise ValueError(f"source {source_name} exhausted while creating phase {name}")
                shard = shards[cursor]
                rows = int(shard["rows"])
                if consumed + rows > wanted_rows:
                    raise ValueError(
                        f"shard crosses curriculum boundary: {source_name}/{shard['repo_path']}"
                    )
                selected.append({"file": Path(str(shard["repo_path"])).name, "rows": rows})
                consumed += rows
                cursors[source_name] += 1
            selections[source_name] = selected
        phases.append({"name": name, "passes": 1, "sources": selections})
    if any(cursors[name] != len(shards) for name, shards in source_shards.items()):
        raise ValueError("curriculum replay plan does not consume every published shard")
    return {
        "format": "tr-hash-token-replay-plan-v1",
        "schema": curriculum.get("schema"),
        "config_sha256": config_sha256(config),
        "curriculum_sha256": config_sha256(curriculum),
        "seq_len": int(config["seq_len"]),
        "selection_mode": "manifest_order",
        "row_alignment": int(config["global_batch_sequences"]),
        "unique_tokens": actual_tokens,
        "trained_tokens": actual_tokens,
        "source_unique_tokens": {
            name: rows * int(config["seq_len"]) for name, rows in rows_by_source.items()
        },
        "source_passes": {name: 1 for name in rows_by_source},
        "phases": phases,
    }


def _phase_boundaries(
    *,
    config: Mapping[str, Any],
    curriculum: Mapping[str, Any] | None,
    rows_by_source: Mapping[str, int],
) -> dict[str, tuple[int, ...]]:
    if curriculum is None:
        return {name: (rows,) for name, rows in rows_by_source.items()}
    phase_rows = allocate_phase_rows(
        config=config,
        curriculum=curriculum,
        rows_by_source=rows_by_source,
    )
    result: dict[str, tuple[int, ...]] = {}
    for source_name, total_rows in rows_by_source.items():
        cumulative = 0
        boundaries = []
        for phase in curriculum["phases"]:
            cumulative += phase_rows[str(phase["name"])].get(source_name, 0)
            if cumulative:
                boundaries.append(cumulative)
        if not boundaries or boundaries[-1] != total_rows:
            raise ValueError(f"invalid curriculum boundary coverage for {source_name}")
        result[source_name] = tuple(dict.fromkeys(boundaries))
    return result


def _skip(iterator: Iterator[Mapping[str, Any]], count: int) -> None:
    for skipped in range(count):
        try:
            next(iterator)
        except StopIteration as error:
            raise RuntimeError(
                f"source exhausted while restoring scan position {skipped}/{count}"
            ) from error


@dataclass(frozen=True)
class PreparedBatch:
    """A source-ordered unit prepared by an I/O worker and merged centrally."""

    scanned: int
    counters: Counter[str]
    candidates: tuple[tuple[str, tuple[str, ...], str], ...]
    exhausted: bool = False
    error: BaseException | None = None


def _queue_packet(
    destination: queue.Queue[PreparedBatch],
    packet: PreparedBatch,
    stop: threading.Event,
) -> None:
    while not stop.is_set():
        try:
            destination.put(packet, timeout=0.5)
            return
        except queue.Full:
            continue


def _prepare_source_batches(
    *,
    source: Mapping[str, Any],
    source_index: int,
    restored_scanned: int,
    config: Mapping[str, Any],
    protected: Sequence[str],
    benchmark_index: Any,
    destination: queue.Queue[PreparedBatch],
    stop: threading.Event,
    source_iterator_factory: Callable[[Mapping[str, Any], int], Iterator[Mapping[str, Any]]]
    | None = None,
) -> None:
    """Read and filter one source without touching global state or output files."""

    iterator: Iterator[Mapping[str, Any]] | None = None
    try:
        source_seed = int(config.get("seed", 1729)) + source_index
        iterator = iter(
            source_iterator_factory(source, source_seed)
            if source_iterator_factory is not None
            else iter_source(source, seed=source_seed)
        )
        if restored_scanned:
            _skip(iterator, restored_scanned)
        candidate_limit = max(
            1,
            int(
                config.get(
                    "producer_candidate_batch_size",
                    config.get("tokenization_batch_size", 256),
                )
            ),
        )
        scan_limit = max(
            candidate_limit, int(config.get("producer_scan_batch_size", candidate_limit * 4))
        )
        while not stop.is_set():
            scanned = 0
            counters: Counter[str] = Counter()
            candidates: list[tuple[str, tuple[str, ...], str]] = []
            exhausted = False
            while scanned < scan_limit and len(candidates) < candidate_limit and not stop.is_set():
                try:
                    row = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                scanned += 1
                counters["scanned"] += 1
                text = row_text(row, source)
                if not text:
                    counters["missing_text"] += 1
                    continue
                selection = str(source.get("selection", "agentic"))
                if selection == "direct":
                    candidates.append((text, (), ""))
                    continue
                if selection == "staged":
                    digest = str(row.get("content_sha256", ""))
                    if not digest or digest != content_sha256(text):
                        raise ValueError(
                            f"staged candidate digest mismatch in source {source['name']}"
                        )
                    signals = tuple(str(value) for value in row.get("agentic_signals", ()))
                    candidates.append((text, signals, digest))
                    continue
                text = normalize_text(text)
                rejected = quality_rejection(
                    text,
                    min_chars=int(config.get("min_chars", 200)),
                    max_chars=int(config.get("max_chars", 100_000)),
                )
                if rejected:
                    counters[rejected] += 1
                    continue
                contaminated = benchmark_match(text, protected, benchmark_index)
                if contaminated:
                    counters[f"benchmark:{contaminated}"] += 1
                    continue
                signals: tuple[str, ...] = ()
                if selection == "agentic":
                    accepted, signals, _ = is_agentic_candidate(
                        text,
                        min_score=int(config.get("agentic_min_score", 4)),
                        min_signal_classes=int(config.get("agentic_min_signal_classes", 2)),
                    )
                    if not accepted:
                        counters["weak_agentic_signal"] += 1
                        continue
                elif selection == "agentic_trajectory":
                    signals = ("tool", "planning", "verification")
                candidates.append((text, signals, content_sha256(text)))
            if scanned or candidates or exhausted:
                _queue_packet(
                    destination,
                    PreparedBatch(
                        scanned=scanned,
                        counters=counters,
                        candidates=tuple(candidates),
                        exhausted=exhausted,
                    ),
                    stop,
                )
            if exhausted:
                return
    except BaseException as error:
        _queue_packet(
            destination,
            PreparedBatch(scanned=0, counters=Counter(), candidates=(), error=error),
            stop,
        )
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()


class _SourcePacker:
    """Pack one source while persisting every flushed batch as a restart point."""

    def __init__(
        self,
        *,
        source: Mapping[str, Any],
        tokenizer: Tokenizer,
        eos_token_id: int | None,
        store: StateStore,
        publisher: HubPublisher,
        work_dir: Path,
        seq_len: int,
        target_rows: int,
        rows_per_shard: int,
        boundaries: tuple[int, ...],
        progress_log_tokens: int,
        publish_state: Any,
    ) -> None:
        self.source = source
        self.name = str(source["name"])
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id
        self.store = store
        self.publisher = publisher
        self.work_dir = work_dir
        self.seq_len = seq_len
        self.target_rows = target_rows
        self.rows_per_shard = rows_per_shard
        self.boundaries = boundaries
        self.progress_log_tokens = max(1, progress_log_tokens)
        self.publish_state = publish_state
        self.progress = store.progress(self.name)
        self.mapping: np.memmap | None = None
        self.partial: Path | None = None
        self.final: Path | None = None
        self.token_count = 0
        self.shard_rows = 0
        self.shard_index = len(store.shards(self.name))
        self.shard_started_at = time.monotonic()
        self.next_progress_position = self.progress_log_tokens

    @property
    def complete(self) -> bool:
        return int(self.progress["rows_done"]) >= self.target_rows

    def _shard_shape(self) -> tuple[int, int]:
        next_boundary = next(
            boundary for boundary in self.boundaries if boundary > self.progress["rows_done"]
        )
        rows = min(
            self.rows_per_shard,
            self.target_rows - int(self.progress["rows_done"]),
            next_boundary - int(self.progress["rows_done"]),
        )
        return rows, rows * self.seq_len + 1

    def _open(self) -> None:
        if self.complete or self.mapping is not None:
            return
        self.shard_index = len(self.store.shards(self.name))
        self.shard_rows, self.token_count = self._shard_shape()
        shard_dir = self.work_dir / "pending" / self.name
        shard_dir.mkdir(parents=True, exist_ok=True)
        for completed in self.store.shards(self.name):
            (shard_dir / Path(str(completed["repo_path"])).name).unlink(missing_ok=True)
        self.partial = shard_dir / f"tokens-{self.shard_index:05d}.bin.partial"
        self.final = self.partial.with_suffix("")
        position = int(self.progress.get("partial_position", 0))
        expected_bytes = self.token_count * np.dtype(np.uint16).itemsize
        existing = self.partial if self.partial.is_file() else self.final
        if position:
            if not existing.is_file() or existing.stat().st_size != expected_bytes:
                raise RuntimeError(
                    f"missing or invalid partial shard for {self.name}: "
                    f"position={position:,} expected_bytes={expected_bytes:,}"
                )
            self.mapping = np.memmap(
                existing, mode="r+", dtype=np.uint16, shape=(self.token_count,)
            )
        else:
            self.partial.unlink(missing_ok=True)
            self.final.unlink(missing_ok=True)
            self.mapping = np.memmap(
                self.partial, mode="w+", dtype=np.uint16, shape=(self.token_count,)
            )
            if self.progress["last_token"] is not None:
                self.mapping[0] = int(self.progress["last_token"])
                self.mapping.flush()
                self.progress["partial_position"] = 1
                self.store.begin()
                self.store.save_progress(source=self.name, progress=self.progress)
                position = 1
        self.shard_started_at = time.monotonic()
        self.next_progress_position = (
            position // self.progress_log_tokens + 1
        ) * self.progress_log_tokens

    def _write_carry(self) -> None:
        self._open()
        if self.mapping is None:
            return
        position = int(self.progress["partial_position"])
        carry = np.asarray(self.progress["carry"], dtype=np.uint16)
        if carry.size:
            take = min(self.token_count - position, carry.size)
            self.mapping[position : position + take] = carry[:take]
            position += take
            self.progress["carry"] = carry[take:].copy()
        self.mapping.flush()
        self.progress["partial_position"] = position
        self.store.begin()
        self.store.save_progress(source=self.name, progress=self.progress)
        self._log_progress(position)

    def _log_progress(self, position: int) -> None:
        if position < self.next_progress_position and position < self.token_count:
            return
        elapsed = max(time.monotonic() - self.shard_started_at, 1e-9)
        rate = position / elapsed
        eta = (self.token_count - position) / max(rate, 1e-9)
        LOGGER.info(
            "%s shard %05d progress: %.2f%% tokens=%s/%s rate=%s tok/s "
            "eta=%.1f min scanned=%s retained=%s",
            self.name,
            self.shard_index,
            100.0 * position / self.token_count,
            f"{position:,}",
            f"{self.token_count:,}",
            f"{rate:,.0f}",
            eta / 60.0,
            f"{self.progress['scanned']:,}",
            f"{self.progress['counters']['retained']:,}",
        )
        self.next_progress_position = (
            position // self.progress_log_tokens + 1
        ) * self.progress_log_tokens

    def _finalize(self) -> None:
        if self.mapping is None or int(self.progress["partial_position"]) != self.token_count:
            return
        self.mapping.flush()
        last_token = int(self.mapping[-1])
        del self.mapping
        self.mapping = None
        assert self.partial is not None and self.final is not None
        if self.partial.is_file():
            os.replace(self.partial, self.final)
        published = self.publisher.publish_file(
            self.final, f"corpora/{self.name}/{self.final.name}"
        )
        shard = {
            **published,
            "shard_index": self.shard_index,
            "rows": self.shard_rows,
            "tokens": self.token_count,
        }
        self.progress["rows_done"] += self.shard_rows
        self.progress["last_token"] = last_token
        self.progress["partial_position"] = 0
        self.store.begin()
        self.store.commit_shard(
            source=self.name,
            scanned=self.progress["scanned"],
            rows_done=self.progress["rows_done"],
            source_tokens=self.progress["source_tokens"],
            last_token=last_token,
            carry=np.asarray(self.progress["carry"], dtype=np.uint16),
            counters=self.progress["counters"],
            signals=self.progress["signals"],
            shard=shard,
            partial_position=0,
        )
        self.publish_state()
        self.final.unlink(missing_ok=True)
        LOGGER.info(
            "%s shard %05d verified and evicted: %.3fB / %.3fB tokens",
            self.name,
            self.shard_index,
            self.progress["rows_done"] * self.seq_len / 1e9,
            self.target_rows * self.seq_len / 1e9,
        )

    def restore(self) -> int:
        """Finish a previously flushed full shard before reading more source rows."""

        self._open()
        finalized = 0
        if self.mapping is not None and int(self.progress["partial_position"]) == self.token_count:
            self._finalize()
            finalized = 1
        while not self.complete and np.asarray(self.progress["carry"]).size:
            self._write_carry()
            if int(self.progress["partial_position"]) == self.token_count:
                self._finalize()
                finalized += 1
            else:
                break
        return finalized

    def consume(self, batch: PreparedBatch) -> int:
        if self.complete:
            return 0
        self._open()
        self.store.begin()
        try:
            self.progress["scanned"] += batch.scanned
            self.progress["counters"].update(batch.counters)
            texts: list[str] = []
            signal_sets: list[tuple[str, ...]] = []
            for text, signals, digest in batch.candidates:
                if digest and not self.store.reserve_digest(digest):
                    self.progress["counters"]["exact_duplicate"] += 1
                    continue
                texts.append(text)
                signal_sets.append(signals)
            pieces: list[np.ndarray] = []
            if texts:
                encodings = self.tokenizer.encode_batch(texts, add_special_tokens=False)
                for encoding, signals in zip(encodings, signal_sets, strict=True):
                    ids = list(encoding.ids)
                    if self.eos_token_id is not None:
                        ids.append(self.eos_token_id)
                    piece = np.asarray(ids, dtype=np.uint16)
                    pieces.append(piece)
                    self.progress["source_tokens"] += piece.size
                    self.progress["counters"]["retained"] += 1
                    for signal in signals:
                        self.progress["signals"][signal] += 1
            carry = np.asarray(self.progress["carry"], dtype=np.uint16)
            arrays = ([carry] if carry.size else []) + pieces
            if arrays:
                self.progress["carry"] = (
                    arrays[0].copy() if len(arrays) == 1 else np.concatenate(arrays)
                )
            assert self.mapping is not None
            position = int(self.progress["partial_position"])
            carry = np.asarray(self.progress["carry"], dtype=np.uint16)
            if carry.size:
                take = min(self.token_count - position, carry.size)
                self.mapping[position : position + take] = carry[:take]
                position += take
                self.progress["carry"] = carry[take:].copy()
            self.mapping.flush()
            self.progress["partial_position"] = position
            self.store.save_progress(source=self.name, progress=self.progress)
        except Exception:
            self.store.rollback()
            raise
        self._log_progress(int(self.progress["partial_position"]))
        finalized = 0
        while not self.complete and int(self.progress["partial_position"]) == self.token_count:
            self._finalize()
            finalized += 1
            if not self.complete and np.asarray(self.progress["carry"]).size:
                self._write_carry()
            else:
                break
        return finalized

    def close(self) -> None:
        if self.mapping is not None:
            self.mapping.flush()
            del self.mapping
            self.mapping = None


def _parallel_source_groups(
    sources: Sequence[Mapping[str, Any]], width: int
) -> list[list[Mapping[str, Any]]]:
    """Create deterministic, bucket-local groups across distinct upstreams."""

    if width < 1:
        raise ValueError("parallel source width must be positive")
    groups: list[list[Mapping[str, Any]]] = []
    start = 0
    while start < len(sources):
        bucket = str(sources[start]["bucket"])
        stop = start
        while stop < len(sources) and str(sources[stop]["bucket"]) == bucket:
            stop += 1
        pending = list(sources[start:stop])
        while pending:
            group: list[Mapping[str, Any]] = []
            used_upstreams: set[str] = set()
            deferred: list[Mapping[str, Any]] = []
            for source in pending:
                upstream = str(
                    source.get("dataset_id")
                    or source.get("archive_url")
                    or source.get("path")
                    or source["name"]
                )
                if len(group) < width and upstream not in used_upstreams:
                    group.append(source)
                    used_upstreams.add(upstream)
                else:
                    deferred.append(source)
            if len(group) < width:
                take = min(width - len(group), len(deferred))
                group.extend(deferred[:take])
                deferred = deferred[take:]
            groups.append(group)
            pending = deferred
        start = stop
    return groups


def _direct_parallel_source_groups(
    sources: Sequence[Mapping[str, Any]], width: int
) -> list[list[Mapping[str, Any]]]:
    """Interleave buckets for direct builds where global dedup order is irrelevant."""

    if width < 1:
        raise ValueError("parallel source width must be positive")
    bucket_order: list[str] = []
    queues: dict[str, list[Mapping[str, Any]]] = {}
    for source in sources:
        bucket = str(source["bucket"])
        if bucket not in queues:
            bucket_order.append(bucket)
            queues[bucket] = []
        queues[bucket].append(source)

    ordered: list[Mapping[str, Any]] = []
    while any(queues.values()):
        for bucket in bucket_order:
            if queues[bucket]:
                ordered.append(queues[bucket].pop(0))
    return [ordered[start : start + width] for start in range(0, len(ordered), width)]


def build(
    *,
    config: Mapping[str, Any],
    tokenizer_path: Path,
    work_dir: Path,
    publisher: HubPublisher,
    curriculum: Mapping[str, Any] | None = None,
    only_source: str | None = None,
    max_shards: int | None = None,
    source_iterator_factory: Callable[[Mapping[str, Any], int], Iterator[Mapping[str, Any]]]
    | None = None,
) -> dict[str, Any]:
    sources = list(config["sources"])
    validate_config(config)
    all_sources = sources
    target_tokens = int(config["target_tokens"])
    seq_len = int(config["seq_len"])
    total_rows, actual_tokens, all_rows_by_source = allocate_rows(
        target_tokens=target_tokens,
        seq_len=seq_len,
        global_batch_sequences=int(config["global_batch_sequences"]),
        sources=all_sources,
    )
    del total_rows
    if only_source is not None:
        selected = [source for source in sources if source["name"] == only_source]
        if not selected:
            raise ValueError(f"unknown source: {only_source}")
        sources = selected

    rows_by_source = all_rows_by_source
    source_indexes = {str(source["name"]): index for index, source in enumerate(all_sources)}
    rows_per_shard = max(1, int(config["shard_trained_tokens"]) // seq_len)
    boundaries = _phase_boundaries(
        config=config,
        curriculum=curriculum,
        rows_by_source=rows_by_source,
    )
    validate_tokenizer_contract(config, tokenizer_path)
    tokenizer, eos_token_id = _tokenizer(tokenizer_path)
    tokenizer_json = (
        tokenizer_path / "tokenizer.json" if tokenizer_path.is_dir() else tokenizer_path
    )
    tokenizer_digest = sha256_file(tokenizer_json)
    protected = tuple(config.get("protected_benchmarks", DEFAULT_PROTECTED_BENCHMARKS))
    benchmark_index = build_benchmark_index(tuple(config.get("protected_benchmark_sources", ())))
    store = StateStore(work_dir / "state.sqlite3")
    digest = config_sha256(config)
    store.bind_run(config_digest=digest, tokenizer_digest=tokenizer_digest)
    publisher.publish_json(config, "_metadata/config.json", work_dir)
    if curriculum is not None:
        publisher.publish_json(curriculum, "_metadata/curriculum.json", work_dir)
    published_shards = 0

    def current_state() -> dict[str, Any]:
        return _state_payload(
            store,
            config=config,
            config_digest=digest,
            actual_tokens=actual_tokens,
            rows_by_source=rows_by_source,
            tokenizer_digest=tokenizer_digest,
            protected_prompt_count=benchmark_index.prompt_count,
            protected_index_digest=benchmark_index.fingerprint(),
        )

    def publish_state() -> None:
        publisher.publish_json(current_state(), "_state/state.json", work_dir)

    configured_parallel_sources = int(config.get("parallel_sources", 1))
    parallel_sources = max(
        1,
        int(os.environ.get("TR_HASH_SOURCE_PARALLELISM", configured_parallel_sources)),
    )
    queue_depth = max(1, int(config.get("producer_queue_depth", 2)))
    LOGGER.info(
        "source pipeline: parallel=%d deterministic_merge=round_robin queue_depth=%d",
        parallel_sources,
        queue_depth,
    )
    source_groups = (
        _direct_parallel_source_groups(sources, parallel_sources)
        if config.get("direct_materialization", False)
        else _parallel_source_groups(sources, parallel_sources)
    )
    for group_index, group in enumerate(source_groups):
        LOGGER.info(
            "source group %d/%d: %s",
            group_index + 1,
            len(source_groups),
            ", ".join(str(source["name"]) for source in group),
        )
        stop = threading.Event()
        contexts: list[
            tuple[Mapping[str, Any], _SourcePacker, queue.Queue[PreparedBatch], threading.Thread]
        ] = []
        try:
            for source in group:
                name = str(source["name"])
                packer = _SourcePacker(
                    source=source,
                    tokenizer=tokenizer,
                    eos_token_id=eos_token_id,
                    store=store,
                    publisher=publisher,
                    work_dir=work_dir,
                    seq_len=seq_len,
                    target_rows=rows_by_source[name],
                    rows_per_shard=rows_per_shard,
                    boundaries=boundaries[name],
                    progress_log_tokens=int(config.get("progress_log_tokens", 25_000_000)),
                    publish_state=publish_state,
                )
                published_shards += packer.restore()
                if packer.complete:
                    publisher.publish_json(
                        _source_manifest(
                            store,
                            source=source,
                            seq_len=seq_len,
                            expected_rows=rows_by_source[name],
                        ),
                        f"corpora/{name}/manifest.json",
                        work_dir,
                    )
                    packer.close()
                    continue
                if max_shards is not None and published_shards >= max_shards:
                    return current_state()
                progress = store.progress(name)
                if progress["scanned"]:
                    LOGGER.info(
                        "%s: restoring deterministic scan position %s",
                        name,
                        f"{progress['scanned']:,}",
                    )
                packets: queue.Queue[PreparedBatch] = queue.Queue(maxsize=queue_depth)
                worker = threading.Thread(
                    target=_prepare_source_batches,
                    kwargs={
                        "source": source,
                        "source_index": source_indexes[name],
                        "restored_scanned": progress["scanned"],
                        "config": config,
                        "protected": protected,
                        "benchmark_index": benchmark_index,
                        "destination": packets,
                        "stop": stop,
                        "source_iterator_factory": source_iterator_factory,
                    },
                    name=f"source-{name}",
                    daemon=True,
                )
                worker.start()
                contexts.append((source, packer, packets, worker))

            active = list(contexts)
            while active:
                next_active = []
                for source, packer, packets, worker in active:
                    packet = packets.get()
                    if packet.error is not None:
                        raise RuntimeError(
                            f"source producer failed: {packer.name}"
                        ) from packet.error
                    published_shards += packer.consume(packet)
                    if max_shards is not None and published_shards >= max_shards:
                        stop.set()
                        return current_state()
                    if packer.complete:
                        publisher.publish_json(
                            _source_manifest(
                                store,
                                source=source,
                                seq_len=seq_len,
                                expected_rows=rows_by_source[packer.name],
                            ),
                            f"corpora/{packer.name}/manifest.json",
                            work_dir,
                        )
                        packer.close()
                        continue
                    if packet.exhausted:
                        raise RuntimeError(
                            f"source {packer.name} exhausted after "
                            f"{packer.progress['scanned']:,} records"
                        )
                    next_active.append((source, packer, packets, worker))
                active = next_active
        finally:
            stop.set()
            for _, packer, _, worker in contexts:
                packer.close()
                worker.join(timeout=1.0)

    state = current_state()
    complete = all(
        store.progress(str(source["name"]))["rows_done"] == rows_by_source[str(source["name"])]
        for source in config["sources"]
    )
    if complete:
        publisher.publish_json(
            _mixture_manifest(
                config=config,
                rows_by_source=rows_by_source,
                actual_tokens=actual_tokens,
                tokenizer_sha256=tokenizer_digest,
            ),
            "mixture_manifest.json",
            work_dir,
        )
        if curriculum is not None:
            publisher.publish_json(
                _replay_plan(
                    store,
                    config=config,
                    curriculum=curriculum,
                    rows_by_source=rows_by_source,
                    actual_tokens=actual_tokens,
                ),
                "pretrain_plan.json",
                work_dir,
            )
    else:
        publisher.publish_json(state, "_state/state.json", work_dir)
    return state


def main(
    *,
    default_config: str = "configs/agentic_pretraining/tr_hash_agentic_50b.json",
    default_work_dir: str = "artifacts/tr_hash_agentic_50b_build",
    default_hf_repo: str = "AETHORIA-AI/TR-HASH-Agentic-Pretraining-50B",
    default_repo_prefix: str = "production",
    default_dataset_card: str | None = None,
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--curriculum")
    parser.add_argument("--tokenizer")
    parser.add_argument("--work-dir", default=default_work_dir)
    parser.add_argument("--hf-repo", default=default_hf_repo)
    parser.add_argument("--repo-prefix", default=default_repo_prefix)
    parser.add_argument("--target-tokens", type=int)
    parser.add_argument("--shard-trained-tokens", type=int)
    parser.add_argument("--only-source")
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--create-private-repo", action="store_true")
    parser.add_argument(
        "--direct-source-curated",
        action="store_true",
        help=(
            "tokenize pinned curated sources directly without per-document filtering, "
            "benchmark decontamination, or global exact deduplication"
        ),
    )
    parser.add_argument("--dataset-card", default=default_dataset_card)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.direct_source_curated:
        config = make_direct_source_curated_config(config)
    if args.target_tokens is not None:
        config["target_tokens"] = args.target_tokens
    if args.shard_trained_tokens is not None:
        config["shard_trained_tokens"] = args.shard_trained_tokens
    validate_config(config)
    curriculum = None
    if args.curriculum:
        curriculum = json.loads(Path(args.curriculum).read_text(encoding="utf-8"))
        validate_curriculum(config, curriculum)
    if args.validate_only:
        _, actual_tokens, rows = allocate_rows(
            target_tokens=int(config["target_tokens"]),
            seq_len=int(config["seq_len"]),
            global_batch_sequences=int(config["global_batch_sequences"]),
            sources=config["sources"],
        )
        print(
            json.dumps(
                {
                    "schema": config.get("schema"),
                    "requested_tokens": config["target_tokens"],
                    "packed_tokens": actual_tokens,
                    "bucket_targets": config.get("bucket_targets"),
                    "source_rows": rows,
                    "tokenizer_status": config.get("tokenizer_contract", {}).get("status"),
                },
                indent=2,
            )
        )
        return
    if not args.tokenizer:
        parser.error("--tokenizer is required unless --validate-only is used")
    publisher = HubPublisher(
        args.hf_repo,
        args.repo_prefix,
        os.environ.get("HF_TOKEN"),
        create_private_repo=args.create_private_repo,
    )
    if args.dataset_card:
        publisher.publish_root_file(Path(args.dataset_card), "README.md")
    state = build(
        config=config,
        tokenizer_path=Path(args.tokenizer),
        work_dir=Path(args.work_dir),
        publisher=publisher,
        curriculum=curriculum,
        only_source=args.only_source,
        max_shards=args.max_shards,
    )
    print(
        json.dumps({"actual_tokens": state["actual_tokens"], "sources": state["sources"]}, indent=2)
    )


if __name__ == "__main__":
    main()
