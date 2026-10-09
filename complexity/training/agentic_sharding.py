"""Restart-safe token shards with an explicit injected publisher.

Publication is optional caller orchestration, never inferred from a config.
Only matching publisher receipts commit shard progress; local files remain
until acknowledgement. No Hub client or training launcher is provided here.
"""

from __future__ import annotations

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
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .agentic_corpus import (
    DEFAULT_PROTECTED_BENCHMARKS,
    benchmark_match,
    build_benchmark_index,
    content_sha256,
    is_agentic_candidate,
    iter_source,
    normalize_text,
    quality_rejection,
    validate_sources,
)

LOGGER = logging.getLogger(__name__)


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

    def bind_run(
        self, *, config_digest: str, tokenizer_digest: str, input_digest: str | None = None
    ) -> None:
        expected = {"config_sha256": config_digest, "tokenizer_sha256": tokenizer_digest}
        if input_digest is not None:
            expected["input_sha256"] = input_digest
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
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
            raise ValueError("content digest must be a SHA-256 hex string")
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

    def close(self) -> None:
        self.connection.close()


class ShardPublisher(Protocol):
    """Caller-owned, idempotent publication with local-hash matching receipts."""

    def publish_file(self, path: Path, relative: str) -> Mapping[str, Any]: ...
    def publish_json(self, payload: Mapping[str, Any], relative: str, work_dir: Path) -> None: ...


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _positive(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def allocate_rows(
    *,
    target_tokens: int,
    seq_len: int,
    global_batch_sequences: int,
    sources: Sequence[Mapping[str, Any]],
) -> tuple[int, int, dict[str, int]]:
    """Round to complete updates and align each source quota to that update."""
    target = _positive(target_tokens, "target_tokens")
    length = _positive(seq_len, "seq_len")
    alignment = _positive(global_batch_sequences, "global_batch_sequences")
    if not sources:
        raise ValueError("allocation requires sources")
    names = [source.get("name") for source in sources]
    weights = [source.get("weight") for source in sources]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(
        names
    ):
        raise ValueError("allocation source names must be nonempty and unique")
    if any(
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(weight)
        or not 0 < weight <= 1
        for weight in weights
    ) or not math.isclose(sum(weights), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("allocation weights must be positive and sum to one")
    units = (target + length * alignment - 1) // (length * alignment)
    exact = [units * weight for weight in weights]
    counts = [math.floor(value) for value in exact]
    for index in sorted(
        range(len(counts)), key=lambda index: exact[index] - counts[index], reverse=True
    )[: units - sum(counts)]:
        counts[index] += 1
    rows = units * alignment
    return (
        rows,
        rows * length,
        {name: count * alignment for name, count in zip(names, counts, strict=True)},
    )


def skip_source_records(
    iterator: Iterator[Mapping[str, Any]], count: int, *, source_name: str, log_every: int = 50_000
) -> None:
    if type(count) is not int or count < 0:
        raise ValueError("restore count must be a nonnegative integer")
    _positive(log_every, "log_every")
    for completed in range(1, count + 1):
        if next(iterator, None) is None:
            raise RuntimeError(
                f"source exhausted while restoring scan position {completed - 1}/{count}"
            )
        if completed % log_every == 0 or completed == count:
            LOGGER.info(
                "%s restore progress: %.2f%% records=%s/%s",
                source_name,
                100 * completed / count,
                completed,
                count,
            )


def validate_config(config: Mapping[str, Any]) -> None:
    validate_sources(config.get("sources"))
    for key in ("target_tokens", "seq_len", "global_batch_sequences"):
        _positive(config.get(key), key)
    if "version" in config and (type(config["version"]) is not int or config["version"] != 1):
        raise ValueError("corpus config version must be 1")
    for key in (
        "shard_trained_tokens",
        "parallel_sources",
        "producer_queue_depth",
        "producer_candidate_batch_size",
        "producer_scan_batch_size",
        "tokenization_batch_size",
    ):
        if key in config:
            _positive(config[key], key)
    minimum = _positive(config.get("min_chars", 200), "min_chars")
    if _positive(config.get("max_chars", 100_000), "max_chars") < minimum:
        raise ValueError("max_chars must be at least min_chars")
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
        for value in explicit_targets:
            _positive(value, "source target_tokens")
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
        if selection not in {"quality", "agentic", "agentic_trajectory", "staged", "direct"}:
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
    direct_materialization = config.get("direct_materialization", False)
    if type(direct_materialization) is not bool:
        raise ValueError("direct_materialization must be a boolean")
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
        if not isinstance(bucket_targets, Mapping):
            raise ValueError("bucket_targets must be an object")
        for value in bucket_targets.values():
            _positive(value, "bucket target_tokens")
        if not all(value is not None for value in explicit_targets):
            raise ValueError("bucket targets require explicit source target_tokens")
        if sum(int(value) for value in bucket_targets.values()) != target_tokens:
            raise ValueError("bucket targets must sum to the corpus target")
        actual = Counter()
        for source in sources:
            actual[str(source["bucket"])] += int(source["target_tokens"])
        if dict(actual) != {str(key): int(value) for key, value in bucket_targets.items()}:
            raise ValueError(f"source targets disagree with bucket targets: {dict(actual)}")


def make_direct_source_curated_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Derive an explicitly unfiltered policy without changing source budgets.

    Batch/queue settings preserve the historical recipe as authoring data;
    the native builder currently executes sources sequentially.
    """

    direct = copy.deepcopy(dict(config))
    direct["schema"] = "complexity-source-curated-corpus-v1"
    direct["description"] = (
        "Explicit source-curated corpus with caller-supplied token budgets. "
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


def prepare_source_batches(
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
            skip_source_records(iterator, restored_scanned, source_name=str(source["name"]))
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


def parallel_source_groups(
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


def direct_parallel_source_groups(
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


def _fsync(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


class _SourcePacker:
    """Flush token/carry/dedup state before attempting publication."""

    def __init__(
        self,
        *,
        source: Mapping[str, Any],
        store: StateStore,
        publisher: ShardPublisher,
        work_dir: Path,
        seq_len: int,
        target_rows: int,
        rows_per_shard: int,
        publish_state: Callable[[], None],
        boundaries: tuple[int, ...] | None = None,
    ):
        self.source, self.store, self.publisher = source, store, publisher
        self.name, self.work_dir = source["name"], work_dir
        self.seq_len, self.target_rows, self.rows_per_shard = seq_len, target_rows, rows_per_shard
        self.publish_state, self.progress = publish_state, store.progress(self.name)
        self.boundaries = boundaries or (target_rows,)
        self.mapping = None
        self.path = None
        self.rows, self.size = 0, 0
        self.directory = work_dir / "pending" / self.name
        self.directory.mkdir(parents=True, exist_ok=True)
        for shard in store.shards(self.name):
            for suffix in ("", ".partial"):
                (self.directory / (Path(shard["repo_path"]).name + suffix)).unlink(missing_ok=True)

    @property
    def complete(self) -> bool:
        return self.progress["rows_done"] == self.target_rows

    def _open(self) -> None:
        if self.complete or self.mapping is not None:
            return
        self.rows = min(self.rows_per_shard, self.target_rows - self.progress["rows_done"])
        boundary = next(stop for stop in self.boundaries if stop > self.progress["rows_done"])
        self.rows = min(self.rows, boundary - self.progress["rows_done"])
        self.size = self.rows * self.seq_len + 1
        index = len(self.store.shards(self.name))
        final = self.directory / f"tokens-{index:05d}.bin"
        partial = final.with_suffix(".bin.partial")
        position = self.progress["partial_position"]
        if position:
            if not 0 < position <= self.size:
                raise ValueError("partial token position is outside its shard")
            self.path = partial if partial.is_file() else final
            if (
                self.path.is_symlink()
                or not self.path.is_file()
                or self.path.stat().st_size != self.size * 2
            ):
                raise RuntimeError("missing or invalid partial shard at recorded restart position")
            self.mapping = np.memmap(self.path, mode="r+", dtype="<u2", shape=(self.size,))
        else:
            if partial.is_symlink() or final.is_symlink():
                raise ValueError("pending shard paths must not be symlinks")
            partial.unlink(missing_ok=True)
            final.unlink(missing_ok=True)
            self.path = partial
            self.mapping = np.memmap(partial, mode="w+", dtype="<u2", shape=(self.size,))
            if self.progress["last_token"] is not None:
                self.mapping[0] = self.progress["last_token"]
                self.progress["partial_position"] = 1

    def _flush_progress(self) -> None:
        self.mapping.flush()
        _fsync(self.path)
        self.store.save_progress(source=self.name, progress=self.progress)

    def _finish(self) -> None:
        if self.progress["partial_position"] != self.size:
            raise RuntimeError("cannot publish an incomplete token shard")
        last = int(self.mapping[-1])
        self.mapping.flush()
        del self.mapping
        self.mapping = None
        final = self.path.with_suffix("") if self.path.suffix == ".partial" else self.path
        if self.path != final:
            os.replace(self.path, final)
        _fsync(final)
        relative = f"corpora/{self.name}/{final.name}"
        expected = {
            "repo_path": relative,
            "bytes": final.stat().st_size,
            "sha256": sha256_file(final),
        }
        receipt = self.publisher.publish_file(final, relative)
        if not isinstance(receipt, Mapping) or any(
            receipt.get(key) != value for key, value in expected.items()
        ):
            raise ValueError("publisher receipt does not match the complete local shard")
        shard = {
            **expected,
            "shard_index": len(self.store.shards(self.name)),
            "rows": self.rows,
            "tokens": self.size,
        }
        self.store.begin()
        self.store.commit_shard(
            source=self.name,
            scanned=self.progress["scanned"],
            rows_done=self.progress["rows_done"] + self.rows,
            source_tokens=self.progress["source_tokens"],
            last_token=last,
            carry=self.progress["carry"],
            counters=self.progress["counters"],
            signals=self.progress["signals"],
            shard=shard,
        )
        self.progress = self.store.progress(self.name)
        self.publish_state()
        final.unlink()

    def drain(self) -> None:
        while not self.complete:
            self._open()
            if self.progress["partial_position"] == self.size:
                self._finish()
                continue
            carry = self.progress["carry"]
            if not carry.size:
                break
            take = min(self.size - self.progress["partial_position"], carry.size)
            position = self.progress["partial_position"]
            self.mapping[position : position + take] = carry[:take]
            self.progress["partial_position"] += take
            self.progress["carry"] = carry[take:].copy()
            self.store.begin()
            self._flush_progress()

    def consume(
        self,
        *,
        tokens: Sequence[int],
        digest: str | None,
        scanned: int,
        rejected: str | None,
        signals: tuple[str, ...],
    ) -> None:
        self._open()
        self.store.begin()
        try:
            self.progress["scanned"] = scanned
            self.progress["counters"]["scanned"] += 1
            if digest is not None and not self.store.reserve_digest(digest):
                rejected = "exact_duplicate"
            if rejected:
                self.progress["counters"][rejected] += 1
            else:
                ids = np.asarray(tokens)
                if (
                    ids.ndim != 1
                    or ids.dtype.kind not in {"i", "u"}
                    or (ids.size and (ids.min() < 0 or ids.max() > 65535))
                ):
                    raise ValueError("token IDs must fit uint16 without wrapping")
                ids = ids.astype("<u2", copy=False)
                self.progress["carry"] = np.concatenate([self.progress["carry"], ids])
                self.progress["source_tokens"] += len(ids)
                self.progress["counters"]["retained"] += 1
                self.progress["signals"].update(signals)
            position = self.progress["partial_position"]
            carry = self.progress["carry"]
            take = min(self.size - position, carry.size)
            self.mapping[position : position + take] = carry[:take]
            self.progress["partial_position"] += take
            self.progress["carry"] = carry[take:].copy()
            self._flush_progress()
        except BaseException:
            self.store.rollback()
            raise
        self.drain()

    def close(self) -> None:
        if self.mapping is not None:
            self.mapping.flush()
            del self.mapping
            self.mapping = None


def build(
    *,
    config: Mapping[str, Any],
    tokenizer_path: Path,
    work_dir: Path,
    publisher: ShardPublisher,
    curriculum: Mapping[str, Any] | None = None,
    source_iterator_factory: Callable[[Mapping[str, Any], int], Iterator[Mapping[str, Any]]]
    | None = None,
) -> dict[str, Any]:
    """Filter/pack explicit source budgets; acknowledge, commit, then evict.

    The injected publisher must make repeated same-path/hash uploads idempotent.
    Receipt matching is checked here; independent remote readback belongs to the
    publisher, and is not established by this module's local fixture tests.
    """
    from tokenizers import Tokenizer

    from ..tokenizer.training import validate_tokenizer_contract
    from .agentic_corpus import config_sha256, local_source_paths, source_row_rejection
    from .agentic_plans import (
        curriculum_plan,
        mixture_manifest,
        phase_boundaries,
        validate_curriculum,
    )

    validate_config(config)
    validate_tokenizer_contract(config, Path(tokenizer_path))
    if curriculum is not None:
        validate_curriculum(config, curriculum)
    _, actual, quotas = allocate_rows(
        target_tokens=config["target_tokens"],
        seq_len=config["seq_len"],
        global_batch_sequences=config["global_batch_sequences"],
        sources=config["sources"],
    )
    boundaries = phase_boundaries(config=config, curriculum=curriculum, rows_by_source=quotas)
    token_path = Path(tokenizer_path)
    token_path = token_path / "tokenizer.json" if token_path.is_dir() else token_path
    tokenizer = Tokenizer.from_file(str(token_path))
    if not 0 < tokenizer.get_vocab_size(with_added_tokens=True) <= 65536:
        raise ValueError("uint16 token shards require at most 65,536 tokenizer IDs")
    token_config = token_path.parent / "tokenizer_config.json"
    eos = json.loads(token_config.read_text()).get("eos_token") if token_config.is_file() else None
    eos_id = tokenizer.token_to_id(eos) if isinstance(eos, str) else None
    benchmark_index = build_benchmark_index(config.get("protected_benchmark_sources", []))
    protected = config.get("protected_benchmarks", DEFAULT_PROTECTED_BENCHMARKS)
    inputs = {
        source["name"]: {
            str(path.resolve()): sha256_file(path) for path in local_source_paths(source)
        }
        for source in config["sources"]
        if "path" in source or "path_env" in source
    }
    token_digest, digest = sha256_file(token_path), config_sha256(config)
    work = Path(work_dir)
    store = StateStore(work / "state.sqlite3")
    try:
        store.bind_run(
            config_digest=digest,
            tokenizer_digest=token_digest,
            input_digest=config_sha256(
                {"local_sources": inputs, "protected_index": benchmark_index.fingerprint()}
            ),
        )
    except BaseException:
        store.close()
        raise
    rows_per_shard = max(1, config["shard_trained_tokens"] // config["seq_len"])

    def current_state():
        staged_provenance = config.get("candidate_provenance", {})
        return {
            "schema": "complexity-token-production-v1",
            "config_sha256": digest,
            "tokenizer_sha256": token_digest,
            "requested_tokens": config["target_tokens"],
            "actual_tokens": actual,
            "seq_len": config["seq_len"],
            "input_sha256": inputs,
            "protected_prompt_count": staged_provenance.get(
                "protected_prompt_count", benchmark_index.prompt_count
            ),
            "protected_index_sha256": staged_provenance.get("protected_index_sha256")
            or benchmark_index.fingerprint(),
            "candidate_manifest_sha256": config.get("candidate_manifest_sha256"),
            "publication_verification": "publisher_receipt_matches_local_hash",
            "execution_mode": "deterministic_sequential",
            "materialization_mode": (
                "direct_source_curated"
                if config.get("direct_materialization")
                else "filtered_deduplicated"
            ),
            "exact_document_deduplication": not config.get("direct_materialization", False),
            "benchmark_decontamination": not config.get("direct_materialization", False),
            "agentic_signal_filtering": not config.get("direct_materialization", False),
            "sources": [
                {
                    "name": source["name"],
                    "bucket": source["bucket"],
                    "weight": source["weight"],
                    "license_audit": source["license_audit"],
                    "target_rows": quotas[source["name"]],
                    "rows_done": store.progress(source["name"])["rows_done"],
                    "scanned": store.progress(source["name"])["scanned"],
                    "counters": dict(store.progress(source["name"])["counters"]),
                    "signal_counts": dict(store.progress(source["name"])["signals"]),
                    "shards": store.shards(source["name"]),
                }
                for source in config["sources"]
            ],
        }

    def publish_state():
        publisher.publish_json(current_state(), "_state/state.json", work)

    packer = None
    try:
        publisher.publish_json(config, "_metadata/config.json", work)
        if curriculum is not None:
            publisher.publish_json(curriculum, "_metadata/curriculum.json", work)
        for index, source in enumerate(config["sources"]):
            name = source["name"]
            if not quotas[name]:
                raise ValueError(f"source received zero complete optimizer updates: {name}")
            packer = _SourcePacker(
                source=source,
                store=store,
                publisher=publisher,
                work_dir=work,
                seq_len=config["seq_len"],
                target_rows=quotas[name],
                rows_per_shard=rows_per_shard,
                publish_state=publish_state,
                boundaries=boundaries[name],
            )
            packer.drain()
            if not packer.complete:
                seed = config.get("seed", 1729) + index
                iterator = (
                    source_iterator_factory(source, seed)
                    if source_iterator_factory
                    else iter_source(source, seed=seed)
                )
                skip_source_records(iterator, packer.progress["scanned"], source_name=name)
                scanned = packer.progress["scanned"]
                for row in iterator:
                    scanned += 1
                    if not isinstance(row, Mapping):
                        raise ValueError("source iterator must yield mappings")
                    raw = row_text(row, source)
                    selection = source.get(
                        "selection", "agentic" if source["bucket"] == "agentic" else "quality"
                    )
                    direct, staged = selection == "direct", selection == "staged"
                    text = raw if direct else normalize_text(raw)
                    rejected = (
                        None
                        if (direct or staged)
                        else source_row_rejection(row, source)
                        or quality_rejection(
                            text,
                            min_chars=config.get("min_chars", 200),
                            max_chars=config.get("max_chars", 100_000),
                        )
                    )
                    matched = (
                        None
                        if (direct or staged)
                        else benchmark_match(text, protected, benchmark_index)
                    )
                    rejected = rejected or (f"benchmark:{matched}" if matched else None)
                    candidate, signals, _ = is_agentic_candidate(
                        text,
                        min_score=config.get("agentic_min_score", 4),
                        min_signal_classes=config.get("agentic_min_signal_classes", 2),
                    )
                    agentic = selection == "agentic"
                    if agentic and not candidate:
                        rejected = rejected or "weak_agentic_signal"
                    if staged:
                        if row.get("content_sha256") != content_sha256(text):
                            raise ValueError(f"staged candidate digest mismatch in source {name}")
                        signals = tuple(row.get("agentic_signals", ()))
                    elif selection == "agentic_trajectory":
                        signals = ("tool", "planning", "verification")
                    ids = (
                        tokenizer.encode(text, add_special_tokens=False).ids if not rejected else []
                    )
                    if ids and eos_id is not None:
                        ids.append(eos_id)
                    if not rejected and not ids:
                        rejected = "empty_tokenization"
                    packer.consume(
                        tokens=ids,
                        digest=content_sha256(text) if not rejected and not direct else None,
                        scanned=scanned,
                        rejected=rejected,
                        signals=signals,
                    )
                    if packer.complete:
                        break
                if not packer.complete:
                    raise RuntimeError(f"source {name} exhausted before its token quota")
            packer.close()
            shards = store.shards(name)
            manifest = {
                "format": "tr-hash-token-mixture-v1",
                "source": name,
                "weight": source["weight"],
                "bucket": source["bucket"],
                "license_audit": source["license_audit"],
                "seq_len": config["seq_len"],
                "dtype": "uint16",
                "rows": quotas[name],
                "trained_tokens": quotas[name] * config["seq_len"],
                "shards": [
                    {
                        "file": Path(shard["repo_path"]).name,
                        **{key: shard[key] for key in ("rows", "tokens", "bytes", "sha256")},
                    }
                    for shard in shards
                ],
            }
            publisher.publish_json(manifest, f"corpora/{name}/manifest.json", work)
        state = current_state()
        if any(
            sha256_file(Path(path)) != expected
            for group in inputs.values()
            for path, expected in group.items()
        ):
            raise ValueError(
                "local input changed during tokenization; no complete mixture will be published"
            )
        if any(
            sha256_file(Path(path)) != expected
            for path, expected in config.get("candidate_shard_sha256", {}).items()
        ):
            raise ValueError(
                "candidate input changed during packing; no complete mixture will be published"
            )
        mixture = mixture_manifest(
            config=config,
            rows_by_source=quotas,
            actual_tokens=actual,
            tokenizer_sha256=token_digest,
        )
        publisher.publish_json(mixture, "mixture_manifest.json", work)
        if curriculum is not None:
            plan = curriculum_plan(
                store,
                config=config,
                curriculum=curriculum,
                rows_by_source=quotas,
                actual_tokens=actual,
            )
            publisher.publish_json(plan, "pretrain_plan.json", work)
        publish_state()
        return state
    finally:
        if packer is not None:
            packer.close()
        store.close()
