"""Filter restart-safe candidate shards, then pack verified local candidates.

The publisher is injected explicitly. No remote dataset launcher or acquisition
backend is selected here. Gzip bytes are deterministic; only matching publisher
receipts commit candidate cursors. Callers own independent remote readback.
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import math
import os
import sqlite3
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from complexity._optional import require_optional

from .agentic_corpus import (
    DEFAULT_PROTECTED_BENCHMARKS,
    BenchmarkContaminationIndex,
    benchmark_match,
    build_benchmark_index,
    config_sha256,
    content_sha256,
    is_agentic_candidate,
    iter_source,
    local_source_paths,
    normalize_text,
    quality_rejection,
    source_row_rejection,
    validate_sources,
)
from .agentic_plans import validate_curriculum
from .agentic_sharding import _positive, build, row_text, sha256_file, validate_config

LOGGER = logging.getLogger(__name__)
SCHEMA = "complexity-candidates-v1"


def effective_stage_config(
    config: Mapping[str, Any],
    *,
    only_sources: Sequence[str] = (),
    target_tokens_per_source: int | None = None,
    shuffle_buffer: int | None = None,
) -> dict[str, Any]:
    """Return a self-consistent full or explicitly bounded pilot config."""

    result = deepcopy(dict(config))
    requested = set(only_sources)
    if requested:
        available = {str(source["name"]) for source in result["sources"]}
        missing = requested - available
        if missing:
            raise ValueError(f"unknown pilot sources: {sorted(missing)}")
        result["sources"] = [
            source for source in result["sources"] if str(source["name"]) in requested
        ]
    if target_tokens_per_source is not None:
        _positive(target_tokens_per_source, "pilot target tokens per source")
        for source in result["sources"]:
            source["target_tokens"] = target_tokens_per_source
    if shuffle_buffer is not None:
        _positive(shuffle_buffer, "shuffle buffer")
        for source in result["sources"]:
            if source.get("source_type") != "software_heritage_stack_edu":
                source["shuffle_buffer"] = shuffle_buffer
    if requested or target_tokens_per_source is not None or shuffle_buffer is not None:
        total = sum(int(source["target_tokens"]) for source in result["sources"])
        result["target_tokens"] = total
        buckets: Counter[str] = Counter()
        for source in result["sources"]:
            buckets[str(source["bucket"])] += int(source["target_tokens"])
            source["weight"] = int(source["target_tokens"]) / total
        result["bucket_targets"] = dict(buckets)
        result["pilot"] = True
    return result


class CandidateState:
    """Per-source completed-shard state; the active shard is intentionally disposable."""

    def __init__(self, path: Path, *, source: str, config_digest: str, tokenizer_digest: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=120)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS seen (digest BLOB PRIMARY KEY) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS progress (
                source TEXT PRIMARY KEY,
                scanned INTEGER NOT NULL,
                retained_tokens INTEGER NOT NULL,
                retained_records INTEGER NOT NULL,
                counters TEXT NOT NULL,
                signals TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS shards (
                shard_index INTEGER PRIMARY KEY,
                repo_path TEXT NOT NULL,
                records INTEGER NOT NULL,
                reference_tokens INTEGER NOT NULL,
                bytes INTEGER NOT NULL,
                sha256 TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """
        )
        expected = {
            "schema": SCHEMA,
            "source": source,
            "config_sha256": config_digest,
            "tokenizer_sha256": tokenizer_digest,
        }
        existing = dict(self.connection.execute("SELECT key, value FROM metadata"))
        conflicts = {
            key: (existing[key], value)
            for key, value in expected.items()
            if key in existing and existing[key] != value
        }
        if conflicts:
            self.connection.close()
            raise ValueError(f"candidate state belongs to another build: {conflicts}")
        self.connection.executemany(
            "INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)", expected.items()
        )
        self.connection.commit()

    def progress(self, source: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT scanned, retained_tokens, retained_records, counters, signals "
            "FROM progress WHERE source=?",
            (source,),
        ).fetchone()
        if row is None:
            return {
                "scanned": 0,
                "retained_tokens": 0,
                "retained_records": 0,
                "counters": Counter(),
                "signals": Counter(),
            }
        return {
            "scanned": int(row[0]),
            "retained_tokens": int(row[1]),
            "retained_records": int(row[2]),
            "counters": Counter(json.loads(row[3])),
            "signals": Counter(json.loads(row[4])),
        }

    def seen(self) -> set[str]:
        return {bytes(row[0]).hex() for row in self.connection.execute("SELECT digest FROM seen")}

    def shards(self) -> list[dict[str, Any]]:
        names = ("shard_index", "repo_path", "records", "reference_tokens", "bytes", "sha256")
        return [
            dict(zip(names, row, strict=True))
            for row in self.connection.execute(
                "SELECT shard_index, repo_path, records, reference_tokens, bytes, sha256 "
                "FROM shards ORDER BY shard_index"
            )
        ]

    def commit_shard(
        self,
        *,
        source: str,
        progress: Mapping[str, Any],
        digests: Sequence[str],
        shard: Mapping[str, Any],
    ) -> None:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self.connection.executemany(
                "INSERT INTO seen(digest) VALUES (?)",
                ((bytes.fromhex(digest),) for digest in digests),
            )
            self.connection.execute(
                """
                INSERT INTO progress(
                    source, scanned, retained_tokens, retained_records, counters, signals
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source) DO UPDATE SET
                    scanned=excluded.scanned,
                    retained_tokens=excluded.retained_tokens,
                    retained_records=excluded.retained_records,
                    counters=excluded.counters,
                    signals=excluded.signals
                """,
                (
                    source,
                    int(progress["scanned"]),
                    int(progress["retained_tokens"]),
                    int(progress["retained_records"]),
                    json.dumps(dict(progress["counters"]), sort_keys=True),
                    json.dumps(dict(progress["signals"]), sort_keys=True),
                ),
            )
            self.connection.execute(
                "INSERT INTO shards VALUES (?, ?, ?, ?, ?, ?)",
                (
                    int(shard["shard_index"]),
                    str(shard["repo_path"]),
                    int(shard["records"]),
                    int(shard["reference_tokens"]),
                    int(shard["bytes"]),
                    str(shard["sha256"]),
                ),
            )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise


def _skip(iterator: Any, count: int) -> None:
    for position in range(count):
        try:
            next(iterator)
        except StopIteration as error:
            raise RuntimeError(
                f"source exhausted while restoring {position:,}/{count:,}"
            ) from error


def _manifest(
    *,
    source: Mapping[str, Any],
    target_tokens: int,
    candidate_target_tokens: int,
    tokenizer_digest: str,
    state: CandidateState,
    protected_prompt_count: int = 0,
    protected_index_sha256: str | None = None,
    input_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    progress = state.progress(str(source["name"]))
    return {
        "schema": SCHEMA,
        "source": source["name"],
        "bucket": source["bucket"],
        "selection": source.get("selection", "agentic"),
        "target_tokens": target_tokens,
        "candidate_target_tokens": candidate_target_tokens,
        "retained_tokens": progress["retained_tokens"],
        "retained_records": progress["retained_records"],
        "scanned_records": progress["scanned"],
        "tokenizer_sha256": tokenizer_digest,
        "config_sha256": dict(state.connection.execute("SELECT key, value FROM metadata"))[
            "config_sha256"
        ],
        "protected_prompt_count": protected_prompt_count,
        "protected_index_sha256": protected_index_sha256,
        "input_sha256": dict(input_sha256 or {}),
        "license_audit": source["license_audit"],
        "publication_verification": "publisher_receipt_matches_local_hash",
        "counters": dict(progress["counters"]),
        "signal_counts": dict(progress["signals"]),
        "shards": state.shards(),
        "complete": progress["retained_tokens"] >= candidate_target_tokens,
    }


def stage_source(
    *,
    source_index: int,
    source: Mapping[str, Any],
    config: Mapping[str, Any],
    tokenizer_path: Path,
    benchmark_index: BenchmarkContaminationIndex | None = None,
    work_dir: Path,
    publisher: Any,
) -> dict[str, Any]:
    """Stage one source. Only verified shards become durable progress."""

    require_optional("tokenizers", "tokenizers")
    from tokenizers import Tokenizer

    if type(source_index) is not int or source_index < 0:
        raise ValueError("source_index must be a nonnegative integer")
    validate_sources([{**source, "weight": 1.0}])
    name = source.get("name")
    if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("candidate source name must be one safe path component")
    if not source.get("license_audit"):
        raise ValueError("candidate source must declare a license audit")
    if source.get("selection", "agentic") not in {"quality", "agentic", "agentic_trajectory"}:
        raise ValueError("candidate staging requires a filtered selection")
    source_dir = work_dir / f"{source_index:02d}-{name}"
    if source_dir.is_symlink() or Path(work_dir).is_symlink():
        raise ValueError("candidate work directories must not be symlinks")
    source_dir.mkdir(parents=True, exist_ok=True)
    if (source_dir / "state.sqlite3").is_symlink():
        raise ValueError("candidate state paths must not be symlinks")
    tokenizer_json = (
        tokenizer_path / "tokenizer.json" if tokenizer_path.is_dir() else tokenizer_path
    )
    tokenizer_digest = sha256_file(tokenizer_json)
    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    benchmark_index = benchmark_index or build_benchmark_index(
        config.get("protected_benchmark_sources", ())
    )
    input_hashes = (
        {str(path.resolve()): sha256_file(path) for path in local_source_paths(source)}
        if "path" in source or "path_env" in source
        else {}
    )
    identity = {
        "config": config,
        "source": source,
        "source_index": source_index,
        "inputs": input_hashes,
        "protected_index_sha256": benchmark_index.fingerprint(),
    }
    state = CandidateState(
        source_dir / "state.sqlite3",
        source=name,
        config_digest=config_sha256(identity),
        tokenizer_digest=tokenizer_digest,
    )
    try:
        progress = state.progress(name)
        seen = state.seen()
        target_tokens = _positive(source.get("target_tokens"), "source target_tokens")
        oversample = config.get("candidate_oversample", 1.05)
        if isinstance(oversample, bool) or not isinstance(oversample, (int, float)):
            raise ValueError("candidate_oversample must be a numeric fraction")
        if not math.isfinite(oversample) or oversample < 1:
            raise ValueError("candidate_oversample must be finite and at least one")
        candidate_target = max(target_tokens, math.ceil(target_tokens * oversample))
        shard_target = _positive(
            config.get("candidate_shard_tokens", 250_000_000), "candidate_shard_tokens"
        )
        tokenizer_batch = _positive(
            config.get("candidate_tokenization_batch_size", 512),
            "candidate_tokenization_batch_size",
        )
        protected = tuple(config.get("protected_benchmarks", DEFAULT_PROTECTED_BENCHMARKS))
        iterator = (
            iter(())
            if progress["retained_tokens"] >= candidate_target
            else iter(iter_source(source, seed=int(config.get("seed", 1729)) + source_index))
        )
        if progress["scanned"] and progress["retained_tokens"] < candidate_target:
            LOGGER.info("source=%s restore scanned=%s", name, f"{progress['scanned']:,}")
            _skip(iterator, progress["scanned"])
        partial = source_dir / "candidate.partial.jsonl.gz"
        if partial.is_symlink() or (source_dir / "state.sqlite3").is_symlink():
            raise ValueError("candidate state and partial paths must not be symlinks")
        partial.unlink(missing_ok=True)

        while progress["retained_tokens"] < candidate_target:
            shard_index = len(state.shards())
            shard_tokens = 0
            shard_records = 0
            shard_digests: list[str] = []
            shard_seen: set[str] = set()
            shard_counters: Counter[str] = Counter()
            shard_signals: Counter[str] = Counter()
            shard_scanned = 0
            with partial.open("wb") as raw_output:
                with gzip.GzipFile(
                    filename="",
                    mode="wb",
                    fileobj=raw_output,
                    compresslevel=3,
                    mtime=0,
                ) as compressed_output:
                    with io.TextIOWrapper(compressed_output, encoding="utf-8") as output:
                        while (
                            shard_tokens < shard_target
                            and progress["retained_tokens"] + shard_tokens < candidate_target
                        ):
                            candidates: list[tuple[str, tuple[str, ...], int, str, Any]] = []
                            while len(candidates) < tokenizer_batch:
                                try:
                                    row = next(iterator)
                                except StopIteration as error:
                                    if candidates:
                                        break
                                    raise RuntimeError(
                                        f"source {name} exhausted at "
                                        f"{progress['retained_tokens'] + shard_tokens:,}/"
                                        f"{candidate_target:,} candidate tokens"
                                    ) from error
                                shard_scanned += 1
                                shard_counters["scanned"] += 1
                                if not isinstance(row, Mapping):
                                    raise ValueError("candidate source iterator must yield objects")
                                licensed = source_row_rejection(row, source)
                                if licensed:
                                    shard_counters[licensed] += 1
                                    continue
                                raw = row_text(row, source)
                                if not raw:
                                    shard_counters["missing_text"] += 1
                                    continue
                                text = normalize_text(raw)
                                rejected = quality_rejection(
                                    text,
                                    min_chars=int(config.get("min_chars", 200)),
                                    max_chars=int(config.get("max_chars", 100_000)),
                                )
                                if rejected:
                                    shard_counters[rejected] += 1
                                    continue
                                contaminated = benchmark_match(text, protected, benchmark_index)
                                if contaminated:
                                    shard_counters[f"benchmark:{contaminated}"] += 1
                                    continue
                                digest = content_sha256(text)
                                if digest in seen or digest in shard_seen:
                                    shard_counters["exact_duplicate"] += 1
                                    continue
                                signals: tuple[str, ...] = ()
                                score = 0
                                selection = str(source.get("selection", "agentic"))
                                if selection == "agentic":
                                    accepted, signals, score = is_agentic_candidate(
                                        text,
                                        min_score=int(config.get("agentic_min_score", 4)),
                                        min_signal_classes=int(
                                            config.get("agentic_min_signal_classes", 2)
                                        ),
                                    )
                                    if not accepted:
                                        shard_counters["weak_agentic_signal"] += 1
                                        continue
                                elif selection == "agentic_trajectory":
                                    signals = ("tool", "planning", "verification")
                                    score = 6
                                record_id = next(
                                    (
                                        row[field]
                                        for field in source.get(
                                            "record_id_fields",
                                            ("_source_record_id", "id", "url", "blob_id"),
                                        )
                                        if row.get(field) is not None
                                    ),
                                    None,
                                )
                                shard_seen.add(digest)
                                candidates.append((text, signals, score, digest, record_id))

                            encodings = tokenizer.encode_batch(
                                [candidate[0] for candidate in candidates],
                                add_special_tokens=False,
                            )
                            for candidate, encoding in zip(candidates, encodings, strict=True):
                                text, signals, score, digest, record_id = candidate
                                token_count = len(encoding.ids)
                                if not token_count:
                                    shard_counters["empty_tokenization"] += 1
                                    continue
                                output.write(
                                    json.dumps(
                                        {
                                            "text": text,
                                            "source": name,
                                            "bucket": source["bucket"],
                                            "agentic_score": score,
                                            "agentic_signals": signals,
                                            "content_sha256": digest,
                                            "source_record_id": record_id,
                                            "reference_tokens": token_count,
                                        },
                                        ensure_ascii=False,
                                        separators=(",", ":"),
                                    )
                                    + "\n"
                                )
                                shard_tokens += token_count
                                shard_records += 1
                                shard_digests.append(digest)
                                shard_counters["retained"] += 1
                                for signal in signals:
                                    shard_signals[signal] += 1

            relative = f"_candidates/{name}/candidate-{shard_index:05d}.jsonl.gz"
            if (
                any(sha256_file(Path(path)) != digest for path, digest in input_hashes.items())
                or sha256_file(tokenizer_json) != tokenizer_digest
            ):
                raise ValueError("candidate input or tokenizer changed during staging")
            with partial.open("rb") as stream:
                os.fsync(stream.fileno())
            expected = {
                "repo_path": relative,
                "bytes": partial.stat().st_size,
                "sha256": sha256_file(partial),
            }
            published = publisher.publish_file(partial, relative)
            if not isinstance(published, Mapping) or any(
                published.get(key) != value for key, value in expected.items()
            ):
                raise ValueError("publisher receipt does not match the complete candidate shard")
            next_progress = {
                "scanned": progress["scanned"] + shard_scanned,
                "retained_tokens": progress["retained_tokens"] + shard_tokens,
                "retained_records": progress["retained_records"] + shard_records,
                "counters": progress["counters"] + shard_counters,
                "signals": progress["signals"] + shard_signals,
            }
            shard = {
                **published,
                "shard_index": shard_index,
                "records": shard_records,
                "reference_tokens": shard_tokens,
            }
            state.commit_shard(
                source=name,
                progress=next_progress,
                digests=shard_digests,
                shard=shard,
            )
            seen.update(shard_digests)
            progress = next_progress
            manifest = _manifest(
                source=source,
                target_tokens=target_tokens,
                candidate_target_tokens=candidate_target,
                tokenizer_digest=tokenizer_digest,
                state=state,
                protected_prompt_count=benchmark_index.prompt_count,
                protected_index_sha256=benchmark_index.fingerprint(),
                input_sha256=input_hashes,
            )
            publisher.publish_json(manifest, f"_candidates/{name}/manifest.json", source_dir)
            partial.unlink(missing_ok=True)
            LOGGER.info(
                "source=%s shard=%05d verified+evicted retained=%s/%s",
                name,
                shard_index,
                f"{progress['retained_tokens']:,}",
                f"{candidate_target:,}",
            )

        manifest = _manifest(
            source=source,
            target_tokens=target_tokens,
            candidate_target_tokens=candidate_target,
            tokenizer_digest=tokenizer_digest,
            state=state,
            protected_prompt_count=benchmark_index.prompt_count,
            protected_index_sha256=benchmark_index.fingerprint(),
            input_sha256=input_hashes,
        )
        publisher.publish_json(manifest, f"_candidates/{name}/manifest.json", source_dir)
        return manifest
    finally:
        state.connection.close()


def load_candidate_manifests(path: Path, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate source order, quotas and checksums before opening staged text."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("complete") is not True:
        raise ValueError("candidate manifest is not complete")
    entries = payload.get("sources")
    if not isinstance(entries, list) or any(not isinstance(item, dict) for item in entries):
        raise ValueError("candidate manifest sources must be a list of objects")
    expected = [source["name"] for source in config["sources"]]
    if [entry.get("source") for entry in entries] != expected:
        raise ValueError("candidate manifest source order differs from the canonical config")
    manifests = {entry["source"]: entry for entry in entries}
    paths: set[str] = set()
    for name, entry in manifests.items():
        if entry.get("complete") is not True:
            raise ValueError(f"candidate source is incomplete: {name}")
        target = _positive(entry.get("candidate_target_tokens"), "candidate target tokens")
        if _positive(entry.get("retained_tokens"), "retained tokens") < target:
            raise ValueError(f"candidate source is below its staged quota: {name}")
        shards = entry.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError(f"candidate source has no shards: {name}")
        if [shard.get("shard_index") for shard in shards] != list(range(len(shards))):
            raise ValueError(f"candidate shard sequence is not contiguous: {name}")
        for shard in shards:
            relative = shard.get("repo_path")
            if (
                not isinstance(relative, str)
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
            ):
                raise ValueError("candidate paths must be contained relative paths")
            if relative in paths:
                raise ValueError("candidate shard path appears more than once")
            paths.add(relative)
            _positive(shard.get("bytes"), "candidate bytes")
            _positive(shard.get("records"), "candidate records")
            _positive(shard.get("reference_tokens"), "candidate reference tokens")
            digest = shard.get("sha256")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise ValueError("candidate shard must declare a SHA-256 checksum")
        if sum(shard["reference_tokens"] for shard in shards) != entry["retained_tokens"]:
            raise ValueError(f"candidate shard token totals differ from retained tokens: {name}")
    return manifests


class CandidateShardReader:
    """Read hash-verified local gzip shards; remote acquisition stays explicit."""

    def __init__(self, tokenizer: Any | None = None) -> None:
        self.tokenizer = tokenizer

    def __call__(self, source: Mapping[str, Any], seed: int) -> Iterator[Mapping[str, Any]]:
        del seed
        for shard in source["candidate_shards"]:
            local = shard.get("local_path")
            if not isinstance(local, str) or not local:
                raise ValueError(
                    "materialize candidate shards locally before final packing; no remote acquisition backend is selected"
                )
            path = Path(local)
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"candidate shard must be a regular local file: {path}")
            if path.stat().st_size != shard["bytes"] or sha256_file(path) != shard["sha256"]:
                raise ValueError(f"candidate SHA-256 or byte count mismatch: {shard['repo_path']}")
            records = tokens = 0
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if (
                        not isinstance(row, dict)
                        or row.get("source") != source["name"]
                        or row.get("bucket") != source["bucket"]
                    ):
                        raise ValueError(f"candidate identity mismatch at {path}:{line_number}")
                    text = row.get("text")
                    if (
                        not isinstance(text, str)
                        or not text
                        or row.get("content_sha256") != content_sha256(text)
                    ):
                        raise ValueError(
                            f"candidate content digest mismatch at {path}:{line_number}"
                        )
                    records += 1
                    count = _positive(row.get("reference_tokens"), "reference_tokens")
                    if (
                        self.tokenizer is not None
                        and len(self.tokenizer.encode(text, add_special_tokens=False).ids) != count
                    ):
                        raise ValueError(
                            f"candidate reference token count mismatch at {path}:{line_number}"
                        )
                    tokens += count
                    yield row
            if records != shard["records"] or tokens != shard["reference_tokens"]:
                raise ValueError(f"candidate record/token count mismatch: {path}")


def pack_candidates(
    *,
    config: Mapping[str, Any],
    curriculum: Mapping[str, Any],
    candidate_manifest_path: Path,
    tokenizer_path: Path,
    work_dir: Path,
    publisher: Any,
    repo_id: str | None = None,
    repo_prefix: str = "",
    candidate_cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Globally deduplicate verified staged text through the native sharder.

    Repository/cache arguments are retained as caller context only. They never
    instantiate a Hub client or grant publication permission.
    """
    del repo_id, repo_prefix, candidate_cache_dir
    validate_config(config)
    validate_curriculum(config, curriculum)
    manifests = load_candidate_manifests(candidate_manifest_path, config)
    token_path = Path(tokenizer_path)
    token_path = token_path / "tokenizer.json" if token_path.is_dir() else token_path
    tokenizer_digest = sha256_file(token_path)
    if any(manifest.get("tokenizer_sha256") != tokenizer_digest for manifest in manifests.values()):
        raise ValueError("candidate tokenizer SHA-256 differs from the final tokenizer")
    # Exhaust every shard during preflight, even when its token quota needs only
    # a prefix; malformed tail rows cannot gain a completed-corpus manifest.
    require_optional("tokenizers", "tokenizers")
    from tokenizers import Tokenizer

    reader = CandidateShardReader(Tokenizer.from_file(str(token_path)))
    derived = deepcopy(dict(config))
    derived["parallel_sources"] = 1
    derived["protected_benchmarks"] = []
    derived["protected_benchmark_sources"] = []
    derived["candidate_manifest_sha256"] = sha256_file(Path(candidate_manifest_path))
    derived["candidate_shard_sha256"] = {
        shard["local_path"]: shard["sha256"]
        for manifest in manifests.values()
        for shard in manifest["shards"]
        if isinstance(shard.get("local_path"), str)
    }
    if config.get("protected_benchmark_sources"):
        fingerprints = {manifest.get("protected_index_sha256") for manifest in manifests.values()}
        if (
            len(fingerprints) != 1
            or None in fingerprints
            or any(
                manifest.get("protected_prompt_count", 0) <= 0 for manifest in manifests.values()
            )
        ):
            raise ValueError(
                "candidate manifests do not prove a common populated decontamination index"
            )
    derived["candidate_provenance"] = {
        "protected_prompt_count": manifests[config["sources"][0]["name"]].get(
            "protected_prompt_count", 0
        ),
        "protected_index_sha256": manifests[config["sources"][0]["name"]].get(
            "protected_index_sha256"
        ),
        "original_config_sha256": config_sha256(config),
    }
    for source in derived["sources"]:
        source["selection"] = "staged"
        source["text_field"] = "text"
        source["candidate_shards"] = manifests[source["name"]]["shards"]
        for _ in reader(source, 0):
            pass
        for key in (
            "path",
            "path_env",
            "source_type",
            "dataset_id",
            "config_name",
            "revision",
            "messages_field",
            "tools_field",
        ):
            source.pop(key, None)
    return build(
        config=derived,
        tokenizer_path=tokenizer_path,
        work_dir=work_dir,
        publisher=publisher,
        curriculum=curriculum,
        source_iterator_factory=reader,
    )
