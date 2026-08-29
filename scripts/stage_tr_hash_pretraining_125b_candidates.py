#!/usr/bin/env python3
"""Stage filtered 125B source documents as verified, restart-safe gzip shards."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import os
import pickle
import sqlite3
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from scripts.build_agentic_pretraining_50b import (
    HubPublisher,
    config_sha256,
    row_text,
    sha256_file,
)
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

LOGGER = logging.getLogger("tr_hash_125b_candidates")
SCHEMA = "tr-hash-125b-candidates-v1"


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
        if target_tokens_per_source < 1:
            raise ValueError("pilot target tokens per source must be positive")
        for source in result["sources"]:
            source["target_tokens"] = target_tokens_per_source
    if shuffle_buffer is not None:
        if shuffle_buffer < 1:
            raise ValueError("shuffle buffer must be positive")
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
    benchmark_index_path: Path,
    work_dir: Path,
    publisher: Any,
) -> dict[str, Any]:
    """Stage one source. Only verified shards become durable progress."""

    name = str(source["name"])
    source_dir = work_dir / f"{source_index:02d}-{name}"
    source_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_json = (
        tokenizer_path / "tokenizer.json" if tokenizer_path.is_dir() else tokenizer_path
    )
    tokenizer_digest = sha256_file(tokenizer_json)
    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    with benchmark_index_path.open("rb") as stream:
        benchmark_index = pickle.load(stream)  # noqa: S301 - trusted local build artifact
    state = CandidateState(
        source_dir / "state.sqlite3",
        source=name,
        config_digest=config_sha256(config),
        tokenizer_digest=tokenizer_digest,
    )
    progress = state.progress(name)
    seen = state.seen()
    target_tokens = int(source["target_tokens"])
    oversample = float(config.get("candidate_oversample", 1.05))
    candidate_target = max(target_tokens, round(target_tokens * oversample))
    shard_target = int(config.get("candidate_shard_tokens", 250_000_000))
    tokenizer_batch = int(config.get("candidate_tokenization_batch_size", 512))
    protected = tuple(config.get("protected_benchmarks", DEFAULT_PROTECTED_BENCHMARKS))
    iterator = iter(iter_source(source, seed=int(config.get("seed", 1729)) + source_index))
    if progress["scanned"]:
        LOGGER.info("source=%s restore scanned=%s", name, f"{progress['scanned']:,}")
        _skip(iterator, progress["scanned"])
    partial = source_dir / "candidate.partial.jsonl.gz"
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
                                raise RuntimeError(
                                    f"source {name} exhausted at "
                                    f"{progress['retained_tokens'] + shard_tokens:,}/"
                                    f"{candidate_target:,} candidate tokens"
                                ) from error
                            shard_scanned += 1
                            shard_counters["scanned"] += 1
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
        published = publisher.publish_file(partial, relative)
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
    )
    publisher.publish_json(manifest, f"_candidates/{name}/manifest.json", source_dir)
    return manifest


def _stage_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    os.environ["RAYON_NUM_THREADS"] = str(payload["rayon_threads"])
    publisher = HubPublisher(str(payload["repo_id"]), "", os.environ.get("HF_TOKEN"))
    return stage_source(
        source_index=int(payload["source_index"]),
        source=payload["source"],
        config=payload["config"],
        tokenizer_path=Path(str(payload["tokenizer_path"])),
        benchmark_index_path=Path(str(payload["benchmark_index_path"])),
        work_dir=Path(str(payload["work_dir"])),
        publisher=publisher,
    )


def stage_all_sources(
    *,
    config: Mapping[str, Any],
    tokenizer_path: Path,
    work_dir: Path,
    repo_id: str,
    source_workers: int,
    rayon_threads_per_source: int,
) -> list[dict[str, Any]]:
    work_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = work_dir / "benchmark_index.pkl"
    if not benchmark_path.is_file():
        index = build_benchmark_index(tuple(config.get("protected_benchmark_sources", ())))
        temporary = benchmark_path.with_suffix(".partial")
        with temporary.open("wb") as stream:
            pickle.dump(index, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, benchmark_path)
    publisher = HubPublisher(repo_id, "", os.environ.get("HF_TOKEN"), create_private_repo=True)
    publisher.publish_json(config, "_candidates/config.json", work_dir)
    sources = list(config["sources"])
    payloads = [
        {
            "source_index": index,
            "source": source,
            "config": config,
            "tokenizer_path": str(tokenizer_path),
            "benchmark_index_path": str(benchmark_path),
            "work_dir": str(work_dir),
            "repo_id": repo_id,
            "rayon_threads": rayon_threads_per_source,
        }
        for index, source in enumerate(sources)
    ]
    results: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=source_workers) as executor:
        futures = {executor.submit(_stage_worker, payload): payload for payload in payloads}
        for future in as_completed(futures):
            result = future.result()
            results[str(result["source"])] = result
            LOGGER.info("candidate source complete: %s", result["source"])
    ordered = [results[str(source["name"])] for source in sources]
    publisher.publish_json(
        {"schema": SCHEMA, "sources": ordered, "complete": True},
        "_candidates/manifest.json",
        work_dir,
    )
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/agentic_pretraining/tr_hash_pretraining_125b.json"
    )
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--hf-repo", default="AETHORIA-AI/TR-HASH-Pretraining-125B-Agentic-32K")
    parser.add_argument("--source-workers", type=int, default=12)
    parser.add_argument("--rayon-threads-per-source", type=int, default=8)
    parser.add_argument("--only-source", action="append", default=[])
    parser.add_argument("--target-tokens-per-source", type=int)
    parser.add_argument("--shuffle-buffer", type=int)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = effective_stage_config(
        json.loads(Path(args.config).read_text(encoding="utf-8")),
        only_sources=args.only_source,
        target_tokens_per_source=args.target_tokens_per_source,
        shuffle_buffer=args.shuffle_buffer,
    )
    stage_all_sources(
        config=config,
        tokenizer_path=Path(args.tokenizer),
        work_dir=Path(args.work_dir),
        repo_id=args.hf_repo,
        source_workers=args.source_workers,
        rayon_threads_per_source=args.rayon_threads_per_source,
    )


if __name__ == "__main__":
    main()
