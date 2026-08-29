#!/usr/bin/env python3
"""Build a provenance-preserving general/agentic pretraining mixture.

The builder operates on raw text, before tokenizer training.  It keeps a
matched general-language core, selects documents rich in tool/API/code and
verification signals for the agentic slice, performs exact deduplication, and
rejects explicit benchmark references.  Every retained row keeps its source,
content hash and detected signal classes for auditability.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import io
import itertools
import json
import os
import re
import unicodedata
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

DEFAULT_PROTECTED_BENCHMARKS = (
    "arc_easy",
    "arc_challenge",
    "ai2_arc",
    "piqa",
    "gsm8k",
    "hellaswag",
    "mmlu",
    "truthfulqa",
    "winogrande",
)

SIGNAL_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "tool": (
        re.compile(r"\b(?:GET|POST|PUT|PATCH|DELETE)\s+/[A-Za-z0-9_./{}-]+"),
        re.compile(r"\b(?:curl|wget|git|pip|npm|docker|kubectl)\s+[-\w]"),
        re.compile(r'"(?:name|tool|function|arguments)"\s*:'),
        re.compile(r"\b(?:API|JSON-RPC|endpoint|tool call|function call)\b", re.I),
    ),
    "structured": (
        re.compile(r"\{\s*\"[^\"]+\"\s*:\s*"),
        re.compile(r"^\s*[A-Za-z_][\w.-]*:\s+\S+", re.M),
        re.compile(r"<\/?[A-Za-z][^>]*>"),
    ),
    "code": (
        re.compile(r"```(?:python|bash|javascript|typescript|json|yaml|sql)?", re.I),
        re.compile(r"^\s*(?:def|class|function|import|from)\s+[A-Za-z_]", re.M),
        re.compile(r"\b(?:try|except|raise|return|async|await)\b"),
    ),
    "procedure": (
        re.compile(r"^\s*(?:step\s+)?\d+[.)]\s+\S+", re.I | re.M),
        re.compile(r"\b(?:first|next|then|finally|before|after)\b", re.I),
        re.compile(r"\b(?:install|configure|create|run|execute|open|verify)\b", re.I),
    ),
    "verification": (
        re.compile(r"\b(?:assert|pytest|unittest|test case|expected output)\b", re.I),
        re.compile(r"\b(?:validate|verify|check|proof|counterexample)\b", re.I),
        re.compile(r"\b(?:error|exception|failed|debug|diagnos)\w*\b", re.I),
    ),
    "planning": (
        re.compile(r"\b(?:objective|goal|constraint|requirement|milestone)\b", re.I),
        re.compile(r"\b(?:plan|decompose|strategy|trade-?off|fallback)\b", re.I),
    ),
    "documentation": (
        re.compile(r"\b(?:parameters?|arguments?|returns?|raises?)\s*:", re.I),
        re.compile(r"\b(?:usage|example|reference|configuration)\b", re.I),
    ),
}


class BenchmarkContaminationIndex:
    """Compact word-ngram index for benchmark prompts embedded in documents."""

    def __init__(self, ngram_size: int = 12) -> None:
        self.ngram_size = ngram_size
        self._ngrams: dict[tuple[int, bytes], str] = {}
        self._widths: set[int] = set()
        self.prompt_count = 0

    @staticmethod
    def _words(text: str) -> tuple[str, ...]:
        return tuple(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))

    @staticmethod
    def _digest(words: Sequence[str]) -> bytes:
        return hashlib.blake2b("\x1f".join(words).encode("utf-8"), digest_size=8).digest()

    def add(self, benchmark: str, text: str) -> None:
        words = self._words(text)
        if len(words) < 5:
            return
        width = min(self.ngram_size, len(words))
        self._widths.add(width)
        for start in range(len(words) - width + 1):
            self._ngrams.setdefault((width, self._digest(words[start : start + width])), benchmark)
        self.prompt_count += 1

    def match(self, text: str) -> str | None:
        words = self._words(text)
        if len(words) < 5:
            return None
        for width in self._widths:
            if width > len(words):
                continue
            for start in range(len(words) - width + 1):
                benchmark = self._ngrams.get((width, self._digest(words[start : start + width])))
                if benchmark is not None:
                    return benchmark
        return None

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for (width, ngram), benchmark in sorted(self._ngrams.items()):
            digest.update(width.to_bytes(2, "big"))
            digest.update(ngram)
            digest.update(benchmark.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()


def _protected_rows(source: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    if "archive_url" in source:
        with urllib.request.urlopen(source["archive_url"], timeout=120) as response:
            payload = response.read()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != source["archive_sha256"]:
            raise ValueError(f"protected archive checksum mismatch for {source['name']}: {actual}")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for member in source["members"]:
                with archive.open(member) as stream:
                    for line in io.TextIOWrapper(stream, encoding="utf-8"):
                        if line.strip():
                            yield json.loads(line)
        return

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError("benchmark audit requires `pip install datasets`") from error
    for split in source.get("splits", ("train",)):
        dataset = load_dataset(
            source["dataset_id"],
            source.get("config_name"),
            split=split,
            revision=source["revision"],
            streaming=True,
        )
        yield from dataset


def build_benchmark_index(sources: Sequence[Mapping[str, Any]]) -> BenchmarkContaminationIndex:
    index = BenchmarkContaminationIndex()
    for source in sources:
        for row in _protected_rows(source):
            text = row.get(source.get("text_field", "text"), "")
            if isinstance(text, str):
                index.add(str(source["name"]), text)
    return index


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\x00", "")
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def content_sha256(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def quality_rejection(text: str, *, min_chars: int, max_chars: int) -> str | None:
    if len(text) < min_chars:
        return "too_short"
    if len(text) > max_chars:
        return "too_long"
    printable = sum(character.isprintable() or character in "\n\t" for character in text)
    if printable / len(text) < 0.95:
        return "non_printable"
    informative = sum(character.isalnum() for character in text)
    if informative / len(text) < 0.25:
        return "low_information"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 8 and len(set(lines)) / len(lines) < 0.35:
        return "repeated_lines"
    if re.search(r"(.)\1{39,}", text, re.S):
        return "repeated_character"
    return None


def benchmark_match(
    text: str,
    protected: Iterable[str],
    index: BenchmarkContaminationIndex | None = None,
) -> str | None:
    folded = text.casefold().replace("-", "_").replace(" ", "_")
    for name in protected:
        normalized = name.casefold().replace("-", "_").replace(" ", "_")
        if normalized in folded:
            return name
    return index.match(text) if index is not None else None


def agentic_signals(text: str) -> tuple[tuple[str, ...], int]:
    signals: list[str] = []
    score = 0
    for category, patterns in SIGNAL_PATTERNS.items():
        matches = sum(bool(pattern.search(text)) for pattern in patterns)
        if matches:
            signals.append(category)
            score += min(matches, 2)
    return tuple(signals), score


def is_agentic_candidate(
    text: str,
    *,
    min_score: int,
    min_signal_classes: int,
) -> tuple[bool, tuple[str, ...], int]:
    signals, score = agentic_signals(text)
    return score >= min_score and len(signals) >= min_signal_classes, signals, score


def _jsonl_rows(path: Path) -> Iterator[Mapping[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {error}") from error
            if isinstance(row, Mapping):
                yield row


def _software_heritage_stack_edu(
    source: Mapping[str, Any], *, seed: int
) -> Iterator[Mapping[str, Any]]:
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError("direct Stack-Edu resolution requires datasets and boto3") from error

    workers = int(source.get("download_workers", 32))
    client = boto3.client(
        "s3",
        config=Config(
            max_pool_connections=workers,
            retries={"max_attempts": 8, "mode": "adaptive"},
            signature_version=UNSIGNED,
        ),
    )

    def fetch(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
        blob_id = str(row["blob_id"])
        try:
            response = client.get_object(Bucket="softwareheritage", Key=f"content/{blob_id}")
            with gzip.GzipFile(fileobj=response["Body"]) as compressed:
                text = compressed.read().decode("utf-8", errors="ignore")
            return {"text": text, "_source_record_id": blob_id} if text else None
        except Exception:
            return None

    for language_index, language in enumerate(source.get("languages", ("Python",))):
        dataset = load_dataset(
            source["dataset_id"],
            language,
            split=source.get("split", "train"),
            revision=source["revision"],
            streaming=True,
        ).shuffle(seed=seed + language_index, buffer_size=int(source.get("shuffle_buffer", 10_000)))
        iterator = iter(dataset)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            while batch := list(itertools.islice(iterator, workers * 4)):
                resolved = list(executor.map(fetch, batch))
                successes = sum(row is not None for row in resolved)
                if successes < max(1, len(batch) // 4):
                    raise RuntimeError(
                        "Stack-Edu Software Heritage resolution failure rate exceeded 75%"
                    )
                yield from (row for row in resolved if row is not None)


def iter_source(source: Mapping[str, Any], *, seed: int) -> Iterator[Mapping[str, Any]]:
    if source.get("source_type") == "software_heritage_stack_edu":
        yield from _software_heritage_stack_edu(source, seed=seed)
        return
    if "path" in source or "path_env" in source:
        raw_path = source.get("path") or os.environ.get(str(source["path_env"]), "")
        if not raw_path:
            raise ValueError(f"source {source['name']!r} requires {source.get('path_env')}")
        paths = (
            [Path(path) for path in sorted(glob.glob(str(raw_path)))]
            if any(character in str(raw_path) for character in "*?[")
            else [Path(raw_path)]
        )
        if not paths or not all(path.is_file() for path in paths):
            raise FileNotFoundError(f"source {source['name']!r} path not found: {raw_path}")
        for path in paths:
            yield from _jsonl_rows(path)
        return

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError("Hugging Face sources require `pip install datasets`") from error
    dataset = load_dataset(
        source["dataset_id"],
        source.get("config_name"),
        split=source.get("split", "train"),
        revision=source.get("revision", "main"),
        streaming=True,
    )
    shuffle_buffer = int(source.get("shuffle_buffer", 10_000))
    if shuffle_buffer > 1:
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
    yield from dataset


def validate_config(config: Mapping[str, Any]) -> None:
    if int(config.get("version", 0)) != 1:
        raise ValueError("agentic corpus config version must be 1")
    target_bytes = int(config.get("target_bytes", 0))
    if target_bytes <= 0:
        raise ValueError("target_bytes must be positive")
    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must be a non-empty list")
    names = [source.get("name") for source in sources]
    if len(names) != len(set(names)):
        raise ValueError("source names must be unique")
    weights = sum(float(source.get("weight", 0.0)) for source in sources)
    if abs(weights - 1.0) > 1e-9:
        raise ValueError(f"source weights must sum to 1.0, got {weights}")
    for source in sources:
        if source.get("bucket") not in {"general", "agentic"}:
            raise ValueError(f"source {source.get('name')!r} has invalid bucket")
        if not source.get("license_audit"):
            raise ValueError(f"source {source.get('name')!r} must declare license_audit")


def build_corpus(config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    validate_config(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    protected = tuple(config.get("protected_benchmarks", DEFAULT_PROTECTED_BENCHMARKS))
    min_chars = int(config.get("min_chars", 200))
    max_chars = int(config.get("max_chars", 100_000))
    min_score = int(config.get("agentic_min_score", 4))
    min_classes = int(config.get("agentic_min_signal_classes", 2))
    max_scan = int(config.get("max_scan_records_per_source", 1_000_000))
    seed = int(config.get("seed", 1729))
    target_bytes = int(config["target_bytes"])
    benchmark_sources = tuple(config.get("protected_benchmark_sources", ()))
    benchmark_index = build_benchmark_index(benchmark_sources)
    seen_hashes: set[str] = set()
    manifest_sources: list[dict[str, Any]] = []

    for source_index, source in enumerate(config["sources"]):
        source_target = round(target_bytes * float(source["weight"]))
        counters: Counter[str] = Counter()
        signal_counts: Counter[str] = Counter()
        retained_bytes = 0
        retained_records = 0
        output = output_dir / f"{source_index:02d}-{source['name']}.jsonl"
        with output.open("w", encoding="utf-8") as destination:
            for scanned, row in enumerate(iter_source(source, seed=seed + source_index), start=1):
                if scanned > max_scan or retained_bytes >= source_target:
                    break
                counters["scanned"] += 1
                text = row.get(source.get("text_field", "text"), "")
                if not isinstance(text, str):
                    counters["missing_text"] += 1
                    continue
                text = normalize_text(text)
                rejected = quality_rejection(text, min_chars=min_chars, max_chars=max_chars)
                if rejected:
                    counters[rejected] += 1
                    continue
                contaminated = benchmark_match(text, protected, benchmark_index)
                if contaminated:
                    counters[f"benchmark:{contaminated}"] += 1
                    continue
                digest = content_sha256(text)
                if digest in seen_hashes:
                    counters["exact_duplicate"] += 1
                    continue
                candidate, signals, score = is_agentic_candidate(
                    text,
                    min_score=min_score,
                    min_signal_classes=min_classes,
                )
                if source["bucket"] == "agentic" and not candidate:
                    counters["weak_agentic_signal"] += 1
                    continue
                encoded_bytes = len(text.encode("utf-8"))
                if retained_bytes and retained_bytes + encoded_bytes > source_target:
                    counters["over_budget"] += 1
                    continue
                seen_hashes.add(digest)
                for signal in signals:
                    signal_counts[signal] += 1
                destination.write(
                    json.dumps(
                        {
                            "text": text,
                            "source": source["name"],
                            "bucket": source["bucket"],
                            "agentic_score": score,
                            "agentic_signals": signals,
                            "content_sha256": digest,
                            "source_record_id": next(
                                (
                                    row[field]
                                    for field in source.get(
                                        "record_id_fields",
                                        ("_source_record_id", "id", "url", "blob_id"),
                                    )
                                    if row.get(field) is not None
                                ),
                                None,
                            ),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                retained_bytes += encoded_bytes
                retained_records += 1
                counters["retained"] += 1

        manifest_sources.append(
            {
                "name": source["name"],
                "bucket": source["bucket"],
                "weight": source["weight"],
                "license_audit": source["license_audit"],
                "input": {
                    key: source[key]
                    for key in (
                        "source_type",
                        "dataset_id",
                        "config_name",
                        "split",
                        "revision",
                        "path",
                        "path_env",
                    )
                    if key in source
                },
                "target_bytes": source_target,
                "retained_bytes": retained_bytes,
                "retained_records": retained_records,
                "signal_counts": dict(sorted(signal_counts.items())),
                "counters": dict(sorted(counters.items())),
                "output": output.name,
                "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            }
        )

    retained_total = sum(source["retained_bytes"] for source in manifest_sources)
    if retained_total < target_bytes * 0.95:
        raise RuntimeError(
            f"corpus reached only {retained_total:,}/{target_bytes:,} bytes; "
            "increase max_scan_records_per_source or repair source availability"
        )
    manifest = {
        "schema": "tr-hash-agentic-pretraining-corpus-v1",
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "target_bytes": target_bytes,
        "retained_bytes": retained_total,
        "seed": seed,
        "protected_benchmarks": protected,
        "protected_benchmark_sources": benchmark_sources,
        "protected_prompt_count": benchmark_index.prompt_count,
        "protected_index_sha256": benchmark_index.fingerprint(),
        "exact_deduplication": True,
        "sources": manifest_sources,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    manifest = build_corpus(config, Path(args.output_dir))
    print(f"Agentic corpus: {manifest['retained_bytes']:,} bytes")
    print(f"Manifest: {Path(args.output_dir) / 'manifest.json'}")


if __name__ == "__main__":
    main()
