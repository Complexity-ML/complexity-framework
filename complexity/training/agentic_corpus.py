"""Reusable raw agentic corpus filtering and provenance.

This is a data operation with caller-owned budgets and pinned sources. Imports
never download data or initialize a model, and no training/upload is implicit.
Signal heuristics and exact/word-ngram decontamination are auditable filters,
not a claim of agent capability or complete benchmark decontamination.
"""

from __future__ import annotations

import glob
import gzip
import hashlib
import json
import math
import os
import re
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
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
        if type(ngram_size) is not int or ngram_size < 1 or ngram_size > 65535:
            raise ValueError("ngram_size must be between 1 and 65535")
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


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\x00", "")
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def content_sha256(text: str) -> str:
    normalized = re.sub(r"\s+", " ", normalize_text(text)).strip().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def quality_rejection(text: str, *, min_chars: int, max_chars: int) -> str | None:
    if min_chars < 1 or max_chars < min_chars:
        raise ValueError("quality character bounds must be positive and ordered")
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


def _stack_download_workers(source: Mapping[str, Any]) -> int:
    """Resolve a runtime-safe Stack-Edu worker count without changing corpus state."""

    configured = int(source.get("download_workers", 32))
    language = re.sub(r"[^A-Za-z0-9]+", "_", str(source.get("config_name", ""))).upper()
    language_override = (
        os.environ.get(f"TR_HASH_STACK_DOWNLOAD_WORKERS_{language}") if language else None
    )
    workers = int(
        language_override
        if language_override is not None
        else os.environ.get("TR_HASH_STACK_DOWNLOAD_WORKERS", configured)
    )
    if workers < 1:
        raise ValueError("Stack-Edu download worker count must be positive")
    return workers


def _positive(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def config_sha256(config: Mapping[str, Any]) -> str:
    data = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def validate_sources(sources: Any) -> None:
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)) or not sources:
        raise ValueError("sources must be a non-empty list")
    names, weights = [], []
    for source in sources:
        if not isinstance(source, Mapping):
            raise ValueError("each source must be a mapping")
        name = source.get("name")
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_-]+", name) is None:
            raise ValueError("source names must be nonempty contained path components")
        names.append(name)
        weight = source.get("weight")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (float, int))
            or not math.isfinite(weight)
            or weight <= 0
        ):
            raise ValueError(f"source weight must be finite and positive: {name}")
        weights.append(weight)
        if not isinstance(source.get("bucket"), str) or not source["bucket"]:
            raise ValueError(f"source must declare a bucket: {name}")
        if not isinstance(source.get("license_audit"), str) or not source["license_audit"].strip():
            raise ValueError(f"source {name!r} must declare license_audit")
        if (
            "dataset_id" in source
            and re.fullmatch(r"[0-9a-f]{40}", str(source.get("revision", ""))) is None
        ):
            raise ValueError(f"source requires an immutable revision: {name}")
        if "allowed_license_types" in source:
            values = source["allowed_license_types"]
            if (
                not isinstance(values, list)
                or not values
                or not all(isinstance(value, str) and value for value in values)
            ):
                raise ValueError(f"allowed_license_types must be explicit nonempty strings: {name}")
    if len(names) != len(set(names)):
        raise ValueError("source names must be unique")
    if not math.isclose(sum(weights), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("source weights must sum to 1.0")


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("version") != 1 or isinstance(config.get("version"), bool):
        raise ValueError("agentic corpus config version must be 1")
    targets = [key for key in ("target_bytes", "target_tokens") if config.get(key, 0)]
    if len(targets) != 1:
        raise ValueError("exactly one of target_bytes/target_tokens is required")
    _positive(config[targets[0]], targets[0])
    validate_sources(config.get("sources"))
    for source in config["sources"]:
        if source["bucket"] not in {"general", "agentic"}:
            raise ValueError(f"invalid raw corpus bucket: {source['bucket']}")
    min_chars = _positive(config.get("min_chars", 200), "min_chars")
    if _positive(config.get("max_chars", 100_000), "max_chars") < min_chars:
        raise ValueError("max_chars must be at least min_chars")


def _rows(stream: Iterable[str], description: str) -> Iterator[Mapping[str, Any]]:
    for number, line in enumerate(stream, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at {description}:{number}") from error
        if not isinstance(row, Mapping):
            raise ValueError(f"expected JSON object at {description}:{number}")
        yield row


def local_source_paths(source: Mapping[str, Any]) -> list[Path]:
    raw = source.get("path") or os.environ.get(str(source.get("path_env", "")), "")
    if not raw:
        raise ValueError(f"source {source.get('name')!r} requires a local path")
    paths = [Path(path) for path in sorted(glob.glob(str(raw)))]
    if not paths or not all(path.is_file() for path in paths):
        raise FileNotFoundError(raw)
    return paths


def iter_source(source: Mapping[str, Any], *, seed: int) -> Iterator[Mapping[str, Any]]:
    """Read local JSONL or an explicitly pinned optional dataset adapter.

    Raw Hub JSONL avoids Arrow schema coercion of heterogeneous tool metadata.
    Stack/SWH acquisition is a separate materialization operation.
    """
    if "path" in source or "path_env" in source:
        for path in local_source_paths(source):
            declared = source.get("sha256")
            if declared and (len(local_source_paths(source)) != 1 or sha256_file(path) != declared):
                raise ValueError(f"local source checksum mismatch: {path}")
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt", encoding="utf-8") as stream:
                yield from _rows(stream, str(path))
        return
    if source.get("source_type") in {"software_heritage_stack_edu", "inline_stack_edu"}:
        raise ValueError("materialize Stack/SWH content locally before corpus preparation")
    dataset, revision = source.get("dataset_id"), source.get("revision")
    if not dataset or re.fullmatch(r"[0-9a-f]{40}", str(revision)) is None:
        raise ValueError("remote sources require a dataset ID and immutable revision")
    if source.get("source_type") == "hf_raw_jsonl":
        from huggingface_hub import HfFileSystem

        files = source.get("repo_files")
        if not isinstance(files, list) or not files:
            raise ValueError("hf_raw_jsonl requires a nonempty repo_files list")
        filesystem = HfFileSystem()
        for filename in files:
            relative = Path(filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("repo_files must be contained relative paths")
            path = f"datasets/{dataset}@{revision}/{filename}"
            with filesystem.open(path, "rt") as stream:
                yield from _rows(stream, path)
        return
    from datasets import load_dataset

    dataset = load_dataset(
        dataset,
        source.get("config_name"),
        split=source.get("split", "train"),
        revision=revision,
        streaming=True,
    )
    buffer = _positive(source.get("shuffle_buffer", 10_000), "shuffle_buffer")
    yield from dataset.shuffle(seed=seed, buffer_size=buffer) if buffer > 1 else dataset


def build_benchmark_index(sources: Sequence[Mapping[str, Any]]) -> BenchmarkContaminationIndex:
    index = BenchmarkContaminationIndex()
    for source in sources:
        if "archive_url" in source:
            raise ValueError(
                "protected benchmark archives must be materialized into pinned local JSONL before filtering"
            )
        for row in iter_source(source, seed=0):
            parts = []
            for field in source.get("text_fields", [source.get("text_field", "text")]):
                value = row.get(field)
                if isinstance(value, str):
                    parts.append(value)
                elif value:
                    parts.append(json.dumps(value, sort_keys=True, ensure_ascii=False))
            if parts:
                index.add(str(source["name"]), "\n".join(parts))
    return index


def source_row_rejection(row: Mapping[str, Any], source: Mapping[str, Any]) -> str | None:
    """Apply explicit upstream score/license guards before content selection."""
    if "allowed_license_types" in source:
        if (
            row.get(source.get("license_type_field", "license_type"))
            not in source["allowed_license_types"]
        ):
            return "license_type"
    if "min_int_score" in source:
        score = row.get(source.get("score_field", "int_score"))
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or score < source["min_int_score"]
        ):
            return "upstream_score"
    return None


def build_corpus(
    config: Mapping[str, Any], output_dir: Path, *, reference_tokenizer: Path | None = None
) -> dict[str, Any]:
    """Create a new filtered raw corpus with source and output integrity records."""
    validate_config(config)
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    targets_tokens = "target_tokens" in config and config["target_tokens"] > 0
    total_target = config["target_tokens" if targets_tokens else "target_bytes"]
    tokenizer, tokenizer_sha = None, None
    if targets_tokens:
        if reference_tokenizer is None:
            raise ValueError("target_tokens requires a reference_tokenizer")
        from tokenizers import Tokenizer

        path = Path(reference_tokenizer)
        path = path / "tokenizer.json" if path.is_dir() else path
        tokenizer, tokenizer_sha = Tokenizer.from_file(str(path)), sha256_file(path)
    protected = tuple(config.get("protected_benchmarks", DEFAULT_PROTECTED_BENCHMARKS))
    benchmark_sources = config.get("protected_benchmark_sources", [])
    index = build_benchmark_index(benchmark_sources)
    weights = [source["weight"] * total_target for source in config["sources"]]
    quotas = [math.floor(value) for value in weights]
    for i in sorted(range(len(quotas)), key=lambda i: weights[i] - quotas[i], reverse=True)[
        : total_target - sum(quotas)
    ]:
        quotas[i] += 1
    seen, entries = set(), []
    max_scan = _positive(
        config.get("max_scan_records_per_source", 1_000_000), "max_scan_records_per_source"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".complexity-corpus-", dir=output.parent) as directory:
        pending = Path(directory)
        for source_index, (source, quota) in enumerate(zip(config["sources"], quotas, strict=True)):
            path = pending / f"{source_index:02d}-{source['name']}.jsonl"
            counts, signals_count = Counter(), Counter()
            retained_bytes, retained_tokens, retained = 0, 0, 0
            input_hashes = (
                {str(path.resolve()): sha256_file(path) for path in local_source_paths(source)}
                if "path" in source or "path_env" in source
                else {}
            )
            with path.open("w", encoding="utf-8") as handle:
                for scanned, row in enumerate(
                    iter_source(source, seed=config.get("seed", 1729) + source_index), 1
                ):
                    units = retained_tokens if targets_tokens else retained_bytes
                    if units >= quota or scanned > max_scan:
                        break
                    counts["scanned"] += 1
                    raw = row.get(source.get("text_field", "text"))
                    rejected = source_row_rejection(row, source)
                    if not isinstance(raw, str):
                        rejected = rejected or "missing_text"
                    text = normalize_text(raw) if isinstance(raw, str) else ""
                    rejected = rejected or quality_rejection(
                        text,
                        min_chars=config.get("min_chars", 200),
                        max_chars=config.get("max_chars", 100_000),
                    )
                    benchmark = benchmark_match(text, protected, index)
                    rejected = rejected or (f"benchmark:{benchmark}" if benchmark else None)
                    digest = content_sha256(text)
                    if digest in seen:
                        rejected = rejected or "exact_duplicate"
                    candidate, signals, score = is_agentic_candidate(
                        text,
                        min_score=config.get("agentic_min_score", 4),
                        min_signal_classes=config.get("agentic_min_signal_classes", 2),
                    )
                    if source["bucket"] == "agentic" and not candidate:
                        rejected = rejected or "weak_agentic_signal"
                    size = len(text.encode("utf-8"))
                    tokens = (
                        len(tokenizer.encode(text, add_special_tokens=False).ids)
                        if tokenizer
                        else 0
                    )
                    if units + (tokens if targets_tokens else size) > quota:
                        rejected = rejected or "over_budget"
                    if rejected:
                        counts[rejected] += 1
                        continue
                    source_id = next(
                        (
                            row.get(key)
                            for key in source.get(
                                "record_id_fields", ("_source_record_id", "id", "url", "blob_id")
                            )
                            if row.get(key) is not None
                        ),
                        None,
                    )
                    handle.write(
                        json.dumps(
                            {
                                "text": text,
                                "source": source["name"],
                                "bucket": source["bucket"],
                                "agentic_score": score,
                                "agentic_signals": signals,
                                "content_sha256": digest,
                                "source_record_id": source_id,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    seen.add(digest)
                    retained_bytes += size
                    retained_tokens += tokens
                    retained += 1
                    counts["retained"] += 1
                    signals_count.update(signals)
            if any(sha256_file(Path(path)) != digest for path, digest in input_hashes.items()):
                raise ValueError(f"input source changed during preparation: {source['name']}")
            entries.append(
                {
                    "name": source["name"],
                    "bucket": source["bucket"],
                    "weight": source["weight"],
                    "license_audit": source["license_audit"],
                    "input": {
                        key: source[key]
                        for key in (
                            "dataset_id",
                            "revision",
                            "config_name",
                            "split",
                            "path",
                            "source_type",
                        )
                        if key in source
                    },
                    "input_sha256": input_hashes,
                    "retained_bytes": retained_bytes,
                    "retained_tokens": retained_tokens if targets_tokens else None,
                    "retained_records": retained,
                    "target_tokens" if targets_tokens else "target_bytes": quota,
                    "counters": dict(counts),
                    "signal_counts": dict(signals_count),
                    "output": path.name,
                    "output_sha256": sha256_file(path),
                }
            )
        retained_total = sum(
            entry["retained_tokens"] if targets_tokens else entry["retained_bytes"]
            for entry in entries
        )
        minimum = config.get("minimum_retained_fraction", 0.95)
        if (
            isinstance(minimum, bool)
            or not isinstance(minimum, (float, int))
            or not 0 < minimum <= 1
        ):
            raise ValueError("minimum_retained_fraction must be in (0,1]")
        if retained_total < total_target * minimum:
            raise RuntimeError(f"corpus reached only {retained_total}/{total_target} units")
        manifest = {
            "schema": "tr-hash-agentic-pretraining-corpus-v1",
            "config_sha256": config_sha256(config),
            "target_tokens" if targets_tokens else "target_bytes": total_target,
            "retained_bytes": sum(entry["retained_bytes"] for entry in entries),
            "retained_tokens": retained_total if targets_tokens else None,
            "reference_tokenizer_sha256": tokenizer_sha,
            "protected_benchmarks": protected,
            "protected_prompt_count": index.prompt_count,
            "protected_index_sha256": index.fingerprint(),
            "exact_deduplication": True,
            "sources": entries,
        }
        (pending / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        output.mkdir()
        try:
            for child in pending.iterdir():
                os.replace(child, output / child.name)
        except BaseException:
            import shutil

            shutil.rmtree(output)
            raise
    return manifest
