"""Lossless, quota-controlled streaming mixtures for text pretraining."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ..parallel.data_parallel import is_main_process

logger = logging.getLogger(__name__)
HF_DATASET_PREFIX = "hf://datasets/"
TOKEN_DTYPE = np.dtype("<u2")


def _collect_transient_http_errors() -> tuple[type[Exception], ...]:
    """huggingface_hub has shipped on both httpx and requests across
    versions; catch whichever transport-error base class is installed."""
    errors: list[type[Exception]] = []
    try:
        import httpx

        errors.append(httpx.TransportError)
    except ImportError:
        pass
    try:
        import requests

        errors.append(requests.exceptions.RequestException)
    except ImportError:
        pass
    return tuple(errors)


_TRANSIENT_HTTP_ERRORS = _collect_transient_http_errors()
REPLAY_PLAN_FORMAT = "tr-hash-token-replay-plan-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_hf_dataset_uri(uri: str) -> str:
    if not uri.startswith(HF_DATASET_PREFIX):
        raise ValueError(
            f"remote token data must use {HF_DATASET_PREFIX}<owner>/<repo>"
        )
    repo_id = uri[len(HF_DATASET_PREFIX) :].strip("/")
    if repo_id.count("/") != 1 or any(part in {"", ".", ".."} for part in repo_id.split("/")):
        raise ValueError(f"invalid Hugging Face dataset URI: {uri!r}")
    return repo_id


def _safe_relative_path(filename: str) -> Path:
    path = Path(filename)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe token dataset path: {filename!r}")
    return path


class _HubShardCache:
    """Bounded, process-safe local cache for one Hub token dataset."""

    def __init__(
        self,
        *,
        repo_id: str,
        cache_dir: str | Path,
        revision: str,
        token: str | None,
        max_cache_bytes: int,
        prefetch_shards: int,
        downloader: Callable[..., str] | None = None,
        file_lister: Callable[..., Sequence[str]] | None = None,
    ) -> None:
        if max_cache_bytes < 1:
            raise ValueError("remote token cache size must be positive")
        if prefetch_shards < 0:
            raise ValueError("prefetch_shards cannot be negative")
        safe_repo = re.sub(r"[^A-Za-z0-9._-]+", "--", repo_id)
        safe_revision = re.sub(r"[^A-Za-z0-9._-]+", "--", revision)
        self.root = Path(cache_dir) / safe_repo / safe_revision
        self.root.mkdir(parents=True, exist_ok=True)
        self.repo_id = repo_id
        self.revision = revision
        self.token = token
        self.max_cache_bytes = int(max_cache_bytes)
        self.prefetch_shards = int(prefetch_shards)
        self._downloader = downloader
        self._file_lister = file_lister
        self._executor: ThreadPoolExecutor | None = None
        self._prefetches: dict[str, Future[Path]] = {}

    @staticmethod
    def _lock(path: Path):
        try:
            from filelock import FileLock
        except ImportError as exc:
            raise RuntimeError(
                "remote token streaming requires filelock; install the framework dependencies"
            ) from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        return FileLock(str(path))

    @staticmethod
    @contextmanager
    def _pin_lock(
        path: Path,
        *,
        shared: bool,
        blocking: bool = True,
    ) -> Iterator[None]:
        """Hold a reader/writer pin for a cached shard.

        Training ranks must be able to mmap the same shard concurrently, while
        eviction must still have exclusive ownership before unlinking it.
        ``filelock.FileLock`` is exclusive-only and deadlocks DDP when one rank
        enters the forward pass while its peers wait to pin that same shard.
        POSIX ``flock`` gives us the required shared-reader semantics.  The
        exclusive FileLock fallback keeps non-POSIX platforms safe, albeit
        without concurrent readers.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        if os.name == "posix":
            import fcntl

            with path.open("a+b") as handle:
                operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
                if not blocking:
                    operation |= fcntl.LOCK_NB
                fcntl.flock(handle.fileno(), operation)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return

        from filelock import Timeout

        lock = _HubShardCache._lock(path)
        try:
            lock.acquire(timeout=-1 if blocking else 0)
        except Timeout as exc:
            raise BlockingIOError from exc
        try:
            yield
        finally:
            lock.release()

    def _download(self, filename: str) -> Path:
        filename = _safe_relative_path(filename).as_posix()
        downloader = self._downloader
        if downloader is None:
            from huggingface_hub import hf_hub_download

            downloader = hf_hub_download
        downloaded = Path(
            downloader(
                repo_id=self.repo_id,
                filename=filename,
                repo_type="dataset",
                revision=self.revision,
                token=self.token,
                local_dir=self.root,
            )
        )
        if not downloaded.is_file():
            raise FileNotFoundError(f"Hub download did not produce {filename}: {downloaded}")
        return downloaded

    @staticmethod
    def _verification_marker(path: Path) -> Path:
        return path.with_name(f"{path.name}.verified.json")

    def _validate(
        self,
        path: Path,
        *,
        expected_bytes: int | None,
        expected_sha256: str | None,
    ) -> None:
        if expected_bytes is not None and path.stat().st_size != expected_bytes:
            raise ValueError(
                f"corrupt cached shard {path}: {path.stat().st_size} != {expected_bytes} bytes"
            )
        if expected_sha256 is None:
            return
        marker = self._verification_marker(path)
        if marker.is_file():
            try:
                recorded = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                recorded = {}
            stat = path.stat()
            if (
                recorded.get("sha256") == expected_sha256
                and int(recorded.get("bytes", -1)) == stat.st_size
                and int(recorded.get("mtime_ns", -1)) == stat.st_mtime_ns
            ):
                return
        actual = _sha256_file(path)
        if actual != expected_sha256:
            raise ValueError(
                f"SHA-256 mismatch for cached shard {path}: {actual} != {expected_sha256}"
            )
        stat = path.stat()
        temporary = marker.with_suffix(f"{marker.suffix}.partial-{os.getpid()}")
        temporary.write_text(
            json.dumps(
                {
                    "sha256": actual,
                    "bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, marker)

    def _evict(self, *, exclude: set[Path]) -> None:
        from filelock import Timeout

        lock = self._lock(self.root / ".locks" / "eviction.lock")
        with lock:
            shards = [path for path in self.root.rglob("*.bin") if path.is_file()]
            total = sum(path.stat().st_size for path in shards)
            if total <= self.max_cache_bytes:
                return
            for candidate in sorted(shards, key=lambda path: path.stat().st_atime_ns):
                if candidate in exclude:
                    continue
                pin_path = (
                    self.root
                    / ".locks"
                    / "pins"
                    / f"{hashlib.sha256(str(candidate).encode()).hexdigest()}.lock"
                )
                try:
                    with self._pin_lock(pin_path, shared=False, blocking=False):
                        size = candidate.stat().st_size if candidate.exists() else 0
                        candidate.unlink(missing_ok=True)
                        self._verification_marker(candidate).unlink(missing_ok=True)
                        total -= size
                except (BlockingIOError, Timeout):
                    continue
                if total <= self.max_cache_bytes:
                    break

    def get(
        self,
        filename: str,
        *,
        expected_bytes: int | None = None,
        expected_sha256: str | None = None,
    ) -> Path:
        filename = _safe_relative_path(filename).as_posix()
        destination = self.root / filename
        lock_name = hashlib.sha256(filename.encode()).hexdigest()
        lock = self._lock(self.root / ".locks" / "downloads" / f"{lock_name}.lock")
        with lock:
            if not destination.is_file():
                destination = self._download(filename)
            try:
                self._validate(
                    destination,
                    expected_bytes=expected_bytes,
                    expected_sha256=expected_sha256,
                )
            except ValueError:
                destination.unlink(missing_ok=True)
                self._verification_marker(destination).unlink(missing_ok=True)
                destination = self._download(filename)
                self._validate(
                    destination,
                    expected_bytes=expected_bytes,
                    expected_sha256=expected_sha256,
                )
            os.utime(destination, None)
        if destination.suffix == ".bin":
            self._evict(exclude={destination})
        return destination

    def list_files(self, *, max_attempts: int = 5) -> set[str]:
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            if attempt:
                time.sleep(min(2**attempt, 30))
            try:
                if self._file_lister is not None:
                    files = self._file_lister(
                        repo_id=self.repo_id,
                        repo_type="dataset",
                        revision=self.revision,
                        token=self.token,
                    )
                else:
                    from huggingface_hub import HfApi

                    files = HfApi(token=self.token).list_repo_files(
                        repo_id=self.repo_id,
                        repo_type="dataset",
                        revision=self.revision,
                    )
                return {str(filename) for filename in files}
            except (OSError, *_TRANSIENT_HTTP_ERRORS) as error:
                # OSError covers raw socket/DNS errors; the rest cover
                # httpx's and requests' own wrapper exceptions for the same
                # (huggingface_hub has used both across versions).
                # Reproduced live: a fresh N-rank run bursts N simultaneous
                # DNS lookups for the same host and some get dropped, which
                # crashed the whole distributed job — and, with
                # autorestart, looped — over a fully transient failure that
                # a short retry clears.
                last_error = error
                logger.warning(
                    "list_repo_files failed (attempt %d/%d): %s",
                    attempt + 1,
                    max_attempts,
                    error,
                )
        assert last_error is not None
        raise last_error

    @contextmanager
    def pinned(
        self,
        filename: str,
        *,
        expected_bytes: int,
        expected_sha256: str | None,
    ) -> Iterator[Path]:
        filename = _safe_relative_path(filename).as_posix()
        destination = self.root / filename
        pin_name = hashlib.sha256(str(destination).encode()).hexdigest()
        pin_path = self.root / ".locks" / "pins" / f"{pin_name}.lock"
        with self._pin_lock(pin_path, shared=True):
            yield self.get(
                filename,
                expected_bytes=expected_bytes,
                expected_sha256=expected_sha256,
            )

    def prefetch(
        self,
        filename: str,
        *,
        expected_bytes: int,
        expected_sha256: str | None,
    ) -> None:
        filename = _safe_relative_path(filename).as_posix()
        if self.prefetch_shards == 0 or filename in self._prefetches:
            return
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.prefetch_shards,
                thread_name_prefix="tr-hash-token-prefetch",
            )
        self._prefetches[filename] = self._executor.submit(
            self.get,
            filename,
            expected_bytes=expected_bytes,
            expected_sha256=expected_sha256,
        )


@dataclass(frozen=True)
class TextCorpusSource:
    """One streaming text source in a pretraining mixture."""

    name: str
    weight: float
    dataset_id: str | None = None
    config_name: str | None = None
    data_files: str | Sequence[str] | None = None
    text_field: str = "text"
    split: str = "train"

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("corpus source name cannot be empty")
        if not 0.0 < self.weight <= 1.0:
            raise ValueError(f"{self.name} weight must be in (0, 1]")
        if (self.dataset_id is None) == (self.data_files is None):
            raise ValueError(
                f"{self.name} must define exactly one of dataset_id or data_files"
            )
        if not self.text_field.strip():
            raise ValueError(f"{self.name} text_field cannot be empty")


def validate_corpus_mixture(sources: Sequence[TextCorpusSource]) -> None:
    """Validate a complete mixture before any remote dataset is opened."""

    if not sources:
        raise ValueError("corpus mixture cannot be empty")
    for source in sources:
        source.validate()
    names = [source.name for source in sources]
    if len(names) != len(set(names)):
        raise ValueError("corpus source names must be unique")
    total = sum(source.weight for source in sources)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"corpus weights must sum to 1.0, got {total:.12f}")


def allocate_weighted_counts(
    total: int, sources: Sequence[TextCorpusSource]
) -> dict[str, int]:
    """Allocate an integer sequence budget without changing mixture weights."""

    validate_corpus_mixture(sources)
    if total < 1:
        raise ValueError("weighted allocation total must be positive")
    exact = {source.name: total * source.weight for source in sources}
    counts = {name: math.floor(value) for name, value in exact.items()}
    remainder = total - sum(counts.values())
    order = sorted(
        sources,
        key=lambda source: (-(exact[source.name] - counts[source.name]), source.name),
    )
    for source in order[:remainder]:
        counts[source.name] += 1
    if sum(counts.values()) != total:
        raise AssertionError("weighted allocation does not conserve the total")
    return counts


class WeightedStreamingTextDataset(IterableDataset):
    """Mix independently packed token streams at exact chunk-level weights.

    Every source keeps its own token buffer. Documents are concatenated
    losslessly *within* a corpus, never across corpora. Since every emitted
    sample contains ``seq_len`` training tokens, weighted sample quotas are
    also weighted token quotas.
    """

    packing_contract = "text-pretraining"

    def __init__(
        self,
        *,
        tokenizer: Any,
        seq_len: int,
        sources: Sequence[TextCorpusSource],
        rank: int = 0,
        world_size: int = 1,
        streams: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
    ) -> None:
        validate_corpus_mixture(sources)
        if seq_len < 1:
            raise ValueError("seq_len must be positive")
        if rank < 0 or world_size < 1 or rank >= world_size:
            raise ValueError("invalid distributed rank/world_size")
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.sources = tuple(sources)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self._provided_streams = streams

    @staticmethod
    def _next_source(
        sources: Sequence[TextCorpusSource], counts: Mapping[str, int]
    ) -> TextCorpusSource:
        # Weighted fair queuing: select the stream with the smallest realized
        # share relative to its target. Stable source order breaks ties.
        return min(sources, key=lambda source: counts[source.name] / source.weight)

    def _load_stream(self, source: TextCorpusSource) -> Iterable[Mapping[str, Any]]:
        if self._provided_streams is not None:
            try:
                return self._provided_streams[source.name]
            except KeyError as exc:
                raise ValueError(f"missing provided stream for {source.name}") from exc

        from datasets import load_dataset

        if source.data_files is not None:
            dataset = load_dataset(
                "json",
                data_files=source.data_files,
                split=source.split,
                streaming=True,
            )
        else:
            dataset = load_dataset(
                source.dataset_id,
                source.config_name,
                split=source.split,
                streaming=True,
            )

        worker = get_worker_info()
        worker_count = worker.num_workers if worker is not None else 1
        worker_id = worker.id if worker is not None else 0
        shard_count = self.world_size * worker_count
        shard_index = self.rank * worker_count + worker_id
        if shard_count > 1:
            dataset = dataset.shard(num_shards=shard_count, index=shard_index)
        return dataset

    def _packed_chunks(
        self,
        source: TextCorpusSource,
        stream: Iterable[Mapping[str, Any]],
    ) -> Iterator[dict[str, torch.Tensor]]:
        buffer: list[int] = []
        eos_token_id = self.tokenizer.eos_token_id
        for example in stream:
            text = example.get(source.text_field, "")
            if not isinstance(text, str) or not text:
                continue
            buffer.extend(self.tokenizer.encode(text, add_special_tokens=False))
            if eos_token_id is not None:
                buffer.append(eos_token_id)
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                del buffer[: self.seq_len]
                yield {
                    "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                    "labels": torch.tensor(chunk[1:], dtype=torch.long),
                }

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        iterators = {
            source.name: iter(self._packed_chunks(source, self._load_stream(source)))
            for source in self.sources
        }
        counts = {source.name: 0 for source in self.sources}
        while True:
            source = self._next_source(self.sources, counts)
            try:
                sample = next(iterators[source.name])
            except StopIteration as exc:
                raise RuntimeError(
                    f"corpus source {source.name!r} exhausted before the token budget"
                ) from exc
            counts[source.name] += 1
            yield sample


class PretokenizedCorpusMixtureDataset(IterableDataset):
    """Read local or lazily cached Hub uint16 token streams."""

    packing_contract = "text-pretraining"

    def __init__(
        self,
        root: str | Path,
        *,
        rank: int = 0,
        world_size: int = 1,
        cache_dir: str | Path | None = None,
        cache_max_bytes: int = 32 * 1024**3,
        revision: str = "main",
        token: str | None = None,
        prefetch_shards: int = 1,
        hub_downloader: Callable[..., str] | None = None,
        hub_file_lister: Callable[..., Sequence[str]] | None = None,
        replay_plan: str | Path | Mapping[str, Any] | None = None,
    ):
        root_string = str(root)
        self._hub_cache: _HubShardCache | None = None
        if root_string.startswith("hf://"):
            repo_id = _parse_hf_dataset_uri(root_string)
            if cache_dir is None:
                cache_dir = os.environ.get(
                    "TR_HASH_TOKEN_CACHE", "artifacts/tr_hash_token_cache"
                )
            self._hub_cache = _HubShardCache(
                repo_id=repo_id,
                cache_dir=cache_dir,
                revision=revision,
                token=token,
                max_cache_bytes=cache_max_bytes,
                prefetch_shards=prefetch_shards,
                downloader=hub_downloader,
                file_lister=hub_file_lister,
            )
            self.root = self._hub_cache.root
            manifest_path = self._hub_cache.get("mixture_manifest.json")
            if is_main_process():
                logger.info(
                    "Remote token mixture: hf://datasets/%s revision=%s cache=%s limit=%.1f GiB",
                    repo_id,
                    revision,
                    self.root,
                    cache_max_bytes / 1024**3,
                )
        else:
            self.root = Path(root)
            manifest_path = self.root / "mixture_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"pretokenized mixture manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != "tr-hash-token-mixture-v1":
            raise ValueError("unsupported pretokenized mixture format")
        if self.manifest.get("dtype") != "uint16":
            raise ValueError("pretokenized mixture must use uint16 tokens")
        self.seq_len = int(self.manifest["seq_len"])
        self.rank = int(rank)
        self.world_size = int(world_size)
        if self.rank < 0 or self.world_size < 1 or self.rank >= self.world_size:
            raise ValueError("invalid distributed rank/world_size")
        sources = []
        for entry in self.manifest["sources"]:
            relative_manifest = str(entry["manifest"])
            source_manifest = (
                self._hub_cache.get(relative_manifest)
                if self._hub_cache is not None
                else self.root / relative_manifest
            )
            sources.append(
                TextCorpusSource(
                    name=entry["name"],
                    weight=float(entry["weight"]),
                    data_files=str(source_manifest),
                )
            )
        self.sources = tuple(sources)
        validate_corpus_mixture(self.sources)
        self._rows_by_source = {
            entry["name"]: int(entry["rows"]) for entry in self.manifest["sources"]
        }
        self._source_manifests: dict[str, Mapping[str, Any]] = {}
        self._shards_by_source: dict[str, dict[str, Mapping[str, Any]]] = {}
        required_remote_files = {"mixture_manifest.json"}
        for source in self.sources:
            source_manifest_path = Path(source.data_files)
            source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
            if source_manifest.get("format") != "tr-hash-token-mixture-v1":
                raise ValueError(f"unsupported source manifest format for {source.name}")
            if source_manifest.get("dtype") != "uint16":
                raise ValueError(f"source {source.name} must use uint16 tokens")
            if int(source_manifest["seq_len"]) != self.seq_len:
                raise ValueError(f"sequence length mismatch for {source.name}")
            shards = source_manifest.get("shards", ())
            if not shards:
                raise ValueError(f"source {source.name} does not contain any token shards")
            filenames = [str(shard["file"]) for shard in shards]
            if len(filenames) != len(set(filenames)):
                raise ValueError(f"source {source.name} contains duplicate shard filenames")
            if any(_safe_relative_path(filename).name != filename for filename in filenames):
                raise ValueError(
                    f"source {source.name} shard filenames must not contain directories"
                )
            rows = sum(int(shard["rows"]) for shard in shards)
            expected_rows = self._rows_by_source[source.name]
            if rows != expected_rows or int(source_manifest["rows"]) != expected_rows:
                raise ValueError(
                    f"source {source.name} shard coverage mismatch: "
                    f"shards={rows}, source={source_manifest['rows']}, mixture={expected_rows}"
                )
            for shard in shards:
                shard_rows = int(shard["rows"])
                shard_tokens = int(shard["tokens"])
                expected_tokens = shard_rows * self.seq_len + 1
                if shard_rows < 1 or shard_tokens != expected_tokens:
                    raise ValueError(
                        f"invalid shard layout for {source.name}/{shard['file']}: "
                        f"rows={shard_rows}, tokens={shard_tokens}, expected={expected_tokens}"
                    )
                relative = str(Path("corpora") / source.name / str(shard["file"]))
                if self._hub_cache is None:
                    token_path = source_manifest_path.parent / str(shard["file"])
                    expected_bytes = int(
                        shard.get("bytes", expected_tokens * TOKEN_DTYPE.itemsize)
                    )
                    if not token_path.is_file():
                        raise FileNotFoundError(f"token shard missing: {token_path}")
                    if token_path.stat().st_size != expected_bytes:
                        raise ValueError(
                            f"corrupt token shard {token_path}: "
                            f"{token_path.stat().st_size} != {expected_bytes} bytes"
                        )
                else:
                    required_remote_files.add(relative)
            self._source_manifests[source.name] = source_manifest
            self._shards_by_source[source.name] = {
                str(shard["file"]): shard for shard in shards
            }

        if self._hub_cache is not None:
            required_remote_files.update(
                str(entry["manifest"]) for entry in self.manifest["sources"]
            )
            missing = sorted(required_remote_files - self._hub_cache.list_files())
            if missing:
                preview = ", ".join(missing[:5])
                suffix = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
                raise FileNotFoundError(
                    f"remote token mixture is incomplete; missing {preview}{suffix}"
                )
            if is_main_process():
                logger.info(
                    "Remote token preflight passed: %d files, %d rows, %d training tokens",
                    len(required_remote_files),
                    sum(self._rows_by_source.values()),
                    sum(self._rows_by_source.values()) * self.seq_len,
                )
        self._replay_phases = self._load_replay_plan(replay_plan)
        if self._replay_phases is None:
            self.unique_tokens = sum(self._rows_by_source.values()) * self.seq_len
            self.trained_tokens = self.unique_tokens
        else:
            unique_rows: dict[tuple[str, str], int] = {}
            trained_rows = 0
            for phase in self._replay_phases:
                phase_rows = 0
                for source_name, selections in phase["sources"].items():
                    for selection in selections:
                        key = (source_name, selection["file"])
                        unique_rows[key] = max(
                            unique_rows.get(key, 0), int(selection["rows"])
                        )
                        phase_rows += int(selection["rows"])
                trained_rows += phase_rows * int(phase["passes"])
            self.unique_tokens = sum(unique_rows.values()) * self.seq_len
            self.trained_tokens = trained_rows * self.seq_len
            if is_main_process():
                logger.info(
                    "Token replay plan: %d phases, %.3fB unique tokens, %.3fB trained tokens",
                    len(self._replay_phases),
                    self.unique_tokens / 1e9,
                    self.trained_tokens / 1e9,
                )

    def _load_replay_plan(
        self, replay_plan: str | Path | Mapping[str, Any] | None
    ) -> tuple[Mapping[str, Any], ...] | None:
        if replay_plan is None:
            return None
        if isinstance(replay_plan, Mapping):
            plan = dict(replay_plan)
        else:
            plan_path = Path(replay_plan)
            if not plan_path.is_file():
                if self._hub_cache is not None:
                    plan_path = self._hub_cache.get(str(replay_plan))
                else:
                    plan_path = self.root / plan_path
            if not plan_path.is_file():
                raise FileNotFoundError(f"token replay plan not found: {replay_plan}")
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("format") != REPLAY_PLAN_FORMAT:
            raise ValueError("unsupported token replay plan format")
        if int(plan.get("seq_len", -1)) != self.seq_len:
            raise ValueError("token replay plan sequence length mismatch")
        phases = plan.get("phases")
        if not isinstance(phases, list) or not phases:
            raise ValueError("token replay plan must contain at least one phase")
        normalized = []
        for phase_index, phase in enumerate(phases):
            name = str(phase.get("name", f"phase_{phase_index + 1}"))
            passes = int(phase.get("passes", 1))
            if passes < 1:
                raise ValueError(f"token replay phase {name} passes must be positive")
            phase_sources = phase.get("sources")
            if not isinstance(phase_sources, Mapping) or not phase_sources:
                raise ValueError(f"token replay phase {name} has no sources")
            normalized_sources = {}
            for source_name, selections in phase_sources.items():
                if source_name not in self._shards_by_source:
                    raise ValueError(
                        f"token replay phase {name} references unknown source {source_name}"
                    )
                if not isinstance(selections, list) or not selections:
                    raise ValueError(
                        f"token replay phase {name}/{source_name} has no shards"
                    )
                seen = set()
                normalized_selections = []
                for selection in selections:
                    filename = str(selection["file"])
                    if filename in seen:
                        raise ValueError(
                            f"token replay phase {name}/{source_name} repeats shard {filename}"
                        )
                    seen.add(filename)
                    try:
                        shard = self._shards_by_source[source_name][filename]
                    except KeyError as exc:
                        raise ValueError(
                            f"token replay phase {name}/{source_name} references "
                            f"unknown shard {filename}"
                        ) from exc
                    rows = int(selection.get("rows", shard["rows"]))
                    if rows < 1 or rows > int(shard["rows"]):
                        raise ValueError(
                            f"token replay phase {name}/{source_name}/{filename} "
                            f"rows={rows} outside 1..{shard['rows']}"
                        )
                    normalized_selections.append({"file": filename, "rows": rows})
                normalized_sources[source_name] = tuple(normalized_selections)
            normalized.append(
                {"name": name, "passes": passes, "sources": normalized_sources}
            )
        result = tuple(normalized)
        unique_rows: dict[tuple[str, str], int] = {}
        trained_rows = 0
        for phase in result:
            phase_rows = 0
            for source_name, selections in phase["sources"].items():
                for selection in selections:
                    key = (source_name, selection["file"])
                    unique_rows[key] = max(unique_rows.get(key, 0), selection["rows"])
                    phase_rows += selection["rows"]
            trained_rows += phase_rows * phase["passes"]
        actual_unique_tokens = sum(unique_rows.values()) * self.seq_len
        actual_trained_tokens = trained_rows * self.seq_len
        for key, actual in (
            ("unique_tokens", actual_unique_tokens),
            ("trained_tokens", actual_trained_tokens),
        ):
            if key in plan and int(plan[key]) != actual:
                raise ValueError(
                    f"token replay plan {key} mismatch: declared={plan[key]}, actual={actual}"
                )
        return result

    def _source_rows(
        self,
        source: TextCorpusSource,
        *,
        shard_index: int,
        shard_count: int,
        selections: Sequence[Mapping[str, Any]] | None = None,
    ) -> Iterator[dict[str, torch.Tensor]]:
        source_manifest_path = Path(source.data_files)
        global_row = 0
        if selections is None:
            selections = tuple(
                {"file": str(shard["file"]), "rows": int(shard["rows"])}
                for shard in self._source_manifests[source.name]["shards"]
            )
        for shard_position, selection in enumerate(selections):
            shard = self._shards_by_source[source.name][str(selection["file"])]
            rows = int(selection["rows"])
            full_rows = int(shard["rows"])
            full_tokens = full_rows * self.seq_len + 1
            expected_bytes = int(
                shard.get("bytes", full_tokens * TOKEN_DTYPE.itemsize)
            )
            expected_sha256 = shard.get("sha256")
            if self._hub_cache is None:
                token_path = source_manifest_path.parent / shard["file"]
                context = nullcontext(token_path)
            else:
                relative = str(
                    Path("corpora") / source.name / str(shard["file"])
                )
                context = self._hub_cache.pinned(
                    relative,
                    expected_bytes=expected_bytes,
                    expected_sha256=expected_sha256,
                )
            with context as token_path:
                tokens = np.memmap(token_path, mode="r", dtype=TOKEN_DTYPE)
                if tokens.size != full_tokens:
                    raise ValueError(
                        f"corrupt token shard {token_path}: {tokens.size} != {full_tokens}"
                    )
                if self._hub_cache is not None:
                    for next_selection in selections[
                        shard_position + 1 : shard_position + 1 + self._hub_cache.prefetch_shards
                    ]:
                        next_shard = self._shards_by_source[source.name][
                            str(next_selection["file"])
                        ]
                        next_full_rows = int(next_shard["rows"])
                        next_expected = next_full_rows * self.seq_len + 1
                        self._hub_cache.prefetch(
                            str(
                                Path("corpora")
                                / source.name
                                / str(next_shard["file"])
                            ),
                            expected_bytes=int(
                                next_shard.get(
                                    "bytes", next_expected * TOKEN_DTYPE.itemsize
                                )
                            ),
                            expected_sha256=next_shard.get("sha256"),
                        )
                first_local = (shard_index - global_row) % shard_count
                for local_row in range(first_local, rows, shard_count):
                    offset = local_row * self.seq_len
                    chunk = np.asarray(
                        tokens[offset : offset + self.seq_len + 1], dtype=np.int64
                    )
                    yield {
                        "input_ids": torch.from_numpy(chunk[:-1].copy()),
                        "labels": torch.from_numpy(chunk[1:].copy()),
                    }
                del tokens
            global_row += rows

    def _replay_phase_rows(
        self,
        phase: Mapping[str, Any],
        *,
        shard_index: int,
        shard_count: int,
    ) -> Iterator[dict[str, torch.Tensor]]:
        rows_by_source = {
            source_name: sum(int(selection["rows"]) for selection in selections)
            for source_name, selections in phase["sources"].items()
        }
        incompatible = {
            name: rows for name, rows in rows_by_source.items() if rows % shard_count
        }
        if incompatible:
            raise ValueError(
                f"token replay phase {phase['name']} rows must be divisible by "
                f"world_size * num_workers; shard_count={shard_count}, "
                f"incompatible={incompatible}"
            )
        local_rows = {name: rows // shard_count for name, rows in rows_by_source.items()}
        local_total = sum(local_rows.values())
        source_lookup = {source.name: source for source in self.sources}
        phase_sources = tuple(
            TextCorpusSource(
                name=name,
                weight=rows / local_total,
                data_files=source_lookup[name].data_files,
            )
            for name, rows in local_rows.items()
        )
        iterators = {
            source.name: iter(
                self._source_rows(
                    source_lookup[source.name],
                    shard_index=shard_index,
                    shard_count=shard_count,
                    selections=phase["sources"][source.name],
                )
            )
            for source in phase_sources
        }
        counts = {source.name: 0 for source in phase_sources}
        for _ in range(local_total):
            source = WeightedStreamingTextDataset._next_source(phase_sources, counts)
            try:
                sample = next(iterators[source.name])
            except StopIteration as exc:
                raise RuntimeError(
                    f"token replay phase {phase['name']} exhausted {source.name} "
                    "before its declared row budget"
                ) from exc
            counts[source.name] += 1
            yield sample

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = get_worker_info()
        worker_count = worker.num_workers if worker is not None else 1
        worker_id = worker.id if worker is not None else 0
        shard_count = self.world_size * worker_count
        shard_index = self.rank * worker_count + worker_id
        if self._replay_phases is not None:
            for phase in self._replay_phases:
                for _ in range(int(phase["passes"])):
                    yield from self._replay_phase_rows(
                        phase,
                        shard_index=shard_index,
                        shard_count=shard_count,
                    )
            return
        incompatible = {
            name: rows
            for name, rows in self._rows_by_source.items()
            if rows % shard_count
        }
        if incompatible:
            raise ValueError(
                "pretokenized source rows must be divisible by world_size * num_workers; "
                f"shard_count={shard_count}, incompatible={incompatible}"
            )
        iterators = {
            source.name: iter(
                self._source_rows(
                    source,
                    shard_index=shard_index,
                    shard_count=shard_count,
                )
            )
            for source in self.sources
        }
        counts = {source.name: 0 for source in self.sources}
        while True:
            source = WeightedStreamingTextDataset._next_source(self.sources, counts)
            try:
                sample = next(iterators[source.name])
            except StopIteration:
                return
            counts[source.name] += 1
            yield sample
