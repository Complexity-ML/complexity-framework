#!/usr/bin/env python3
"""Materialize the 200B TR-Hash corpus mixture as uint16 mmap shards."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import logging
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from complexity.training import TextCorpusSource, allocate_weighted_counts
from scripts.train_tr_hash_200m_200b import TARGET_TOKENS, corpus_sources

logger = logging.getLogger("tr_hash_200m_tokenizer")
FORMAT = "tr-hash-token-mixture-v1"
DEFAULT_HF_REPO = "Pacific-i64/data-32k-200b-tokens"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TokenShardWriter:
    """Write one continuous token stream into independently mmap-able shards."""

    def __init__(
        self,
        root: Path,
        *,
        seq_len: int,
        total_rows: int,
        rows_per_shard: int,
    ) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        if any(self.root.iterdir()):
            raise FileExistsError(
                f"partial output already exists for {self.root.name}: {self.root}. "
                "Keep completed corpora, but clear this incomplete corpus directory before retrying."
            )
        self.seq_len = int(seq_len)
        self.total_rows = int(total_rows)
        self.rows_per_shard = int(rows_per_shard)
        self.rows_written = 0
        self.source_tokens_written = 0
        self.shards: list[dict[str, int | str]] = []
        self._mapping: np.memmap | None = None
        self._position = 0
        self._current_rows = 0
        self._partial_path: Path | None = None
        self._final_path: Path | None = None
        self._carry_token: int | None = None

    @property
    def complete(self) -> bool:
        return self.rows_written == self.total_rows and self._mapping is None

    @property
    def required_source_tokens(self) -> int:
        return self.total_rows * self.seq_len + 1

    def _open_shard(self) -> None:
        remaining_rows = self.total_rows - self.rows_written
        if remaining_rows <= 0:
            return
        self._current_rows = min(self.rows_per_shard, remaining_rows)
        index = len(self.shards)
        self._final_path = self.root / f"tokens-{index:05d}.bin"
        self._partial_path = self.root / f"tokens-{index:05d}.bin.partial"
        token_count = self._current_rows * self.seq_len + 1
        self._mapping = np.memmap(
            self._partial_path,
            mode="w+",
            dtype=np.uint16,
            shape=(token_count,),
        )
        self._position = 0
        if self._carry_token is not None:
            self._mapping[0] = self._carry_token
            self._position = 1

    def _close_full_shard(self) -> None:
        if self._mapping is None or self._position != self._mapping.size:
            raise RuntimeError("cannot finalize an incomplete token shard")
        self._carry_token = int(self._mapping[-1])
        token_count = int(self._mapping.size)
        self._mapping.flush()
        del self._mapping
        self._mapping = None
        assert self._partial_path is not None and self._final_path is not None
        os.replace(self._partial_path, self._final_path)
        sha256 = _sha256_file(self._final_path)
        self.shards.append(
            {
                "file": self._final_path.name,
                "rows": self._current_rows,
                "tokens": token_count,
                "bytes": self._final_path.stat().st_size,
                "sha256": sha256,
            }
        )
        self.rows_written += self._current_rows
        self._position = 0

    def feed(self, tokens: np.ndarray) -> int:
        """Consume new source tokens; duplicated shard-boundary tokens are internal."""

        if tokens.dtype != np.uint16:
            tokens = tokens.astype(np.uint16, copy=False)
        consumed = 0
        while consumed < tokens.size and self.rows_written < self.total_rows:
            if self._mapping is None:
                self._open_shard()
            assert self._mapping is not None
            available = self._mapping.size - self._position
            take = min(available, tokens.size - consumed)
            self._mapping[self._position : self._position + take] = tokens[
                consumed : consumed + take
            ]
            self._position += take
            consumed += take
            self.source_tokens_written += take
            if self._position == self._mapping.size:
                self._close_full_shard()
        return consumed

    def write_manifest(self, *, source: TextCorpusSource) -> Path:
        if not self.complete:
            raise RuntimeError(
                f"source {source.name} incomplete: {self.rows_written}/{self.total_rows} rows"
            )
        manifest = {
            "format": FORMAT,
            "source": source.name,
            "weight": source.weight,
            "seq_len": self.seq_len,
            "dtype": "uint16",
            "rows": self.total_rows,
            "trained_tokens": self.total_rows * self.seq_len,
            "source_tokens_consumed": self.source_tokens_written,
            "shards": self.shards,
        }
        path = self.root / "manifest.json"
        temporary = path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        return path


def resolve_layout(
    *,
    target_tokens: int,
    seq_len: int,
    global_batch_sequences: int,
    sources: Sequence[TextCorpusSource],
) -> tuple[int, int, dict[str, int]]:
    """Round once to complete optimizer updates, then allocate exact quotas."""

    if target_tokens < 1 or seq_len < 1 or global_batch_sequences < 1:
        raise ValueError("tokenization layout values must be positive")
    total_rows = math.ceil(target_tokens / (seq_len * global_batch_sequences))
    total_rows *= global_batch_sequences
    rows_by_source = allocate_weighted_counts(total_rows, sources)
    return total_rows, total_rows * seq_len, rows_by_source


STACK_SWH_SENTINEL = "__software_heritage__"


def _software_heritage_stream(
    *, languages: Sequence[str], download_workers: int
) -> Iterator[dict[str, str]]:
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError as exc:
        raise RuntimeError(
            "Direct Stack-Edu resolution requires boto3: pip install boto3"
        ) from exc
    from datasets import load_dataset

    client = boto3.client(
        "s3",
        config=Config(
            max_pool_connections=download_workers,
            retries={"max_attempts": 8, "mode": "adaptive"},
            signature_version=UNSIGNED,
        ),
    )

    def fetch(example: Mapping[str, object]) -> dict[str, str] | None:
        blob_id = str(example["blob_id"])
        try:
            response = client.get_object(Bucket="softwareheritage", Key=f"content/{blob_id}")
            with gzip.GzipFile(fileobj=response["Body"]) as compressed:
                text = compressed.read().decode("utf-8", errors="ignore")
            return {"text": text} if text else None
        except Exception as exc:  # individual missing/corrupt blobs must not kill 20B-token jobs
            logger.debug("Stack-Edu blob %s unavailable: %s", blob_id, exc)
            return None

    metadata_streams: list[tuple[str, Iterator[Mapping[str, object]]]] = []
    for language in languages:
        logger.info("Stack-Edu: loading %s metadata", language)
        metadata = load_dataset(
                "HuggingFaceTB/stack-edu",
                language,
                split="train",
                streaming=True,
            )
        metadata_streams.append((language, iter(metadata)))

    with ThreadPoolExecutor(max_workers=download_workers) as executor:
        while metadata_streams:
            remaining: list[tuple[str, Iterator[Mapping[str, object]]]] = []
            for language, metadata in metadata_streams:
                batch = list(itertools.islice(metadata, download_workers * 4))
                if not batch:
                    logger.info("Stack-Edu: exhausted %s metadata", language)
                    continue
                remaining.append((language, metadata))
                logger.debug("Stack-Edu: resolving %s batch (%d blobs)", language, len(batch))
                resolved_batch = list(executor.map(fetch, batch))
                successes = sum(resolved is not None for resolved in resolved_batch)
                if successes < max(1, len(batch) // 4):
                    raise RuntimeError(
                        "Stack-Edu Software Heritage resolution failure rate exceeded 75%; "
                        "check network/S3 access before continuing"
                    )
                for resolved in resolved_batch:
                    if resolved is not None:
                        yield resolved
            metadata_streams = remaining


def _load_source(
    source: TextCorpusSource,
    *,
    stack_edu_languages: Sequence[str],
    stack_download_workers: int,
):
    from datasets import load_dataset

    if source.name == "stack_edu" and source.data_files == STACK_SWH_SENTINEL:
        return _software_heritage_stream(
            languages=stack_edu_languages,
            download_workers=stack_download_workers,
        )
    if source.data_files is not None:
        return load_dataset(
            "json", data_files=source.data_files, split=source.split, streaming=True
        )
    return load_dataset(
        source.dataset_id,
        source.config_name,
        split=source.split,
        streaming=True,
    )


def _text_batches(
    dataset: Iterable[Mapping[str, object]], *, text_field: str, batch_size: int
) -> Iterator[list[str]]:
    batch: list[str] = []
    for example in dataset:
        text = example.get(text_field, "")
        if not isinstance(text, str) or not text:
            continue
        batch.append(text)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _flatten_with_eos(encoded: Sequence[Sequence[int]], eos_token_id: int | None) -> np.ndarray:
    suffix = () if eos_token_id is None else (int(eos_token_id),)
    token_count = sum(len(tokens) + len(suffix) for tokens in encoded)
    return np.fromiter(
        itertools.chain.from_iterable(itertools.chain(tokens, suffix) for tokens in encoded),
        dtype=np.uint16,
        count=token_count,
    )


def tokenize_source(
    *,
    source: TextCorpusSource,
    tokenizer_path: str,
    output_root: Path,
    seq_len: int,
    rows: int,
    rows_per_shard: int,
    document_batch_size: int,
    log_every_tokens: int,
    stack_edu_languages: Sequence[str],
    stack_download_workers: int,
) -> Path:
    from transformers import PreTrainedTokenizerFast

    output = output_root / "corpora" / source.name
    manifest = output / "manifest.json"
    if manifest.is_file():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        if int(existing["rows"]) != rows or int(existing["seq_len"]) != seq_len:
            raise ValueError(f"completed corpus {source.name} has an incompatible layout")
        logger.info("Skipping completed corpus %s", source.name)
        return manifest

    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    if len(tokenizer) > np.iinfo(np.uint16).max + 1:
        raise ValueError("uint16 token storage requires a tokenizer with at most 65,536 tokens")
    writer = TokenShardWriter(
        output,
        seq_len=seq_len,
        total_rows=rows,
        rows_per_shard=rows_per_shard,
    )
    dataset = _load_source(
        source,
        stack_edu_languages=stack_edu_languages,
        stack_download_workers=stack_download_workers,
    )
    next_log = log_every_tokens
    try:
        for texts in _text_batches(
            dataset, text_field=source.text_field, batch_size=document_batch_size
        ):
            encoded = tokenizer(
                texts,
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )["input_ids"]
            flat = _flatten_with_eos(encoded, tokenizer.eos_token_id)
            writer.feed(flat)
            if writer.source_tokens_written >= next_log:
                logger.info(
                    "%s: %.3fB / %.3fB source tokens",
                    source.name,
                    writer.source_tokens_written / 1e9,
                    writer.required_source_tokens / 1e9,
                )
                next_log += log_every_tokens
            if writer.complete:
                return writer.write_manifest(source=source)
    finally:
        close = getattr(dataset, "close", None)
        if close is not None:
            close()
    raise RuntimeError(
        f"source {source.name} exhausted at {writer.source_tokens_written:,} tokens; "
        f"need {writer.required_source_tokens:,}"
    )


def write_mixture_manifest(
    *,
    output_root: Path,
    sources: Sequence[TextCorpusSource],
    seq_len: int,
    requested_tokens: int,
    actual_tokens: int,
    global_batch_sequences: int,
    rows_by_source: Mapping[str, int],
) -> Path:
    entries = []
    for source in sources:
        relative = Path("corpora") / source.name / "manifest.json"
        path = output_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"corpus manifest missing: {path}")
        entries.append(
            {
                "name": source.name,
                "weight": source.weight,
                "rows": rows_by_source[source.name],
                "trained_tokens": rows_by_source[source.name] * seq_len,
                "manifest": str(relative),
            }
        )
    manifest = {
        "format": FORMAT,
        "dtype": "uint16",
        "seq_len": seq_len,
        "requested_tokens": requested_tokens,
        "actual_tokens": actual_tokens,
        "global_batch_sequences": global_batch_sequences,
        "sources": entries,
    }
    path = output_root / "mixture_manifest.json"
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def upload_dataset_subset(
    *,
    output_root: Path,
    repo_id: str,
    allow_patterns: Sequence[str],
    token: str | None,
    workers: int,
    api=None,
) -> None:
    """Upload one completed subset with the Hub's resumable large-folder path."""

    if workers < 1:
        raise ValueError("Hugging Face upload workers must be positive")
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
    logger.info(
        "Uploading %s to hf://datasets/%s",
        ", ".join(allow_patterns),
        repo_id,
    )
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=output_root,
        allow_patterns=list(allow_patterns),
        num_workers=workers,
        print_report=True,
        print_report_every=60,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--stack-edu-data",
        default=None,
        help="Optional materialized Stack-Edu JSONL glob. Default: resolve SWHIDs directly.",
    )
    parser.add_argument(
        "--stack-edu-languages",
        default="Python,Java,Cpp,JavaScript,TypeScript,Shell,Go,Rust",
    )
    parser.add_argument("--stack-download-workers", type=int, default=64)
    parser.add_argument("--output", default="artifacts/tr_hash_200m_200b_tokens")
    parser.add_argument("--target-tokens", type=int, default=TARGET_TOKENS)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--global-batch-sequences", type=int, default=512)
    parser.add_argument("--shard-trained-tokens", type=int, default=1_000_000_000)
    parser.add_argument("--document-batch-size", type=int, default=512)
    parser.add_argument("--parallel-corpora", type=int, default=2)
    parser.add_argument("--log-every-tokens", type=int, default=100_000_000)
    parser.add_argument(
        "--hf-repo",
        default=os.environ.get("HF_DATASET_REPO"),
        help=(
            "Optional Hugging Face dataset repo. Each completed corpus is uploaded "
            "with the resumable large-folder API; the global manifest is uploaded last."
        ),
    )
    parser.add_argument("--hf-upload-workers", type=int, default=32)
    parser.add_argument(
        "--hf-token-env",
        default="HF_TOKEN",
        help="Environment variable containing the Hub write token (never logged).",
    )
    parser.add_argument(
        "--corpora",
        default="all",
        help="Comma-separated source names, or all. Completed sources are skipped.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    sources = corpus_sources(args.stack_edu_data or STACK_SWH_SENTINEL)
    stack_edu_languages = tuple(
        language.strip()
        for language in args.stack_edu_languages.split(",")
        if language.strip()
    )
    if not stack_edu_languages:
        raise ValueError("--stack-edu-languages cannot be empty")
    total_rows, actual_tokens, rows_by_source = resolve_layout(
        target_tokens=args.target_tokens,
        seq_len=args.seq_len,
        global_batch_sequences=args.global_batch_sequences,
        sources=sources,
    )
    logger.info(
        "Layout: %s rows x %s = %s actual tokens (requested %s)",
        f"{total_rows:,}",
        f"{args.seq_len:,}",
        f"{actual_tokens:,}",
        f"{args.target_tokens:,}",
    )
    for source in sources:
        logger.info(
            "  %-20s %5.1f%%  rows=%s  tokens=%.6fB",
            source.name,
            source.weight * 100,
            f"{rows_by_source[source.name]:,}",
            rows_by_source[source.name] * args.seq_len / 1e9,
        )
    if args.dry_run:
        return

    selected_names = (
        {source.name for source in sources}
        if args.corpora == "all"
        else {name.strip() for name in args.corpora.split(",") if name.strip()}
    )
    known_names = {source.name for source in sources}
    unknown = selected_names - known_names
    if unknown:
        raise ValueError(f"unknown corpora: {sorted(unknown)}")
    selected = [source for source in sources if source.name in selected_names]
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    rows_per_shard = max(1, args.shard_trained_tokens // args.seq_len)
    hf_token = os.environ.get(args.hf_token_env) if args.hf_token_env else None
    hf_api = None
    if args.hf_repo:
        from huggingface_hub import HfApi

        hf_api = HfApi(token=hf_token)
        hf_api.repo_info(repo_id=args.hf_repo, repo_type="dataset")
        logger.info("Validated Hub destination: hf://datasets/%s", args.hf_repo)

    with ThreadPoolExecutor(max_workers=args.parallel_corpora) as executor:
        futures = {
            executor.submit(
                tokenize_source,
                source=source,
                tokenizer_path=args.tokenizer,
                output_root=output_root,
                seq_len=args.seq_len,
                rows=rows_by_source[source.name],
                rows_per_shard=rows_per_shard,
                document_batch_size=args.document_batch_size,
                log_every_tokens=args.log_every_tokens,
                stack_edu_languages=stack_edu_languages,
                stack_download_workers=args.stack_download_workers,
            ): source.name
            for source in selected
        }
        for future in as_completed(futures):
            source_name = futures[future]
            logger.info("Completed %s: %s", source_name, future.result())
            if args.hf_repo:
                upload_dataset_subset(
                    output_root=output_root,
                    repo_id=args.hf_repo,
                    allow_patterns=(f"corpora/{source_name}/**",),
                    token=hf_token,
                    workers=args.hf_upload_workers,
                    api=hf_api,
                )

    if all((output_root / "corpora" / source.name / "manifest.json").is_file() for source in sources):
        path = write_mixture_manifest(
            output_root=output_root,
            sources=sources,
            seq_len=args.seq_len,
            requested_tokens=args.target_tokens,
            actual_tokens=actual_tokens,
            global_batch_sequences=args.global_batch_sequences,
            rows_by_source=rows_by_source,
        )
        logger.info("Pretokenized mixture ready: %s", path)
        if args.hf_repo:
            upload_dataset_subset(
                output_root=output_root,
                repo_id=args.hf_repo,
                allow_patterns=("mixture_manifest.json",),
                token=hf_token,
                workers=args.hf_upload_workers,
                api=hf_api,
            )
            logger.info("Published complete mixture: hf://datasets/%s", args.hf_repo)
    else:
        logger.info("Selected corpora complete; run the remaining corpora to finalize the mixture")


if __name__ == "__main__":
    main()
