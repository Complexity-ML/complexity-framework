import gzip
import hashlib
import json
import pickle
import shutil
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from scripts.build_agentic_pretraining_corpus import (
    build_benchmark_index,
    content_sha256,
    normalize_text,
)
from scripts.pack_tr_hash_pretraining_125b_candidates import pack_candidates
from scripts.stage_tr_hash_pretraining_125b_candidates import (
    effective_stage_config,
    stage_source,
)


class MemoryPublisher:
    def __init__(self, *, fail_first_candidate: bool = False) -> None:
        self.fail_first_candidate = fail_first_candidate
        self.files: dict[str, bytes] = {}

    def publish_file(self, local_path: Path, relative: str) -> dict[str, object]:
        if self.fail_first_candidate and relative.endswith(".jsonl.gz"):
            self.fail_first_candidate = False
            raise RuntimeError("injected upload failure")
        payload = local_path.read_bytes()
        self.files[relative] = payload
        return {
            "repo_path": relative,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    def publish_json(self, payload, relative: str, work_dir: Path) -> None:
        del work_dir
        self.files[relative] = (json.dumps(payload, sort_keys=True) + "\n").encode()


def _tokenizer(path: Path) -> Path:
    tokenizer = Tokenizer(
        models.WordLevel(
            {
                "[UNK]": 0,
                "alpha": 1,
                "beta": 2,
                "gamma": 3,
                "delta": 4,
                "epsilon": 5,
                "zeta": 6,
                "eta": 7,
                "theta": 8,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    path.mkdir()
    tokenizer.save(str(path / "tokenizer.json"))
    return path


def _fixture(tmp_path: Path) -> tuple[dict, dict, Path, Path]:
    source_path = tmp_path / "source.jsonl"
    source_path.write_text(
        "\n".join(
            json.dumps({"id": index, "text": text})
            for index, text in enumerate(
                (
                    "alpha beta gamma delta",
                    "epsilon zeta eta theta",
                    "alpha beta gamma epsilon",
                )
            )
        )
        + "\n",
        encoding="utf-8",
    )
    source = {
        "name": "fixture",
        "bucket": "foundation",
        "selection": "quality",
        "target_tokens": 8,
        "path": str(source_path),
        "license_audit": "fixture",
    }
    config = {
        "seed": 1729,
        "candidate_oversample": 1.0,
        "candidate_shard_tokens": 4,
        "candidate_tokenization_batch_size": 2,
        "min_chars": 1,
        "max_chars": 1_000,
        "protected_benchmarks": [],
        "sources": [source],
    }
    tokenizer_path = _tokenizer(tmp_path / "tokenizer")
    benchmark_path = tmp_path / "benchmark.pkl"
    with benchmark_path.open("wb") as stream:
        pickle.dump(build_benchmark_index(()), stream)
    return config, source, tokenizer_path, benchmark_path


def test_candidate_stage_resumes_only_after_verified_upload(tmp_path: Path) -> None:
    config, source, tokenizer_path, benchmark_path = _fixture(tmp_path)
    publisher = MemoryPublisher(fail_first_candidate=True)
    work_dir = tmp_path / "work"

    with pytest.raises(RuntimeError, match="injected upload failure"):
        stage_source(
            source_index=0,
            source=source,
            config=config,
            tokenizer_path=tokenizer_path,
            benchmark_index_path=benchmark_path,
            work_dir=work_dir,
            publisher=publisher,
        )

    result = stage_source(
        source_index=0,
        source=source,
        config=config,
        tokenizer_path=tokenizer_path,
        benchmark_index_path=benchmark_path,
        work_dir=work_dir,
        publisher=publisher,
    )
    assert result["complete"] is True
    assert result["retained_tokens"] >= 8
    assert not list(work_dir.rglob("*.partial*"))
    candidate = publisher.files["_candidates/fixture/candidate-00000.jsonl.gz"]
    rows = [json.loads(line) for line in gzip.decompress(candidate).decode().splitlines()]
    assert [row["source_record_id"] for row in rows] == [0, 1]


def test_candidate_gzip_is_byte_reproducible(tmp_path: Path) -> None:
    config, source, tokenizer_path, benchmark_path = _fixture(tmp_path)
    first = MemoryPublisher()
    second = MemoryPublisher()
    for work_dir, publisher in ((tmp_path / "first", first), (tmp_path / "second", second)):
        stage_source(
            source_index=0,
            source=source,
            config=config,
            tokenizer_path=tokenizer_path,
            benchmark_index_path=benchmark_path,
            work_dir=work_dir,
            publisher=publisher,
        )
    path = "_candidates/fixture/candidate-00000.jsonl.gz"
    assert first.files[path] == second.files[path]


def test_pilot_config_is_explicitly_bounded_and_self_consistent() -> None:
    config = {
        "target_tokens": 300,
        "bucket_targets": {"agentic": 100, "foundation": 200},
        "sources": [
            {"name": "a", "bucket": "agentic", "target_tokens": 100, "weight": 1 / 3},
            {"name": "b", "bucket": "foundation", "target_tokens": 200, "weight": 2 / 3},
        ],
    }
    pilot = effective_stage_config(
        config,
        only_sources=("a", "b"),
        target_tokens_per_source=10,
        shuffle_buffer=1,
    )
    assert pilot["pilot"] is True
    assert pilot["target_tokens"] == 20
    assert pilot["bucket_targets"] == {"agentic": 10, "foundation": 10}
    assert [source["weight"] for source in pilot["sources"]] == [0.5, 0.5]
    assert [source["shuffle_buffer"] for source in pilot["sources"]] == [1, 1]
    assert config["target_tokens"] == 300


class FilePublisher:
    def __init__(self, root: Path) -> None:
        self.root = root

    def publish_file(self, path: Path, relative: str) -> dict[str, object]:
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        return {
            "repo_path": relative,
            "bytes": destination.stat().st_size,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        }

    def publish_json(self, payload, relative: str, work_dir: Path) -> None:
        del work_dir
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_final_pack_globally_deduplicates_staged_candidates(tmp_path: Path) -> None:
    vocab = {"<unk>": 0, **{f"token_{index}": index for index in range(1, 32_000)}}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_dir = tmp_path / "tokenizer-32k"
    tokenizer_dir.mkdir()
    tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
    tokenizer_manifest = tokenizer_dir / "agentic_tokenizer_manifest.json"
    tokenizer_manifest.write_text('{"vocab_size": 32000}\n', encoding="utf-8")

    sources = []
    source_manifests = []
    for name, bucket in (("agentic", "agentic"), ("foundation", "foundation")):
        rows = []
        for index in range(40):
            text = normalize_text(
                "shared verified technical record with enough explanatory words"
                if index == 0
                else f"{name} verified technical record number {index} with explanatory words"
            )
            rows.append(
                {
                    "text": text,
                    "source": name,
                    "bucket": bucket,
                    "agentic_signals": ["planning"] if bucket == "agentic" else [],
                    "content_sha256": content_sha256(text),
                    "reference_tokens": 8,
                }
            )
        shard_path = tmp_path / f"{name}.jsonl.gz"
        with shard_path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
                gz.write("".join(json.dumps(row) + "\n" for row in rows).encode())
        shard = {
            "shard_index": 0,
            "repo_path": f"_candidates/{name}/candidate-00000.jsonl.gz",
            "local_path": str(shard_path),
            "records": len(rows),
            "reference_tokens": 320,
            "bytes": shard_path.stat().st_size,
            "sha256": hashlib.sha256(shard_path.read_bytes()).hexdigest(),
        }
        source_manifests.append(
            {
                "source": name,
                "complete": True,
                "candidate_target_tokens": 64,
                "retained_tokens": 320,
                "shards": [shard],
            }
        )
        sources.append(
            {
                "name": name,
                "bucket": bucket,
                "selection": "quality",
                "target_tokens": 64,
                "weight": 0.5,
                "path": "unused-after-staging",
                "text_field": "text",
                "license_audit": "fixture",
            }
        )
    candidate_manifest = tmp_path / "candidate-manifest.json"
    candidate_manifest.write_text(
        json.dumps({"complete": True, "sources": source_manifests}) + "\n",
        encoding="utf-8",
    )
    config = {
        "version": 1,
        "target_tokens": 128,
        "bucket_targets": {"foundation": 64, "agentic": 64},
        "seq_len": 4,
        "global_batch_sequences": 2,
        "shard_trained_tokens": 32,
        "tokenization_batch_size": 2,
        "producer_candidate_batch_size": 2,
        "producer_scan_batch_size": 2,
        "protected_benchmarks": [],
        "protected_benchmark_sources": [],
        "tokenizer_contract": {
            "status": "validated",
            "vocab_size": 32_000,
            "required_manifest": "agentic_tokenizer_manifest.json",
            "revision": "a" * 40,
            "manifest_sha256": hashlib.sha256(tokenizer_manifest.read_bytes()).hexdigest(),
            "tokenizer_sha256": hashlib.sha256(
                (tokenizer_dir / "tokenizer.json").read_bytes()
            ).hexdigest(),
        },
        "sources": sources,
    }
    curriculum = {
        "version": 1,
        "total_tokens": 128,
        "phases": [
            {
                "name": "only",
                "target_tokens": 128,
                "bucket_tokens": {"foundation": 64, "agentic": 64},
                "bucket_shares": {"foundation": 0.5, "agentic": 0.5},
            }
        ],
        "invariants": {"replay": False, "each_packed_row_consumed_once": True},
    }
    hub = tmp_path / "hub"
    pack_candidates(
        config=config,
        curriculum=curriculum,
        candidate_manifest_path=candidate_manifest,
        tokenizer_path=tokenizer_dir,
        work_dir=tmp_path / "pack-work",
        repo_id="fixture",
        repo_prefix="production",
        candidate_cache_dir=tmp_path / "cache",
        publisher=FilePublisher(hub),
    )
    state = json.loads((hub / "_state/state.json").read_text(encoding="utf-8"))
    by_source = {source["name"]: source for source in state["sources"]}
    assert by_source["foundation"]["counters"]["exact_duplicate"] >= 1
    assert (hub / "mixture_manifest.json").is_file()
    assert (hub / "pretrain_plan.json").is_file()
