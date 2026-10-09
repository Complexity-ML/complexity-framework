import gzip
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from complexity.training.agentic_candidates import (
    effective_stage_config,
    pack_candidates,
    stage_source,
)
from complexity.training.agentic_corpus import (
    BenchmarkContaminationIndex,
    build_benchmark_index,
    content_sha256,
    normalize_text,
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


def _fixture(tmp_path: Path) -> tuple[dict, dict, Path, BenchmarkContaminationIndex]:
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
    benchmark_path = build_benchmark_index(())
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
            benchmark_index=benchmark_path,
            work_dir=work_dir,
            publisher=publisher,
        )

    result = stage_source(
        source_index=0,
        source=source,
        config=config,
        tokenizer_path=tokenizer_path,
        benchmark_index=benchmark_path,
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
            benchmark_index=benchmark_path,
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
                    "reference_tokens": len(tokenizer.encode(text, add_special_tokens=False).ids),
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
            "reference_tokens": sum(row["reference_tokens"] for row in rows),
            "bytes": shard_path.stat().st_size,
            "sha256": hashlib.sha256(shard_path.read_bytes()).hexdigest(),
        }
        source_manifests.append(
            {
                "source": name,
                "tokenizer_sha256": hashlib.sha256(
                    (tokenizer_dir / "tokenizer.json").read_bytes()
                ).hexdigest(),
                "complete": True,
                "candidate_target_tokens": 64,
                "retained_tokens": sum(row["reference_tokens"] for row in rows),
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


def test_wrong_candidate_receipt_preserves_uncommitted_cursor_and_retries(tmp_path: Path) -> None:
    config, source, tokenizer_path, benchmark = _fixture(tmp_path)

    class WrongReceipt(MemoryPublisher):
        def publish_file(self, local_path: Path, relative: str) -> dict[str, object]:
            receipt = super().publish_file(local_path, relative)
            receipt["sha256"] = "0" * 64
            return receipt

    work = tmp_path / "work"
    with pytest.raises(ValueError, match="publisher receipt"):
        stage_source(
            source_index=0,
            source=source,
            config=config,
            tokenizer_path=tokenizer_path,
            benchmark_index=benchmark,
            work_dir=work,
            publisher=WrongReceipt(),
        )
    with sqlite3.connect(work / "00-fixture/state.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM progress").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM seen").fetchone()[0] == 0
    assert list(work.rglob("*.partial*"))
    publisher = MemoryPublisher()
    manifest = stage_source(
        source_index=0,
        source=source,
        config=config,
        tokenizer_path=tokenizer_path,
        benchmark_index=benchmark,
        work_dir=work,
        publisher=publisher,
    )
    rows = [
        json.loads(line)
        for line in gzip.decompress(publisher.files[manifest["shards"][0]["repo_path"]])
        .decode()
        .splitlines()
    ]
    assert [row["source_record_id"] for row in rows] == [0, 1]
    assert manifest["complete"] is True


def test_candidate_manifest_repair_does_not_reread_completed_source(
    tmp_path: Path, monkeypatch
) -> None:
    config, source, tokenizer_path, benchmark = _fixture(tmp_path)

    class FailManifest(MemoryPublisher):
        def publish_json(self, payload, relative: str, work_dir: Path) -> None:
            raise RuntimeError("manifest acknowledgement failed")

    work = tmp_path / "work"
    with pytest.raises(RuntimeError, match="manifest acknowledgement"):
        stage_source(
            source_index=0,
            source=source,
            config=config,
            tokenizer_path=tokenizer_path,
            benchmark_index=benchmark,
            work_dir=work,
            publisher=FailManifest(),
        )
    monkeypatch.setattr(
        "complexity.training.agentic_candidates.iter_source",
        lambda *args, **kwargs: pytest.fail("completed source was reread"),
    )
    publisher = MemoryPublisher()
    manifest = stage_source(
        source_index=0,
        source=source,
        config=config,
        tokenizer_path=tokenizer_path,
        benchmark_index=benchmark,
        work_dir=work,
        publisher=publisher,
    )
    assert manifest["complete"] is True
    assert len(manifest["shards"]) == 1
    assert not list(work.rglob("*.partial*"))


@pytest.mark.parametrize("changed", ["source", "tokenizer", "protected_index"])
def test_candidate_resume_rejects_changed_identity(tmp_path: Path, changed: str) -> None:
    config, source, tokenizer_path, benchmark = _fixture(tmp_path)
    work = tmp_path / "work"
    stage_source(
        source_index=0,
        source=source,
        config=config,
        tokenizer_path=tokenizer_path,
        benchmark_index=benchmark,
        work_dir=work,
        publisher=MemoryPublisher(),
    )
    if changed == "source":
        with Path(source["path"]).open("a") as stream:
            stream.write('{"text":"changed input"}\n')
    elif changed == "tokenizer":
        path = tokenizer_path / "tokenizer.json"
        path.write_text(path.read_text() + "\n")
    else:
        benchmark.add("protected", "one two three four five six seven eight nine ten")
    with pytest.raises(ValueError, match="another build"):
        stage_source(
            source_index=0,
            source=source,
            config=config,
            tokenizer_path=tokenizer_path,
            benchmark_index=benchmark,
            work_dir=work,
            publisher=MemoryPublisher(),
        )


def _staged_fixture(tmp_path: Path):
    config, source, tokenizer_path, benchmark = _fixture(tmp_path)
    config["candidate_oversample"] = 1.5
    publisher = MemoryPublisher()
    manifest = stage_source(
        source_index=0,
        source=source,
        config=config,
        tokenizer_path=tokenizer_path,
        benchmark_index=benchmark,
        work_dir=tmp_path / "stage",
        publisher=publisher,
    )
    for index, shard in enumerate(manifest["shards"]):
        local = tmp_path / f"candidate-{index}.jsonl.gz"
        local.write_bytes(publisher.files[shard["repo_path"]])
        shard["local_path"] = str(local)
    candidate_manifest = tmp_path / "candidates.json"
    candidate_manifest.write_text(json.dumps({"complete": True, "sources": [manifest]}))
    source["weight"] = 1.0
    config.update(
        version=1,
        target_tokens=8,
        bucket_targets={"foundation": 8},
        seq_len=2,
        global_batch_sequences=1,
        shard_trained_tokens=4,
    )
    curriculum = {
        "total_tokens": 8,
        "phases": [
            {
                "name": "only",
                "target_tokens": 8,
                "bucket_tokens": {"foundation": 8},
                "bucket_shares": {"foundation": 1.0},
            }
        ],
        "invariants": {"replay": False, "each_packed_row_consumed_once": True},
    }
    return config, curriculum, tokenizer_path, candidate_manifest


def test_candidate_stage_and_pack_reads_native_runtime_rows(tmp_path: Path) -> None:
    from complexity.training import PretokenizedCorpusMixtureDataset

    config, curriculum, tokenizer_path, manifest_path = _staged_fixture(tmp_path)
    hub = tmp_path / "hub"
    state = pack_candidates(
        config=config,
        curriculum=curriculum,
        tokenizer_path=tokenizer_path,
        candidate_manifest_path=manifest_path,
        work_dir=tmp_path / "pack",
        publisher=FilePublisher(hub),
    )
    dataset = PretokenizedCorpusMixtureDataset(hub, replay_plan=hub / "pretrain_plan.json")
    assert len(list(dataset)) == 4
    assert dataset.unique_tokens == dataset.trained_tokens == 8
    assert state["execution_mode"] == "deterministic_sequential"
    assert (
        state["candidate_manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("changed", ["bytes", "tokenizer", "tail", "reference_tokens", "remote"])
def test_candidate_pack_preflight_rejects_invalid_artifacts_before_writes(
    tmp_path: Path, changed: str
) -> None:
    config, curriculum, tokenizer_path, manifest_path = _staged_fixture(tmp_path)
    payload = json.loads(manifest_path.read_text())
    manifest = payload["sources"][0]
    shard = manifest["shards"][0]
    if changed == "tokenizer":
        manifest["tokenizer_sha256"] = "0" * 64
    elif changed == "remote":
        shard.pop("local_path")
    else:
        local = Path(shard["local_path"])
        if changed == "bytes":
            local.write_bytes(local.read_bytes() + b"tampered")
        else:
            rows = [
                json.loads(line)
                for line in gzip.decompress(local.read_bytes()).decode().splitlines()
            ]
            if changed == "tail":
                rows[-1]["text"] = "changed tail"
            else:
                rows[-1]["reference_tokens"] += 1
                shard["reference_tokens"] += 1
                manifest["retained_tokens"] += 1
            local.write_bytes(
                gzip.compress("".join(json.dumps(row) + "\n" for row in rows).encode(), mtime=0)
            )
            shard.update(
                bytes=local.stat().st_size, sha256=hashlib.sha256(local.read_bytes()).hexdigest()
            )
    manifest_path.write_text(json.dumps(payload))
    work = tmp_path / "pack"
    publisher = MemoryPublisher()
    with pytest.raises(ValueError):
        pack_candidates(
            config=config,
            curriculum=curriculum,
            tokenizer_path=tokenizer_path,
            candidate_manifest_path=manifest_path,
            work_dir=work,
            publisher=publisher,
        )
    assert not work.exists()
    assert publisher.files == {}
