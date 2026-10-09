import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from complexity.training.agentic_sharding import (
    StateStore,
    allocate_rows,
    build,
    skip_source_records,
)


def test_restore_skip_reports_bounded_progress(caplog: pytest.LogCaptureFixture) -> None:
    iterator = iter({"text": str(index)} for index in range(5))

    with caplog.at_level("INFO"):
        skip_source_records(iterator, 5, source_name="fixture", log_every=2)

    messages = [record.getMessage() for record in caplog.records]
    assert any("fixture restore progress: 40.00%" in message for message in messages)
    assert any("fixture restore progress: 100.00%" in message for message in messages)
    assert next(iterator, None) is None


def test_50b_layout_is_optimizer_aligned_and_exactly_allocated() -> None:
    sources = [
        {"name": "stack", "weight": 0.4},
        {"name": "web", "weight": 0.2},
        {"name": "math_a", "weight": 0.15},
        {"name": "math_b", "weight": 0.15},
        {"name": "synthetic", "weight": 0.1},
    ]
    rows, tokens, allocated = allocate_rows(
        target_tokens=50_000_000_000,
        seq_len=1_024,
        global_batch_sequences=64,
        sources=sources,
    )
    assert rows % 64 == 0
    assert tokens == 50_000_035_840
    assert sum(allocated.values()) == rows


def test_state_transaction_rolls_back_unpublished_hashes(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    digest = "ab" * 32
    store.begin()
    assert store.reserve_digest(digest)
    store.rollback()
    store.begin()
    assert store.reserve_digest(digest)
    store.rollback()


def test_state_commits_progress_shard_and_carry_atomically(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    store.begin()
    assert store.reserve_digest("cd" * 32)
    store.commit_shard(
        source="fixture",
        scanned=9,
        rows_done=4,
        source_tokens=17,
        last_token=12,
        carry=np.asarray([13, 14], dtype=np.uint16),
        counters={"retained": 1},
        signals={"tool": 1},
        shard={
            "shard_index": 0,
            "repo_path": "smoke/tokens-00000.bin",
            "rows": 4,
            "tokens": 17,
            "bytes": 34,
            "sha256": "ef" * 32,
        },
    )
    progress = store.progress("fixture")
    assert progress["scanned"] == 9
    assert progress["rows_done"] == 4
    assert progress["carry"].tolist() == [13, 14]
    assert store.shards("fixture")[0]["bytes"] == 34


def test_build_publishes_then_evicts_complete_local_shards(tmp_path: Path) -> None:
    vocab = {"<unk>": 0, **{f"token_{index}": index for index in range(1, 32_000)}}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer.save(str(tokenizer_dir / "tokenizer.json"))

    text = (
        "POST /v1/jobs with JSON arguments. First run the command, then verify the expected "
        "output using assert and a test case. " * 8
    )
    source = tmp_path / "source.jsonl"
    source.write_text("\n".join(f'{{"text": {text!r}}}'.replace("'", '"') for _ in range(3)))

    class Publisher:
        def __init__(self) -> None:
            self.files: dict[str, bytes] = {}

        def publish_file(self, path: Path, relative: str) -> dict[str, object]:
            payload = path.read_bytes()
            self.files[relative] = payload
            import hashlib

            return {
                "repo_path": relative,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }

        def publish_json(self, payload, relative: str, work_dir: Path) -> None:
            self.files[relative] = str(payload).encode()

    config = {
        "version": 1,
        "target_tokens": 16,
        "seq_len": 4,
        "global_batch_sequences": 1,
        "shard_trained_tokens": 8,
        "min_chars": 20,
        "max_chars": 10_000,
        "agentic_min_score": 4,
        "agentic_min_signal_classes": 2,
        "protected_benchmarks": [],
        "protected_benchmark_sources": [],
        "sources": [
            {
                "name": "fixture",
                "bucket": "agentic",
                "weight": 1.0,
                "path": str(source),
                "text_field": "text",
                "license_audit": "fixture",
            }
        ],
    }
    publisher = Publisher()
    state = build(
        config=config,
        tokenizer_path=tokenizer_dir,
        work_dir=tmp_path / "work",
        publisher=publisher,
    )

    assert state["sources"][0]["rows_done"] == 4
    assert len(state["sources"][0]["shards"]) == 2
    assert "corpora/fixture/tokens-00000.bin" in publisher.files
    assert "mixture_manifest.json" in publisher.files
    assert not list((tmp_path / "work" / "pending").rglob("*.bin"))


def test_full_partial_shard_resumes_after_upload_failure(tmp_path: Path) -> None:
    vocab = {"<unk>": 0, **{f"token_{index}": index for index in range(1, 32_000)}}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(
            [
                '{"text":"A unique technical procedure with enough explanatory words for test one."}',
                '{"text":"Another unique technical procedure with enough explanatory words for test two."}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class Publisher:
        def __init__(self, *, fail_corpus_upload: bool) -> None:
            self.fail_corpus_upload = fail_corpus_upload
            self.files: dict[str, bytes] = {}

        def publish_file(self, path: Path, relative: str) -> dict[str, object]:
            if self.fail_corpus_upload and relative.startswith("corpora/"):
                self.fail_corpus_upload = False
                raise RuntimeError("injected upload failure")
            payload = path.read_bytes()
            self.files[relative] = payload
            import hashlib

            return {
                "repo_path": relative,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }

        def publish_json(self, payload, relative: str, work_dir: Path) -> None:
            del work_dir
            self.files[relative] = str(payload).encode()

    config = {
        "version": 1,
        "target_tokens": 8,
        "seq_len": 4,
        "global_batch_sequences": 1,
        "shard_trained_tokens": 8,
        "tokenization_batch_size": 2,
        "producer_scan_batch_size": 2,
        "min_chars": 20,
        "max_chars": 10_000,
        "protected_benchmarks": [],
        "protected_benchmark_sources": [],
        "sources": [
            {
                "name": "fixture",
                "bucket": "foundation",
                "selection": "quality",
                "weight": 1.0,
                "path": str(source),
                "text_field": "text",
                "license_audit": "fixture",
            }
        ],
    }
    work_dir = tmp_path / "work"
    with pytest.raises(RuntimeError, match="injected upload failure"):
        build(
            config=config,
            tokenizer_path=tokenizer_dir,
            work_dir=work_dir,
            publisher=Publisher(fail_corpus_upload=True),
        )

    interrupted = StateStore(work_dir / "state.sqlite3").progress("fixture")
    assert interrupted["rows_done"] == 0
    assert interrupted["partial_position"] == 9
    assert (work_dir / "pending/fixture/tokens-00000.bin").is_file()

    publisher = Publisher(fail_corpus_upload=False)
    state = build(
        config=config,
        tokenizer_path=tokenizer_dir,
        work_dir=work_dir,
        publisher=publisher,
    )
    assert state["sources"][0]["rows_done"] == 2
    assert "corpora/fixture/tokens-00000.bin" in publisher.files
    assert not list((work_dir / "pending").rglob("*.bin*"))


class _MemoryPublisher:
    def __init__(self, *, wrong_hash=False, fail_state=False):
        self.files = {}
        self.wrong_hash, self.fail_state = wrong_hash, fail_state

    def publish_file(self, path, relative):
        payload = path.read_bytes()
        self.files[relative] = payload
        return {
            "repo_path": relative,
            "bytes": len(payload),
            "sha256": "00" * 32 if self.wrong_hash else hashlib.sha256(payload).hexdigest(),
        }

    def publish_json(self, payload, relative, work_dir):
        if self.fail_state and relative == "_state/state.json":
            raise RuntimeError("injected state publication failure")
        self.files[relative] = json.dumps(payload).encode()


@pytest.fixture()
def small_build(tmp_path):
    words = "first run a command and verify its expected output with assert then finish the procedure now".split()
    tokenizer = Tokenizer(
        models.WordLevel(
            vocab={"<unk>": 0, **{word: i for i, word in enumerate(words, 1)}}, unk_token="<unk>"
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"text": " ".join(words)}) + "\n")
    config = {
        "version": 1,
        "target_tokens": 8,
        "seq_len": 4,
        "global_batch_sequences": 1,
        "shard_trained_tokens": 8,
        "min_chars": 20,
        "max_chars": 1000,
        "protected_benchmarks": [],
        "sources": [
            {
                "name": "fixture",
                "bucket": "general",
                "weight": 1.0,
                "path": str(source),
                "license_audit": "fixture",
            }
        ],
    }
    return (
        {"config": config, "tokenizer_path": tokenizer_dir, "work_dir": tmp_path / "work"},
        source,
        tokenizer,
    )


def test_wrong_receipt_preserves_full_partial_and_resume_republishes_exact_bytes(small_build):
    args, _, _ = small_build
    with pytest.raises(ValueError, match="publisher receipt"):
        build(**args, publisher=_MemoryPublisher(wrong_hash=True))
    store = StateStore(args["work_dir"] / "state.sqlite3")
    assert store.progress("fixture")["rows_done"] == 0
    assert store.progress("fixture")["partial_position"] == 9
    store.close()
    pending = args["work_dir"] / "pending/fixture/tokens-00000.bin"
    original = pending.read_bytes()
    publisher = _MemoryPublisher()
    state = build(**args, publisher=publisher)
    assert state["sources"][0]["rows_done"] == 2
    assert publisher.files["corpora/fixture/tokens-00000.bin"] == original
    assert not pending.exists()


def test_committed_shard_survives_state_failure_and_resume_cleans_retained_file(small_build):
    args, _, _ = small_build
    with pytest.raises(RuntimeError, match="state publication"):
        build(**args, publisher=_MemoryPublisher(fail_state=True))
    store = StateStore(args["work_dir"] / "state.sqlite3")
    assert store.progress("fixture")["rows_done"] == 2
    store.close()
    assert (args["work_dir"] / "pending/fixture/tokens-00000.bin").exists()
    publisher = _MemoryPublisher()
    state = build(**args, publisher=publisher)
    assert state["sources"][0]["rows_done"] == 2
    assert "mixture_manifest.json" in publisher.files
    assert not list((args["work_dir"] / "pending").rglob("*.bin"))


def test_resume_rejects_changed_local_input(small_build):
    args, source, _ = small_build
    with pytest.raises(ValueError, match="publisher receipt"):
        build(**args, publisher=_MemoryPublisher(wrong_hash=True))
    source.write_text(source.read_text() + json.dumps({"text": "different source"}) + "\n")
    with pytest.raises(ValueError, match="incompatible build"):
        build(**args, publisher=_MemoryPublisher())


def test_shards_keep_one_boundary_token_and_match_native_loader(small_build, tmp_path):
    from complexity.training.corpus_mixture import PretokenizedCorpusMixtureDataset

    args, _, tokenizer = small_build
    args["config"]["target_tokens"] = 12
    publisher = _MemoryPublisher()
    state = build(**args, publisher=publisher)
    root = tmp_path / "published"
    for relative, payload in publisher.files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    first = np.frombuffer(publisher.files["corpora/fixture/tokens-00000.bin"], dtype="<u2")
    second = np.frombuffer(publisher.files["corpora/fixture/tokens-00001.bin"], dtype="<u2")
    assert first[-1] == second[0]
    assert state["actual_tokens"] == 12
    dataset = PretokenizedCorpusMixtureDataset(str(root), rank=0, world_size=1)
    expected = tokenizer.encode(
        "first run a command and verify its expected output with assert then finish the procedure now"
    ).ids
    rows = list(dataset)
    assert len(rows) == 3
    for index, row in enumerate(rows):
        assert row["input_ids"].tolist() == expected[index * 4 : index * 4 + 4]
        assert row["labels"].tolist() == expected[index * 4 + 1 : index * 4 + 5]


def test_layout_rejects_boolean_and_duplicate_sources():
    with pytest.raises(ValueError, match="positive integer"):
        allocate_rows(
            target_tokens=True,
            seq_len=4,
            global_batch_sequences=1,
            sources=[{"name": "x", "weight": 1.0}],
        )
    with pytest.raises(ValueError, match="unique"):
        allocate_rows(
            target_tokens=8,
            seq_len=4,
            global_batch_sequences=1,
            sources=[{"name": "x", "weight": 0.5}, {"name": "x", "weight": 0.5}],
        )


def test_restore_never_silently_skips_past_source_end():
    with pytest.raises(RuntimeError, match="exhausted"):
        skip_source_records(iter([]), 1, source_name="fixture")
