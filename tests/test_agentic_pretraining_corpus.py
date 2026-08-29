import json
import sys
import types
from io import StringIO
from pathlib import Path

import pytest

from scripts.build_agentic_pretraining_corpus import (
    BenchmarkContaminationIndex,
    _stack_download_workers,
    agentic_signals,
    benchmark_match,
    build_corpus,
    content_sha256,
    is_agentic_candidate,
    iter_source,
    normalize_text,
    quality_rejection,
    validate_config,
)


def test_stack_download_workers_supports_a_bounded_runtime_override(monkeypatch) -> None:
    source = {"download_workers": 256}
    assert _stack_download_workers(source) == 256
    monkeypatch.setenv("TR_HASH_STACK_DOWNLOAD_WORKERS", "64")
    assert _stack_download_workers(source) == 64
    monkeypatch.setenv("TR_HASH_STACK_DOWNLOAD_WORKERS", "0")
    with pytest.raises(ValueError, match="must be positive"):
        _stack_download_workers(source)


def _fixed_text(prefix: str, length: int = 400) -> str:
    filler = " Documentation explains configuration, validation, and expected output."
    text = prefix
    while len(text) < length:
        text += filler
    return text[:length]


def test_agentic_filter_requires_multiple_signal_classes() -> None:
    text = _fixed_text(
        "POST /v1/jobs with JSON arguments. ```python\nassert response.ok\n``` "
        "First execute the request, then verify the expected output."
    )
    accepted, signals, score = is_agentic_candidate(
        text,
        min_score=4,
        min_signal_classes=2,
    )
    assert accepted
    assert score >= 4
    assert {"tool", "code", "procedure", "verification"}.issubset(signals)


@pytest.mark.parametrize("name", ["ARC Challenge", "gsm8k", "AI2-ARC"])
def test_protected_benchmark_references_are_rejected(name: str) -> None:
    assert benchmark_match(f"Copied evaluation row from {name}.", (name,)) == name


def test_benchmark_index_rejects_prompt_without_benchmark_name() -> None:
    index = BenchmarkContaminationIndex()
    prompt = "Which material is best suited to keep a hot drink warm for several hours?"
    index.add("private_eval", prompt)

    document = f"Training example: {prompt} Explain the answer using physical principles."

    assert benchmark_match(document, (), index) == "private_eval"
    assert index.prompt_count == 1
    assert len(index.fingerprint()) == 64


def test_quality_and_hash_are_normalized() -> None:
    assert quality_rejection("short", min_chars=20, max_chars=1_000) == "too_short"
    assert content_sha256("Hello   world") == content_sha256("hello world")
    signals, score = agentic_signals("ordinary prose with no operational structure")
    assert signals == ()
    assert score == 0


def test_builder_keeps_general_and_agentic_buckets_with_provenance(tmp_path: Path) -> None:
    general_text = _fixed_text("A general technical document describes distributed systems.")
    agentic_text = _fixed_text(
        "POST /v1/jobs with JSON arguments. ```python\nassert response.ok\n``` "
        "First run the command, then verify its expected output."
    )
    general = tmp_path / "general.jsonl"
    agentic = tmp_path / "agentic.jsonl"
    general.write_text(json.dumps({"text": general_text}) + "\n", encoding="utf-8")
    agentic.write_text(json.dumps({"text": agentic_text}) + "\n", encoding="utf-8")
    config = {
        "version": 1,
        "target_bytes": 800,
        "min_chars": 100,
        "max_chars": 1_000,
        "agentic_min_score": 4,
        "agentic_min_signal_classes": 2,
        "protected_benchmarks": ["arc_challenge"],
        "sources": [
            {
                "name": "general",
                "bucket": "general",
                "weight": 0.5,
                "path": str(general),
                "license_audit": "test fixture",
            },
            {
                "name": "agentic",
                "bucket": "agentic",
                "weight": 0.5,
                "path": str(agentic),
                "license_audit": "test fixture",
            },
        ],
    }

    output = tmp_path / "output"
    manifest = build_corpus(config, output)

    expected_bytes = sum(
        len(normalize_text(text).encode("utf-8")) for text in (general_text, agentic_text)
    )
    assert manifest["retained_bytes"] == expected_bytes
    assert [source["retained_records"] for source in manifest["sources"]] == [1, 1]
    row = json.loads((output / "01-agentic.jsonl").read_text(encoding="utf-8"))
    assert row["bucket"] == "agentic"
    assert row["content_sha256"] == content_sha256(agentic_text)
    assert "tool" in row["agentic_signals"]


def test_config_rejects_untracked_licenses() -> None:
    with pytest.raises(ValueError, match="license_audit"):
        validate_config(
            {
                "version": 1,
                "target_bytes": 1,
                "sources": [{"name": "x", "bucket": "general", "weight": 1.0}],
            }
        )


def test_raw_hub_jsonl_preserves_heterogeneous_tool_schemas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {"messages": [{"role": "user", "content": "first"}], "tools": [{"a": 1}]},
        {"messages": [{"role": "user", "content": "second"}], "tools": [{"b": []}]},
    ]
    payload = "".join(json.dumps(row) + "\n" for row in rows)
    opened: list[tuple[str, str]] = []

    class FakeHfFileSystem:
        def open(self, path: str, mode: str) -> StringIO:
            opened.append((path, mode))
            return StringIO(payload)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(HfFileSystem=FakeHfFileSystem),
    )
    source = {
        "name": "trajectory",
        "source_type": "hf_raw_jsonl",
        "dataset_id": "owner/repository",
        "revision": "a" * 40,
        "repo_files": ["data/tool_calling.jsonl"],
    }

    assert list(iter_source(source, seed=17)) == rows
    assert opened == [
        (
            f"datasets/owner/repository@{'a' * 40}/data/tool_calling.jsonl",
            "rt",
        )
    ]
