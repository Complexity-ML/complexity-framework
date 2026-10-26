import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from complexity.training.agentic_plans import (
    build_pretraining_plan,
    resolve_bucket_sources,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bucket_resolution_preserves_60_40_mixture_and_checksums(tmp_path: Path) -> None:
    outputs = []
    for index, (name, bucket, weight) in enumerate(
        (
            ("general_a", "general", 0.35),
            ("general_b", "general", 0.25),
            ("agentic_a", "agentic", 0.4),
        )
    ):
        path = tmp_path / f"{index:02d}-{name}.jsonl"
        path.write_text(json.dumps({"text": f"document {name}"}) + "\n", encoding="utf-8")
        outputs.append(
            {
                "name": name,
                "bucket": bucket,
                "weight": weight,
                "output": path.name,
                "output_sha256": _sha256(path),
            }
        )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "tr-hash-agentic-pretraining-corpus-v1",
                "sources": outputs,
            }
        ),
        encoding="utf-8",
    )

    sources, paths = resolve_bucket_sources(tmp_path)

    assert [(source.name, source.weight) for source in sources] == [
        ("general", 0.6),
        ("agentic", 0.4),
    ]
    assert len(paths["general"]) == 2
    assert len(paths["agentic"]) == 1


def test_pretraining_plan_is_one_pass_unique_core(tmp_path: Path) -> None:
    from complexity.training.corpus_mixture import PretokenizedCorpusMixtureDataset

    sources, _ = _write_token_manifests(tmp_path)

    path = build_pretraining_plan(
        output_root=tmp_path,
        sources=sources,
        seq_len=1_024,
        actual_tokens=8_192,
        global_batch_sequences=8,
    )
    plan = json.loads(path.read_text(encoding="utf-8"))

    assert plan["unique_tokens"] == plan["trained_tokens"] == 8_192
    assert plan["row_alignment"] == 8
    assert plan["source_passes"] == {"general": 1, "agentic": 1}
    assert [phase["name"] for phase in plan["phases"]] == ["unique_core"]
    assert plan["phases"][0]["passes"] == 1
    assert plan["phases"][0]["sources"] == {
        "general": [{"file": "tokens-00000.bin", "rows": 5}],
        "agentic": [{"file": "tokens-00000.bin", "rows": 3}],
    }
    assert plan["source_unique_tokens"] == {"general": 5_120, "agentic": 3_072}
    assert plan["dataset"] == str(tmp_path.resolve())
    assert plan["revision"] == "local-audited"
    assert plan["mixture_manifest_sha256"] == _sha256(tmp_path / "mixture_manifest.json")
    assert plan["source_manifest_sha256"] == {
        source.name: _sha256(tmp_path / "corpora" / source.name / "manifest.json")
        for source in sources
    }
    dataset = PretokenizedCorpusMixtureDataset(tmp_path, replay_plan=path)
    assert dataset.unique_tokens == dataset.trained_tokens == 8_192
    assert len(list(dataset)) == 8


def _write_raw_manifest(root: Path, output: str, sha256: str) -> None:
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "tr-hash-agentic-pretraining-corpus-v1",
                "sources": [
                    {
                        "name": "general_a",
                        "bucket": "general",
                        "weight": 1.0,
                        "output": output,
                        "output_sha256": sha256,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_raw_bucket_resolution_rejects_checksum_mismatch(tmp_path: Path) -> None:
    shard = tmp_path / "source.jsonl"
    shard.write_text('{"text": "original"}\n', encoding="utf-8")
    _write_raw_manifest(tmp_path, shard.name, _sha256(shard))
    shard.write_text('{"text": "changed"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        resolve_bucket_sources(tmp_path)


@pytest.mark.parametrize("escape", ["parent", "absolute", "symlink"])
def test_raw_bucket_resolution_rejects_paths_outside_corpus(tmp_path: Path, escape: str) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"text": "outside"}\n', encoding="utf-8")
    if escape == "parent":
        output = "../outside.jsonl"
    elif escape == "absolute":
        output = str(outside)
    else:
        (root / "linked.jsonl").symlink_to(outside)
        output = "linked.jsonl"
    _write_raw_manifest(root, output, _sha256(outside))

    with pytest.raises(ValueError, match="must remain inside"):
        resolve_bucket_sources(root)


def test_raw_bucket_resolution_rejects_duplicate_files(tmp_path: Path) -> None:
    shard = tmp_path / "source.jsonl"
    shard.write_text('{"text": "original"}\n', encoding="utf-8")
    _write_raw_manifest(tmp_path, shard.name, _sha256(shard))
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entry = manifest["sources"][0]
    entry["weight"] = 0.5
    manifest["sources"].append({**entry, "name": "general_b", "output": f"./{shard.name}"})
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="appears more than once"):
        resolve_bucket_sources(tmp_path)


def test_pretraining_plan_rejects_token_checksum_mismatch(tmp_path: Path) -> None:
    sources, _ = _write_token_manifests(tmp_path)
    shard = tmp_path / "corpora" / "general" / "tokens-00000.bin"
    content = bytearray(shard.read_bytes())
    content[0] ^= 1
    shard.write_bytes(content)

    with pytest.raises(ValueError, match="token shard checksum mismatch"):
        _build_plan(tmp_path, sources)
    assert not (tmp_path / "pretrain_plan.json").exists()


def test_pretraining_plan_audits_paths_relative_to_manifest_location(tmp_path: Path) -> None:
    sources, _ = _write_token_manifests(tmp_path)
    root = tmp_path / "corpora" / "general"
    archive = tmp_path / "archive"
    archive.mkdir()
    manifest = root / "manifest.json"
    shard = root / "tokens-00000.bin"
    (archive / manifest.name).write_bytes(manifest.read_bytes())
    (archive / shard.name).write_bytes(shard.read_bytes())
    manifest.unlink()
    manifest.symlink_to(archive / manifest.name)
    content = bytearray(shard.read_bytes())
    content[0] ^= 1
    shard.write_bytes(content)

    with pytest.raises(ValueError, match="token shard checksum mismatch"):
        _build_plan(tmp_path, sources)
    assert not (tmp_path / "pretrain_plan.json").exists()


@pytest.mark.parametrize("escape", ["parent", "symlink"])
def test_pretraining_plan_rejects_token_paths_outside_source(tmp_path: Path, escape: str) -> None:
    sources, _ = _write_token_manifests(tmp_path)
    source_root = tmp_path / "corpora" / "general"
    path = source_root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    outside = tmp_path / "corpora" / "outside.bin"
    original = source_root / "tokens-00000.bin"
    outside.write_bytes(original.read_bytes())
    if escape == "parent":
        manifest["shards"][0]["file"] = "../outside.bin"
    else:
        original.unlink()
        original.symlink_to(outside)
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="must remain inside"):
        _build_plan(tmp_path, sources)
    assert not (tmp_path / "pretrain_plan.json").exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("actual_tokens", 16_384, "token budget or batch alignment"),
        ("global_batch_sequences", 16, "token budget or batch alignment"),
        ("seq_len", 512, "sequence length"),
    ],
)
def test_pretraining_plan_rejects_requested_layout_mismatch(
    tmp_path: Path, field: str, value: int, message: str
) -> None:
    sources, _ = _write_token_manifests(tmp_path)

    with pytest.raises(ValueError, match=message):
        _build_plan(tmp_path, sources, **{field: value})
    assert not (tmp_path / "pretrain_plan.json").exists()


def test_pretraining_plan_rejects_inconsistent_source_token_budget(tmp_path: Path) -> None:
    sources, _ = _write_token_manifests(tmp_path)
    path = tmp_path / "corpora" / "general" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["trained_tokens"] += 1_024
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="trained_tokens coverage mismatch"):
        _build_plan(tmp_path, sources)
    assert not (tmp_path / "pretrain_plan.json").exists()


def _build_plan(tmp_path: Path, sources, **overrides) -> Path:
    settings = {"seq_len": 1_024, "actual_tokens": 8_192, "global_batch_sequences": 8}
    settings.update(overrides)
    return build_pretraining_plan(output_root=tmp_path, sources=sources, **settings)


def _write_token_manifests(tmp_path: Path):
    from complexity.training import TextCorpusSource
    from complexity.training.corpus_preparation import TokenShardWriter, write_mixture_manifest

    sources = [
        TextCorpusSource(name="general", weight=0.6, data_files="general.jsonl"),
        TextCorpusSource(name="agentic", weight=0.4, data_files="agentic.jsonl"),
    ]
    for source, rows in zip(sources, (5, 3), strict=True):
        root = tmp_path / "corpora" / source.name
        writer = TokenShardWriter(root, seq_len=1_024, total_rows=rows, rows_per_shard=rows)
        writer.feed(np.arange(rows * 1_024 + 1, dtype=np.uint16))
        writer.write_manifest(source=source)
    write_mixture_manifest(
        output_root=tmp_path,
        sources=sources,
        seq_len=1_024,
        requested_tokens=8_192,
        actual_tokens=8_192,
        global_batch_sequences=8,
        rows_by_source={"general": 5, "agentic": 3},
    )
    return sources, tmp_path
