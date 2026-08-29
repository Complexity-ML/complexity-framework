import hashlib
import json
from pathlib import Path

from scripts.tokenize_agentic_pretraining_corpus import (
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


def _write_token_manifests(tmp_path: Path):
    from complexity.training import TextCorpusSource

    sources = [
        TextCorpusSource(name="general", weight=0.6, data_files="general.jsonl"),
        TextCorpusSource(name="agentic", weight=0.4, data_files="agentic.jsonl"),
    ]
    for source, rows in zip(sources, (5, 3), strict=True):
        root = tmp_path / "corpora" / source.name
        root.mkdir(parents=True)
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "trained_tokens": rows * 1_024,
                    "shards": [{"file": "tokens-00000.bin", "rows": rows}],
                }
            ),
            encoding="utf-8",
        )
    return sources, tmp_path
