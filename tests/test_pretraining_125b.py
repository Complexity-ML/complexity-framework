import hashlib
import json
import queue
import shutil
import threading
from collections import Counter
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from complexity.training import PretokenizedCorpusMixtureDataset
from scripts.audit_tr_hash_pretraining_125b_release import audit_release_documents
from scripts.build_agentic_pretraining_50b import (
    _mixture_manifest,
    _parallel_source_groups,
    _phase_boundaries,
    _prepare_source_batches,
    _replay_plan,
    allocate_rows,
    build,
    make_direct_source_curated_config,
    row_text,
    validate_config,
    validate_curriculum,
    validate_tokenizer_contract,
)
from scripts.pin_tr_hash_pretraining_tokenizer import pin_tokenizer_contract

CONFIG_PATH = Path("configs/agentic_pretraining/tr_hash_pretraining_125b.json")
CURRICULUM_PATH = Path("configs/agentic_pretraining/tr_hash_pretraining_125b_curriculum.json")


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_125b_manifest_has_exact_bucket_and_source_budgets() -> None:
    config = _config()
    validate_config(config)
    assert config["target_tokens"] == 125_000_000_000
    assert config["parallel_sources"] == 3
    assert config["producer_queue_depth"] == 4
    assert config["producer_candidate_batch_size"] == 256
    assert config["candidate_oversample"] == pytest.approx(1.05)
    assert config["candidate_shard_tokens"] == 250_000_000
    assert config["candidate_tokenization_batch_size"] == 512
    by_bucket = Counter()
    for source in config["sources"]:
        by_bucket[source["bucket"]] += source["target_tokens"]
    assert by_bucket == {"foundation": 75_000_000_000, "agentic": 50_000_000_000}
    assert sum(source["weight"] for source in config["sources"]) == pytest.approx(1.0)


def test_direct_125b_config_preserves_budgets_and_discloses_fast_path() -> None:
    config = make_direct_source_curated_config(_config())
    validate_config(config)
    by_bucket = Counter()
    for source in config["sources"]:
        by_bucket[source["bucket"]] += source["target_tokens"]
    assert by_bucket == {"foundation": 75_000_000_000, "agentic": 50_000_000_000}
    assert config["target_tokens"] == 125_000_000_000
    assert config["tokenization_batch_size"] == 4096
    assert config["producer_candidate_batch_size"] == 4096
    assert config["producer_scan_batch_size"] == 4096
    assert config["parallel_sources"] == 3
    assert config["protected_benchmarks"] == []
    assert config["protected_benchmark_sources"] == []
    assert all(source["selection"] == "direct" for source in config["sources"])


def test_direct_selection_requires_explicit_unfiltered_contract() -> None:
    config = make_direct_source_curated_config(_config())
    config["direct_materialization"] = False
    with pytest.raises(ValueError, match="requires direct_materialization=true"):
        validate_config(config)

    config = make_direct_source_curated_config(_config())
    config["protected_benchmarks"] = ["arc_easy"]
    with pytest.raises(ValueError, match="cannot claim benchmark decontamination"):
        validate_config(config)


def test_direct_source_preparation_keeps_raw_duplicates_without_hashing() -> None:
    packets: queue.Queue = queue.Queue()
    stop = threading.Event()
    source = {
        "name": "fixture",
        "bucket": "foundation",
        "selection": "direct",
        "text_field": "text",
        "license_audit": "fixture",
    }

    _prepare_source_batches(
        source=source,
        source_index=0,
        restored_scanned=0,
        config={"producer_candidate_batch_size": 8, "producer_scan_batch_size": 8},
        protected=(),
        benchmark_index=None,
        destination=packets,
        stop=stop,
        source_iterator_factory=lambda _source, _seed: iter(({"text": "same"}, {"text": "same"})),
    )

    packet = packets.get_nowait()
    assert packet.exhausted is True
    assert packet.scanned == 2
    assert packet.candidates == (("same", (), ""), ("same", (), ""))


def test_125b_sources_are_pinned_and_use_the_quality_variants() -> None:
    config = _config()
    sources = {source["name"]: source for source in config["sources"]}
    assert sources["fineweb2_french_foundation"]["config_name"] == "fra_Latn"
    assert sources["fineweb2_french_foundation"]["revision"] == (
        "af9c13333eb981300149d5ca60a8e9d659b276b9"
    )
    assert sources["finemath_4plus_agentic"]["config_name"] == "finemath-4plus"
    assert sources["infiwebmath_4plus_agentic"]["config_name"] == "infiwebmath-4plus"
    assert sources["finemath_3plus_foundation"]["config_name"] == "finemath-3plus"
    assert sources["infiwebmath_3plus_foundation"]["config_name"] == "infiwebmath-3plus"
    assert sources["nemotron_tool_calling"]["target_tokens"] == 500_000_000
    assert sources["nemotron_tool_calling"]["source_type"] == "hf_raw_jsonl"
    assert sources["nemotron_tool_calling"]["repo_files"] == ["data/tool_calling.jsonl"]
    assert all(
        len(source.get("revision", "")) == 40
        for source in config["sources"]
        if "dataset_id" in source
    )


def test_stack_edu_is_permissive_only_and_language_balanced() -> None:
    stack = [source for source in _config()["sources"] if source["name"].startswith("stack_")]
    assert sum(source["target_tokens"] for source in stack) == 10_000_000_000
    assert {source["config_name"] for source in stack} == {
        "Python",
        "JavaScript",
        "TypeScript",
        "Shell",
        "Rust",
        "Go",
        "SQL",
        "Java",
        "Markdown",
    }
    assert all(source["allowed_license_types"] == ["permissive"] for source in stack)
    assert all(source["min_int_score"] >= 3 for source in stack)


def test_agentic_sources_are_consumed_before_overlapping_foundation_sources() -> None:
    sources = _config()["sources"]
    last_agentic = max(
        index for index, source in enumerate(sources) if source["bucket"] == "agentic"
    )
    first_foundation = min(
        index for index, source in enumerate(sources) if source["bucket"] == "foundation"
    )
    assert last_agentic < first_foundation


def test_parallel_groups_diversify_upstreams_without_crossing_buckets() -> None:
    groups = _parallel_source_groups(_config()["sources"], 3)
    assert [source["name"] for source in groups[0]] == [
        "stack_python_agentic",
        "fineweb_edu_agentic",
        "finemath_4plus_agentic",
    ]
    assert all(len({source["bucket"] for source in group}) == 1 for group in groups)
    flattened = [source["name"] for group in groups for source in group]
    assert sorted(flattened) == sorted(source["name"] for source in _config()["sources"])


def test_125b_curriculum_is_no_replay_and_matches_the_corpus() -> None:
    config = _config()
    curriculum = json.loads(CURRICULUM_PATH.read_text(encoding="utf-8"))
    validate_curriculum(config, curriculum)
    assert [phase["target_tokens"] for phase in curriculum["phases"]] == [
        75_000_000_000,
        50_000_000_000,
    ]
    assert curriculum["phases"][0]["bucket_tokens"] == {
        "foundation": 60_000_000_000,
        "agentic": 15_000_000_000,
    }
    assert curriculum["phases"][1]["bucket_tokens"] == {
        "foundation": 15_000_000_000,
        "agentic": 35_000_000_000,
    }
    assert curriculum["invariants"]["replay"] is False


def test_125b_layout_is_global_batch_aligned() -> None:
    config = _config()
    rows, tokens, allocated = allocate_rows(
        target_tokens=config["target_tokens"],
        seq_len=config["seq_len"],
        global_batch_sequences=config["global_batch_sequences"],
        sources=config["sources"],
    )
    assert rows % config["global_batch_sequences"] == 0
    assert tokens >= config["target_tokens"]
    assert tokens - config["target_tokens"] < (config["seq_len"] * config["global_batch_sequences"])
    assert sum(allocated.values()) == rows
    assert all(value % config["global_batch_sequences"] == 0 for value in allocated.values())


def test_curriculum_boundaries_generate_a_no_replay_runtime_plan() -> None:
    config = {
        "target_tokens": 400,
        "bucket_targets": {"foundation": 240, "agentic": 160},
        "seq_len": 4,
        "global_batch_sequences": 2,
        "sources": [
            {
                "name": "foundation",
                "bucket": "foundation",
                "selection": "quality",
                "target_tokens": 240,
                "weight": 0.6,
                "path": "fixture",
                "license_audit": "fixture",
            },
            {
                "name": "agentic",
                "bucket": "agentic",
                "selection": "agentic",
                "target_tokens": 160,
                "weight": 0.4,
                "path": "fixture",
                "license_audit": "fixture",
            },
        ],
    }
    curriculum = {
        "schema": "fixture",
        "total_tokens": 400,
        "phases": [
            {
                "name": "first",
                "target_tokens": 240,
                "bucket_tokens": {"foundation": 192, "agentic": 48},
                "bucket_shares": {"foundation": 0.8, "agentic": 0.2},
            },
            {
                "name": "second",
                "target_tokens": 160,
                "bucket_tokens": {"foundation": 48, "agentic": 112},
                "bucket_shares": {"foundation": 0.3, "agentic": 0.7},
            },
        ],
        "invariants": {"replay": False, "each_packed_row_consumed_once": True},
    }
    validate_config(config)
    validate_curriculum(config, curriculum)
    _, actual_tokens, rows = allocate_rows(
        target_tokens=400,
        seq_len=4,
        global_batch_sequences=2,
        sources=config["sources"],
    )
    boundaries = _phase_boundaries(config=config, curriculum=curriculum, rows_by_source=rows)

    class Store:
        def shards(self, source: str):
            starts = (0, *boundaries[source][:-1])
            return [
                {
                    "repo_path": f"production/corpora/{source}/tokens-{index:05d}.bin",
                    "rows": stop - start,
                }
                for index, (start, stop) in enumerate(zip(starts, boundaries[source], strict=True))
            ]

    plan = _replay_plan(
        Store(),
        config=config,
        curriculum=curriculum,
        rows_by_source=rows,
        actual_tokens=actual_tokens,
    )
    selected = [
        (source, shard["file"])
        for phase in plan["phases"]
        for source, shards in phase["sources"].items()
        for shard in shards
    ]
    assert len(selected) == len(set(selected))
    assert plan["unique_tokens"] == plan["trained_tokens"] == 400
    assert plan["source_passes"] == {"foundation": 1, "agentic": 1}


def test_mixture_manifest_matches_runtime_loader_contract() -> None:
    config = _config()
    _, actual_tokens, rows = allocate_rows(
        target_tokens=config["target_tokens"],
        seq_len=config["seq_len"],
        global_batch_sequences=config["global_batch_sequences"],
        sources=config["sources"],
    )
    manifest = _mixture_manifest(
        config=config,
        rows_by_source=rows,
        actual_tokens=actual_tokens,
        tokenizer_sha256="ab" * 32,
    )
    assert manifest["format"] == "tr-hash-token-mixture-v1"
    assert manifest["dtype"] == "uint16"
    assert manifest["tokenizer_sha256"] == "ab" * 32
    assert manifest["actual_tokens"] == sum(
        source["trained_tokens"] for source in manifest["sources"]
    )
    assert all(
        source["manifest"] == f"corpora/{source['name']}/manifest.json"
        for source in manifest["sources"]
    )

    direct = make_direct_source_curated_config(config)
    direct_manifest = _mixture_manifest(
        config=direct,
        rows_by_source=rows,
        actual_tokens=actual_tokens,
        tokenizer_sha256="cd" * 32,
    )
    assert direct_manifest["materialization_mode"] == "direct_source_curated"
    assert direct_manifest["exact_document_deduplication"] is False
    assert direct_manifest["benchmark_decontamination"] is False
    assert direct_manifest["agentic_signal_filtering"] is False


def test_small_curriculum_build_loads_in_the_runtime_without_replay(tmp_path: Path) -> None:
    vocab = {"<unk>": 0, **{f"token_{index}": index for index in range(1, 32_000)}}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
    tokenizer_manifest = tokenizer_dir / "agentic_tokenizer_manifest.json"
    tokenizer_manifest.write_text('{"vocab_size": 32000}\n', encoding="utf-8")

    sources = []
    for name, bucket in (("agentic", "agentic"), ("foundation", "foundation")):
        path = tmp_path / f"{name}.jsonl"
        path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "text": (
                            f"{'shared' if index == 0 else name} record {index}. "
                            "This verified technical procedure has "
                            "enough unique explanatory material for a tokenizer fixture."
                        )
                    }
                )
                for index in range(100)
            )
            + "\n",
            encoding="utf-8",
        )
        sources.append(
            {
                "name": name,
                "bucket": bucket,
                "selection": "quality",
                "target_tokens": 64,
                "weight": 0.5,
                "path": str(path),
                "text_field": "text",
                "license_audit": "fixture",
            }
        )
    config = {
        "version": 1,
        "target_tokens": 128,
        "bucket_targets": {"foundation": 64, "agentic": 64},
        "seq_len": 4,
        "global_batch_sequences": 2,
        "shard_trained_tokens": 32,
        "tokenization_batch_size": 2,
        "parallel_sources": 2,
        "producer_queue_depth": 1,
        "producer_scan_batch_size": 2,
        "min_chars": 20,
        "max_chars": 10_000,
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
                "name": "first",
                "target_tokens": 64,
                "bucket_tokens": {"foundation": 48, "agentic": 16},
                "bucket_shares": {"foundation": 0.75, "agentic": 0.25},
            },
            {
                "name": "second",
                "target_tokens": 64,
                "bucket_tokens": {"foundation": 16, "agentic": 48},
                "bucket_shares": {"foundation": 0.25, "agentic": 0.75},
            },
        ],
        "invariants": {"replay": False, "each_packed_row_consumed_once": True},
    }

    class Publisher:
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

    hub = tmp_path / "hub"
    build(
        config=config,
        tokenizer_path=tokenizer_dir,
        work_dir=tmp_path / "work",
        publisher=Publisher(hub),
        curriculum=curriculum,
    )
    assert (hub / "_metadata/config.json").is_file()
    assert (hub / "_metadata/curriculum.json").is_file()

    # A completed build remains repairable: rerunning it republishes manifests
    # and the runtime plan without reading or retokenizing source documents.
    (hub / "corpora/agentic/manifest.json").unlink()
    (hub / "mixture_manifest.json").unlink()
    (hub / "pretrain_plan.json").unlink()
    build(
        config=config,
        tokenizer_path=tokenizer_dir,
        work_dir=tmp_path / "work",
        publisher=Publisher(hub),
        curriculum=curriculum,
    )
    assert (hub / "corpora/agentic/manifest.json").is_file()
    assert (hub / "mixture_manifest.json").is_file()
    assert (hub / "pretrain_plan.json").is_file()

    dataset = PretokenizedCorpusMixtureDataset(
        hub,
        replay_plan=hub / "pretrain_plan.json",
    )
    samples = list(dataset)
    assert len(samples) == 32
    assert dataset.unique_tokens == dataset.trained_tokens == 128
    completed_state = json.loads((hub / "_state/state.json").read_text(encoding="utf-8"))
    by_source = {source["name"]: source for source in completed_state["sources"]}
    assert by_source["foundation"]["counters"]["exact_duplicate"] >= 1

    report = audit_release_documents(
        config=json.loads((hub / "_metadata/config.json").read_text(encoding="utf-8")),
        curriculum=json.loads((hub / "_metadata/curriculum.json").read_text(encoding="utf-8")),
        mixture=json.loads((hub / "mixture_manifest.json").read_text(encoding="utf-8")),
        plan=json.loads((hub / "pretrain_plan.json").read_text(encoding="utf-8")),
        state=completed_state,
        source_manifests={
            name: json.loads((hub / f"corpora/{name}/manifest.json").read_text(encoding="utf-8"))
            for name in ("agentic", "foundation")
        },
    )
    assert report["actual_tokens"] == 128
    assert report["source_count"] == 2
    assert report["shard_count"] == 6

    # Parallel producers may finish at different times, but the fixed
    # round-robin merger must produce byte-identical shards on a clean rebuild.
    second_hub = tmp_path / "hub-second"
    build(
        config=config,
        tokenizer_path=tokenizer_dir,
        work_dir=tmp_path / "work-second",
        publisher=Publisher(second_hub),
        curriculum=curriculum,
    )
    first_shards = {path.relative_to(hub): path.read_bytes() for path in hub.rglob("tokens-*.bin")}
    second_shards = {
        path.relative_to(second_hub): path.read_bytes() for path in second_hub.rglob("tokens-*.bin")
    }
    assert second_shards == first_shards


def test_tool_trajectory_serialization_uses_native_32k_markers() -> None:
    source = {"messages_field": "messages", "tools_field": "tools"}
    text = row_text(
        {
            "tools": [{"type": "function", "function": {"name": "search"}}],
            "messages": [
                {"role": "user", "content": "Find it"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"name": "search", "arguments": {"q": "it"}}],
                },
                {"role": "tool", "content": "found"},
                {"role": "assistant", "content": "Done"},
            ],
        },
        source,
    )
    assert "<|system|>Available tools:" in text
    assert "<|user|>Find it<|end_of_turn|>" in text
    assert "<|tool_call_start|>" in text
    assert "<|tool_result_start|>found<|tool_result_end|>" in text
    assert text.endswith("<|assistant|>Done<|end_of_turn|>")


def test_125b_build_is_gated_until_the_tokenizer_manifest_is_pinned(tmp_path: Path) -> None:
    config = _config()
    config["tokenizer_contract"] = {
        "repo_id": "AETHORIA-AI/TR-HASH-Tokenizer-32K-Agentic",
        "vocab_size": 32_000,
        "required_manifest": "agentic_tokenizer_manifest.json",
        "status": "pending-validation",
    }
    with pytest.raises(ValueError, match="gated"):
        validate_tokenizer_contract(config, tmp_path)

    manifest = tmp_path / "agentic_tokenizer_manifest.json"
    manifest.write_text('{"vocab_size": 32000}\n', encoding="utf-8")
    config["tokenizer_contract"].update(
        {
            "status": "validated",
            "revision": "a" * 40,
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "tokenizer_sha256": hashlib.sha256(
                (tmp_path / "tokenizer.json").read_bytes()
            ).hexdigest()
            if (tmp_path / "tokenizer.json").is_file()
            else "0" * 64,
        }
    )
    with pytest.raises(ValueError, match="tokenizer.json"):
        validate_tokenizer_contract(config, tmp_path)


def test_pin_tokenizer_contract_validates_vocab_markers_and_hashes(tmp_path: Path) -> None:
    from scripts.train_tr_hash_agentic_tokenizer import train_tokenizer_from_iterator

    tokenizer_dir = tmp_path / "tokenizer"
    train_tokenizer_from_iterator(
        iter(
            [
                "Plan the operation, call the tool, inspect the result, and verify the answer. "
                f"Unique procedure {index}."
                for index in range(300)
            ]
        ),
        tokenizer_dir,
        vocab_size=384,
        min_frequency=1,
    )
    config = _config()
    config["tokenizer_contract"]["vocab_size"] = 384
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    contract = pin_tokenizer_contract(config_path, tokenizer_dir, "a" * 40)
    pinned = json.loads(config_path.read_text(encoding="utf-8"))["tokenizer_contract"]
    assert contract == pinned
    assert pinned["status"] == "validated"
    assert (
        pinned["manifest_sha256"]
        == hashlib.sha256(
            (tokenizer_dir / "agentic_tokenizer_manifest.json").read_bytes()
        ).hexdigest()
    )
    assert (
        pinned["tokenizer_sha256"]
        == hashlib.sha256((tokenizer_dir / "tokenizer.json").read_bytes()).hexdigest()
    )


def test_protected_benchmark_contract_covers_public_evals() -> None:
    config = _config()
    protected = set(config["protected_benchmarks"])
    assert {"arc_easy", "arc_challenge", "piqa", "gsm8k", "hellaswag"} <= protected
    indexed = {source["name"] for source in config["protected_benchmark_sources"]}
    assert {
        "arc_easy",
        "arc_challenge",
        "piqa",
        "gsm8k",
        "hellaswag",
        "mmlu",
        "truthfulqa",
        "winogrande",
    } == indexed
