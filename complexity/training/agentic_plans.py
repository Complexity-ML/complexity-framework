"""Audit local agentic corpora and publish their exact one-pass token plan.

Raw source weights come from the corpus manifest. Token layout and replay
selection use the same native contracts as corpus preparation and training;
these helpers do not load a tokenizer or start a training or upload job.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .agentic_corpus import config_sha256
from .corpus_mixture import (
    PretokenizedCorpusMixtureDataset,
    TextCorpusSource,
    validate_corpus_mixture,
)
from .corpus_preparation import _positive_integer, _sha256_file
from .replay import build_replay_plan

RAW_CORPUS_SCHEMA = "tr-hash-agentic-pretraining-corpus-v1"


def _contained_file(root: Path, relative: str, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} must be a nonempty relative path")
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts:
        raise ValueError(f"{label} must remain inside {root}: {relative}")
    path = root / part
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"{label} must remain inside {root}: {relative}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    return path


def _verify_sha256(path: Path, expected: str, *, label: str) -> None:
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(char not in "0123456789abcdef" for char in expected.lower())
    ):
        raise ValueError(f"{label} must declare a SHA-256 checksum: {path}")
    if _sha256_file(path) != expected.lower():
        raise ValueError(f"{label} checksum mismatch: {path}")


def resolve_bucket_sources(
    corpus_dir: str | Path,
) -> tuple[list[TextCorpusSource], dict[str, list[Path]]]:
    """Group audited raw files by bucket without replacing manifest weights."""
    root = Path(corpus_dir).resolve()
    manifest_path = _contained_file(root, "manifest.json", label="agentic corpus manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != RAW_CORPUS_SCHEMA:
        raise ValueError("unsupported agentic corpus manifest")
    entries = manifest.get("sources")
    if not isinstance(entries, list) or not entries:
        raise ValueError("agentic corpus manifest must contain sources")

    weights: defaultdict[str, float] = defaultdict(float)
    paths: defaultdict[str, list[Path]] = defaultdict(list)
    seen: set[Path] = set()
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("bucket"), str)
            or entry["bucket"] not in {"general", "agentic"}
        ):
            raise ValueError("agentic corpus sources must use general or agentic buckets")
        bucket = entry["bucket"]
        weight = entry.get("weight")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or not 0.0 < weight <= 1.0
        ):
            raise ValueError(f"invalid raw corpus source weight for {bucket}")
        path = _contained_file(root, entry.get("output"), label="raw corpus shard")
        identity = path.resolve()
        if identity in seen:
            raise ValueError(f"raw corpus shard appears more than once: {path}")
        _verify_sha256(path, entry.get("output_sha256"), label="raw corpus shard")
        seen.add(identity)
        weights[bucket] += weight
        paths[bucket].append(path)

    sources = [
        TextCorpusSource(
            name=bucket,
            weight=weights[bucket],
            data_files=[str(path) for path in paths[bucket]],
        )
        for bucket in ("general", "agentic")
        if paths[bucket]
    ]
    validate_corpus_mixture(sources)
    return sources, dict(paths)


def build_pretraining_plan(
    *,
    output_root: str | Path,
    sources: Sequence[TextCorpusSource],
    seq_len: int,
    actual_tokens: int,
    global_batch_sequences: int,
) -> Path:
    """Write an audited plan covering every local token shard exactly once.

    The aggregate rows must cover complete optimizer updates. Individual
    buckets retain their exact integer quotas, including the rounding from
    native weighted allocation; no source rows are dropped for alignment.
    """
    validate_corpus_mixture(sources)
    seq_len = _positive_integer(seq_len, "seq_len")
    actual_tokens = _positive_integer(actual_tokens, "actual_tokens")
    global_batch_sequences = _positive_integer(global_batch_sequences, "global_batch_sequences")
    root = Path(output_root).resolve()
    mixture_path = _contained_file(root, "mixture_manifest.json", label="token mixture manifest")
    mixture = json.loads(mixture_path.read_text(encoding="utf-8"))
    if not isinstance(mixture, dict):
        raise ValueError("token mixture manifest must contain a JSON object")
    entries = mixture.get("sources")
    if not isinstance(entries, list) or not entries:
        raise ValueError("token mixture manifest must contain sources")
    expected_sources = {source.name: source for source in sources}
    if (
        any(
            not isinstance(entry, dict) or not isinstance(entry.get("name"), str)
            for entry in entries
        )
        or len(entries) != len(expected_sources)
        or {entry.get("name") for entry in entries} != expected_sources.keys()
    ):
        raise ValueError("token mixture sources differ from the audited source selection")

    manifest_hashes = {}
    token_paths: set[Path] = set()
    for entry in entries:
        name = entry["name"]
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("corpus source names must be single safe path components")
        expected_relative = str(Path("corpora") / name / "manifest.json")
        if entry.get("manifest") != expected_relative:
            raise ValueError(f"token mixture source manifest path mismatch for {name}")
        path = _contained_file(root, expected_relative, label="token source manifest")
        source_manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(source_manifest, dict) or source_manifest.get("source") != name:
            raise ValueError(f"token source manifest identity mismatch for {name}")
        weight = entry.get("weight")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isclose(weight, expected_sources[name].weight, rel_tol=0, abs_tol=1e-9)
            or source_manifest.get("weight") != weight
        ):
            raise ValueError(f"token mixture source weight mismatch for {name}")
        shards = source_manifest.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError(f"token source {name} must contain shards")
        for shard in shards:
            if not isinstance(shard, dict):
                raise ValueError(f"invalid token shard entry for {name}")
            token_path = _contained_file(path.parent, shard.get("file"), label="token shard")
            identity = token_path.resolve()
            if identity in token_paths:
                raise ValueError(f"token shard appears more than once: {token_path}")
            _verify_sha256(token_path, shard.get("sha256"), label="token shard")
            token_paths.add(identity)
        manifest_hashes[name] = _sha256_file(path)

    dataset = PretokenizedCorpusMixtureDataset(root)
    if dataset.seq_len != seq_len:
        raise ValueError("pretraining plan sequence length differs from the token corpus")
    source_tokens = {}
    for entry in entries:
        name = entry["name"]
        manifest = dataset._source_manifests[name]
        tokens = _positive_integer(manifest.get("trained_tokens"), f"trained_tokens for {name}")
        if (
            tokens != dataset._rows_by_source[name] * seq_len
            or entry.get("trained_tokens") != tokens
        ):
            raise ValueError(f"token source trained_tokens coverage mismatch for {name}")
        source_tokens[name] = tokens
    if (
        sum(source_tokens.values()) != actual_tokens
        or mixture.get("actual_tokens") != actual_tokens
        or actual_tokens % (seq_len * global_batch_sequences)
        or mixture.get("global_batch_sequences") != global_batch_sequences
    ):
        raise ValueError("pretraining plan token budget or batch alignment differs from the corpus")

    plan = build_replay_plan(
        dataset,
        unique_token_budgets=source_tokens,
        replay_passes={source.name: 1 for source in sources},
        row_alignment=1,
    )
    plan.update(
        dataset=str(root),
        revision="local-audited",
        row_alignment=global_batch_sequences,
        mixture_manifest_sha256=_sha256_file(mixture_path),
        source_manifest_sha256=manifest_hashes,
    )
    path = root / "pretrain_plan.json"
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def allocate_phase_rows(
    *,
    config: Mapping[str, Any],
    curriculum: Mapping[str, Any],
    rows_by_source: Mapping[str, int],
) -> dict[str, dict[str, int]]:
    """Allocate aligned source rows to phases without replay or shard overlap."""

    validate_curriculum(config, curriculum)
    alignment = _positive_integer(config["global_batch_sequences"], "global_batch_sequences")
    names = {source["name"] for source in config["sources"]}
    if set(rows_by_source) != names or any(
        type(rows) is not int or rows < 0 or rows % alignment for rows in rows_by_source.values()
    ):
        raise ValueError("curriculum source rows must be exact nonnegative aligned quotas")
    phases = list(curriculum["phases"])
    result = {str(phase["name"]): {} for phase in phases}
    sources_by_bucket: dict[str, list[Mapping[str, Any]]] = {}
    for source in config["sources"]:
        sources_by_bucket.setdefault(str(source["bucket"]), []).append(source)

    for bucket, sources in sources_by_bucket.items():
        source_units = {
            str(source["name"]): int(rows_by_source[str(source["name"])]) // alignment
            for source in sources
        }
        total_units = sum(source_units.values())
        requested_bucket = int(config["bucket_targets"][bucket])
        phase_exact = [
            total_units * int(phase["bucket_tokens"][bucket]) / requested_bucket for phase in phases
        ]
        phase_units = [math.floor(value) for value in phase_exact]
        remainder = total_units - sum(phase_units)
        order = sorted(
            range(len(phases)),
            key=lambda index: phase_exact[index] - phase_units[index],
            reverse=True,
        )
        for index in order[:remainder]:
            phase_units[index] += 1

        remaining = dict(source_units)
        for phase_index, phase in enumerate(phases):
            phase_name = str(phase["name"])
            if phase_index == len(phases) - 1:
                allocation = remaining
            else:
                share = int(phase["bucket_tokens"][bucket]) / requested_bucket
                exact = {name: source_units[name] * share for name in source_units}
                allocation = {
                    name: min(remaining[name], math.floor(value)) for name, value in exact.items()
                }
                needed = phase_units[phase_index] - sum(allocation.values())
                candidates = sorted(
                    source_units,
                    key=lambda name: (exact[name] - allocation[name], remaining[name]),
                    reverse=True,
                )
                for name in candidates:
                    if needed == 0:
                        break
                    if allocation[name] < remaining[name]:
                        allocation[name] += 1
                        needed -= 1
                if needed:
                    raise ValueError(
                        f"cannot align curriculum allocation for {bucket}/{phase_name}"
                    )
                remaining = {name: remaining[name] - allocation[name] for name in remaining}
            for name, units in allocation.items():
                if units:
                    result[phase_name][name] = units * alignment

    for source_name, rows in rows_by_source.items():
        if sum(phase.get(source_name, 0) for phase in result.values()) != rows:
            raise ValueError(f"curriculum does not consume source exactly once: {source_name}")
    return result


def validate_curriculum(config: Mapping[str, Any], curriculum: Mapping[str, Any]) -> None:
    if not isinstance(curriculum, Mapping):
        raise ValueError("curriculum must contain an object")
    phases = list(curriculum.get("phases", ()))
    if not phases:
        raise ValueError("curriculum must contain phases")
    _positive_integer(curriculum.get("total_tokens"), "curriculum total_tokens")
    for phase in phases:
        if not isinstance(phase, Mapping):
            raise ValueError("curriculum phases must contain objects")
        _positive_integer(phase.get("target_tokens"), "phase target_tokens")
        if not isinstance(phase.get("name"), str):
            raise ValueError("curriculum phase names must be strings")
        if not isinstance(phase.get("bucket_tokens"), Mapping) or not isinstance(
            phase.get("bucket_shares"), Mapping
        ):
            raise ValueError("curriculum phases require bucket_tokens and bucket_shares objects")
        if any(type(value) is not int or value < 0 for value in phase["bucket_tokens"].values()):
            raise ValueError("phase bucket tokens must be nonnegative integers")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
            for value in phase["bucket_shares"].values()
        ):
            raise ValueError("phase bucket shares must be finite fractions")
    if int(curriculum.get("total_tokens", -1)) != int(config["target_tokens"]):
        raise ValueError("curriculum total does not match corpus target")
    if sum(int(phase["target_tokens"]) for phase in phases) != int(config["target_tokens"]):
        raise ValueError("phase targets do not sum to corpus target")
    aggregate: Counter[str] = Counter()
    names: set[str] = set()
    for phase in phases:
        name = str(phase.get("name", ""))
        if not name or name in names:
            raise ValueError("curriculum phase names must be non-empty and unique")
        names.add(name)
        bucket_tokens = {str(key): int(value) for key, value in phase["bucket_tokens"].items()}
        if sum(bucket_tokens.values()) != int(phase["target_tokens"]):
            raise ValueError(f"bucket tokens do not sum within phase {name}")
        shares = {str(key): float(value) for key, value in phase["bucket_shares"].items()}
        if set(shares) != set(bucket_tokens) or abs(sum(shares.values()) - 1.0) > 1e-9:
            raise ValueError(f"invalid bucket shares in phase {name}")
        for bucket, tokens in bucket_tokens.items():
            if abs(shares[bucket] - tokens / int(phase["target_tokens"])) > 1e-12:
                raise ValueError(f"bucket share disagrees with tokens in phase {name}")
            aggregate[bucket] += tokens
    expected = {str(key): int(value) for key, value in config["bucket_targets"].items()}
    if dict(aggregate) != expected:
        raise ValueError(f"curriculum bucket totals disagree with corpus: {dict(aggregate)}")
    invariants = curriculum.get("invariants", {})
    if (
        invariants.get("replay") is not False
        or invariants.get("each_packed_row_consumed_once") is not True
    ):
        raise ValueError("curriculum must be no-replay and consume every row once")


def mixture_manifest(
    *,
    config: Mapping[str, Any],
    rows_by_source: Mapping[str, int],
    actual_tokens: int,
    tokenizer_sha256: str,
) -> dict[str, Any]:
    seq_len = int(config["seq_len"])
    direct_materialization = bool(config.get("direct_materialization", False))
    return {
        "format": "tr-hash-token-mixture-v1",
        "schema": config.get("schema", "complexity-token-production-v1"),
        "dtype": "uint16",
        "seq_len": seq_len,
        "requested_tokens": int(config["target_tokens"]),
        "actual_tokens": actual_tokens,
        "global_batch_sequences": int(config["global_batch_sequences"]),
        "bucket_targets": config.get("bucket_targets"),
        "config_sha256": config_sha256(config),
        "tokenizer_sha256": tokenizer_sha256,
        "candidate_manifest_sha256": config.get("candidate_manifest_sha256"),
        "materialization_mode": (
            "direct_source_curated" if direct_materialization else "filtered_deduplicated"
        ),
        "exact_document_deduplication": not direct_materialization,
        "benchmark_decontamination": not direct_materialization,
        "agentic_signal_filtering": not direct_materialization,
        "sources": [
            {
                "name": source["name"],
                "bucket": source["bucket"],
                "weight": source["weight"],
                "rows": rows_by_source[str(source["name"])],
                "trained_tokens": rows_by_source[str(source["name"])] * seq_len,
                "manifest": f"corpora/{source['name']}/manifest.json",
            }
            for source in config["sources"]
        ],
    }


def curriculum_plan(
    store: Any,
    *,
    config: Mapping[str, Any],
    curriculum: Mapping[str, Any],
    rows_by_source: Mapping[str, int],
    actual_tokens: int,
) -> dict[str, Any]:
    if sum(rows_by_source.values()) * config["seq_len"] != actual_tokens:
        raise ValueError("runtime plan actual_tokens differs from its exact source rows")
    phase_rows = allocate_phase_rows(
        config=config,
        curriculum=curriculum,
        rows_by_source=rows_by_source,
    )
    cursors = {str(source["name"]): 0 for source in config["sources"]}
    source_shards = {
        str(source["name"]): store.shards(str(source["name"])) for source in config["sources"]
    }
    for source_name, shards in source_shards.items():
        files = [Path(str(shard["repo_path"])).name for shard in shards]
        if (
            len(files) != len(set(files))
            or sum(shard["rows"] for shard in shards) != rows_by_source[source_name]
        ):
            raise ValueError(f"runtime shard coverage mismatch for {source_name}")
    phases = []
    for phase in curriculum["phases"]:
        name = str(phase["name"])
        selections: dict[str, list[dict[str, Any]]] = {}
        for source_name, wanted_rows in phase_rows[name].items():
            consumed = 0
            selected = []
            shards = source_shards[source_name]
            while consumed < wanted_rows:
                cursor = cursors[source_name]
                if cursor >= len(shards):
                    raise ValueError(f"source {source_name} exhausted while creating phase {name}")
                shard = shards[cursor]
                rows = int(shard["rows"])
                if consumed + rows > wanted_rows:
                    raise ValueError(
                        f"shard crosses curriculum boundary: {source_name}/{shard['repo_path']}"
                    )
                selected.append({"file": Path(str(shard["repo_path"])).name, "rows": rows})
                consumed += rows
                cursors[source_name] += 1
            selections[source_name] = selected
        phases.append({"name": name, "passes": 1, "sources": selections})
    if any(cursors[name] != len(shards) for name, shards in source_shards.items()):
        raise ValueError("curriculum replay plan does not consume every published shard")
    return {
        "format": "tr-hash-token-replay-plan-v1",
        "schema": curriculum.get("schema"),
        "config_sha256": config_sha256(config),
        "curriculum_sha256": config_sha256(curriculum),
        "seq_len": int(config["seq_len"]),
        "selection_mode": "manifest_order",
        "row_alignment": int(config["global_batch_sequences"]),
        "unique_tokens": actual_tokens,
        "trained_tokens": actual_tokens,
        "source_unique_tokens": {
            name: rows * int(config["seq_len"]) for name, rows in rows_by_source.items()
        },
        "source_passes": {name: 1 for name in rows_by_source},
        "phases": phases,
    }


def phase_boundaries(
    *,
    config: Mapping[str, Any],
    curriculum: Mapping[str, Any] | None,
    rows_by_source: Mapping[str, int],
) -> dict[str, tuple[int, ...]]:
    if curriculum is None:
        return {name: (rows,) for name, rows in rows_by_source.items()}
    phase_rows = allocate_phase_rows(
        config=config,
        curriculum=curriculum,
        rows_by_source=rows_by_source,
    )
    result: dict[str, tuple[int, ...]] = {}
    for source_name, total_rows in rows_by_source.items():
        cumulative = 0
        boundaries = []
        for phase in curriculum["phases"]:
            cumulative += phase_rows[str(phase["name"])].get(source_name, 0)
            if cumulative:
                boundaries.append(cumulative)
        if not boundaries or boundaries[-1] != total_rows:
            raise ValueError(f"invalid curriculum boundary coverage for {source_name}")
        result[source_name] = tuple(dict.fromkeys(boundaries))
    return result


def audit_release_documents(
    *,
    config: Mapping[str, Any],
    curriculum: Mapping[str, Any],
    mixture: Mapping[str, Any],
    plan: Mapping[str, Any],
    state: Mapping[str, Any],
    source_manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Cross-check supplied release metadata; this does not read remote bytes."""
    from .agentic_sharding import validate_config

    validate_config(config)
    validate_curriculum(config, curriculum)
    config_digest = config_sha256(config)
    curriculum_digest = config_sha256(curriculum)
    if mixture.get("config_sha256") != config_digest or plan.get("config_sha256") != config_digest:
        raise ValueError("config SHA-256 disagrees with the published runtime metadata")
    if plan.get("curriculum_sha256") != curriculum_digest:
        raise ValueError("curriculum SHA-256 disagrees with the runtime plan")
    if state.get("config_sha256") != config_digest:
        raise ValueError("build state config SHA-256 disagrees with the published config")
    if state.get("exact_document_deduplication") is not True:
        raise ValueError("build state does not prove global exact-document deduplication")
    protected_count = int(state.get("protected_prompt_count", 0))
    if config.get("protected_benchmark_sources") and protected_count <= 0:
        raise ValueError("build state does not prove a populated benchmark decontamination index")
    protected_digest = str(state.get("protected_index_sha256", ""))
    if len(protected_digest) != 64:
        raise ValueError("build state has no valid protected-index SHA-256")

    actual_tokens = int(mixture["actual_tokens"])
    if int(plan["unique_tokens"]) != actual_tokens or int(plan["trained_tokens"]) != actual_tokens:
        raise ValueError("the runtime plan replays or omits packed tokens")
    contract = config["tokenizer_contract"]
    if contract.get("status") != "validated":
        raise ValueError("published corpus tokenizer contract is not validated")
    if mixture.get("tokenizer_sha256") != contract.get("tokenizer_sha256"):
        raise ValueError("mixture tokenizer SHA-256 disagrees with the pinned tokenizer")
    if state.get("tokenizer_sha256") != mixture.get("tokenizer_sha256"):
        raise ValueError("build state tokenizer SHA-256 disagrees with the mixture")

    expected_sources = {str(source["name"]): source for source in mixture["sources"]}
    configured = {source["name"]: source for source in config["sources"]}
    if len(expected_sources) != len(mixture["sources"]) or set(expected_sources) != set(configured):
        raise ValueError("mixture source set disagrees with the configured corpus")
    if set(source_manifests) != set(expected_sources):
        raise ValueError("source manifest set disagrees with the mixture manifest")
    if set(plan.get("source_passes", {})) != set(expected_sources) or any(
        int(value) != 1 for value in plan["source_passes"].values()
    ):
        raise ValueError("every source must have exactly one training pass")
    state_sources = {str(source["name"]): source for source in state.get("sources", ())}
    if set(state_sources) != set(expected_sources):
        raise ValueError("build state source set disagrees with the mixture manifest")

    expected_shards: dict[str, dict[str, Any]] = {}
    source_rows: dict[str, int] = {}
    for source_name, entry in expected_sources.items():
        manifest = source_manifests[source_name]
        if manifest.get("source") != source_name:
            raise ValueError(f"source manifest name mismatch: {source_name}")
        if int(manifest["seq_len"]) != int(mixture["seq_len"]) or manifest["dtype"] != "uint16":
            raise ValueError(f"source layout mismatch: {source_name}")
        rows = sum(int(shard["rows"]) for shard in manifest["shards"])
        if rows != int(manifest["rows"]) or rows != int(entry["rows"]):
            raise ValueError(f"source row mismatch: {source_name}")
        if int(manifest["trained_tokens"]) != rows * int(mixture["seq_len"]):
            raise ValueError(f"source token mismatch: {source_name}")
        source_rows[source_name] = rows
        if int(state_sources[source_name]["rows_done"]) != rows:
            raise ValueError(f"build state is incomplete for source: {source_name}")
        manifest_parent = PurePosixPath(str(entry["manifest"])).parent
        if str(entry["manifest"]) != f"corpora/{source_name}/manifest.json":
            raise ValueError(f"source manifest path mismatch: {source_name}")
        for shard in manifest["shards"]:
            if (
                not isinstance(shard.get("file"), str)
                or PurePosixPath(shard["file"]).name != shard["file"]
                or shard["file"] in {".", ".."}
            ):
                raise ValueError("token shard files must be safe relative basenames")
            if (
                type(shard.get("rows")) is not int
                or shard["rows"] <= 0
                or shard.get("tokens") != shard["rows"] * mixture["seq_len"] + 1
            ):
                raise ValueError("token shard layout does not cover exact shared-boundary rows")
            digest = shard.get("sha256")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise ValueError("token shard must declare a valid SHA-256")
            relative = str(manifest_parent / str(shard["file"]))
            if relative in expected_shards:
                raise ValueError(f"duplicate shard path: {relative}")
            if int(shard["bytes"]) != int(shard["tokens"]) * 2:
                raise ValueError(f"uint16 byte count mismatch: {relative}")
            expected_shards[relative] = {
                "bytes": int(shard["bytes"]),
                "sha256": str(shard["sha256"]),
                "source": source_name,
                "file": str(shard["file"]),
                "rows": int(shard["rows"]),
            }
        recorded = {
            Path(item["repo_path"]).name: item
            for item in state_sources[source_name].get("shards", ())
        }
        if set(recorded) != {shard["file"] for shard in manifest["shards"]} or any(
            any(
                recorded[shard["file"]].get(key) != shard[key]
                for key in ("rows", "tokens", "bytes", "sha256")
            )
            for shard in manifest["shards"]
        ):
            raise ValueError(
                f"source manifest differs from acknowledged shard state: {source_name}"
            )

    if sum(int(entry["trained_tokens"]) for entry in expected_sources.values()) != actual_tokens:
        raise ValueError("source tokens do not sum to actual_tokens")

    planned: list[tuple[str, str]] = []
    planned_rows = {name: 0 for name in expected_sources}
    phase_rows = allocate_phase_rows(
        config=config,
        curriculum=curriculum,
        rows_by_source=source_rows,
    )
    if [phase.get("name") for phase in plan["phases"]] != [
        phase["name"] for phase in curriculum["phases"]
    ]:
        raise ValueError("runtime phases differ from the audited curriculum")
    by_pair = {(item["source"], item["file"]): item for item in expected_shards.values()}
    for phase in plan["phases"]:
        if int(phase.get("passes", 0)) != 1:
            raise ValueError(f"phase {phase.get('name')} does not use one pass")
        for source_name, shards in phase["sources"].items():
            if source_name not in expected_sources:
                raise ValueError(f"unknown source in runtime plan: {source_name}")
            for shard in shards:
                expected = by_pair.get((source_name, str(shard["file"])))
                if expected is None or shard.get("rows") != expected["rows"]:
                    raise ValueError("planned shard rows differ from acknowledged source metadata")
                planned.append((source_name, str(shard["file"])))
                planned_rows[source_name] += int(shard["rows"])
        actual_phase_rows = {
            source: sum(shard["rows"] for shard in shards)
            for source, shards in phase["sources"].items()
        }
        if actual_phase_rows != phase_rows[phase["name"]]:
            raise ValueError("runtime phase rows differ from the no-replay curriculum allocation")
    expected_pairs = {
        (metadata["source"], metadata["file"]) for metadata in expected_shards.values()
    }
    if len(planned) != len(set(planned)) or set(planned) != expected_pairs:
        raise ValueError("runtime plan does not consume every shard exactly once")
    if planned_rows != source_rows:
        raise ValueError("runtime plan row totals disagree with source manifests")

    return {
        "requested_tokens": int(mixture["requested_tokens"]),
        "actual_tokens": actual_tokens,
        "alignment_overhead_tokens": actual_tokens - int(mixture["requested_tokens"]),
        "source_count": len(expected_sources),
        "shard_count": len(expected_shards),
        "tokenizer_sha256": mixture["tokenizer_sha256"],
        "config_sha256": config_digest,
        "curriculum_sha256": curriculum_digest,
        "protected_prompt_count": protected_count,
        "protected_index_sha256": protected_digest,
        "expected_shards": expected_shards,
    }
