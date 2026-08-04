"""Deterministic, runtime-only curriculum selection for SFT shards.

The curriculum never rewrites or republishes a dataset.  It selects examples
from the existing ``examples.jsonl`` index when a stage starts.  Optional row
metadata (difficulty and interaction mode) is read from the projected Parquet
stored beside the token shards.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"curriculum {field_name} must be a list of strings")
    return tuple(value)


@dataclass(frozen=True)
class CurriculumFilters:
    tasks: tuple[str, ...] = ()
    exclude_tasks: tuple[str, ...] = ()
    difficulties: tuple[str, ...] = ()
    modes: tuple[str, ...] = ()
    training_representations: tuple[str, ...] = ()
    max_num_tokens: int | None = None
    max_supervised_tokens: int | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "CurriculumFilters":
        raw = raw or {}
        supported = {
            "tasks",
            "exclude_tasks",
            "difficulties",
            "modes",
            "training_representations",
            "max_num_tokens",
            "max_supervised_tokens",
        }
        unknown = set(raw).difference(supported)
        if unknown:
            raise ValueError(f"unsupported curriculum filters: {sorted(unknown)}")
        max_num_tokens = raw.get("max_num_tokens")
        max_supervised_tokens = raw.get("max_supervised_tokens")
        for name, value in (
            ("max_num_tokens", max_num_tokens),
            ("max_supervised_tokens", max_supervised_tokens),
        ):
            if value is not None and (not isinstance(value, int) or value < 1):
                raise ValueError(f"curriculum {name} must be a positive integer")
        return cls(
            tasks=_string_tuple(raw.get("tasks"), "filters.tasks"),
            exclude_tasks=_string_tuple(
                raw.get("exclude_tasks"), "filters.exclude_tasks"
            ),
            difficulties=_string_tuple(
                raw.get("difficulties"), "filters.difficulties"
            ),
            modes=_string_tuple(raw.get("modes"), "filters.modes"),
            training_representations=_string_tuple(
                raw.get("training_representations"),
                "filters.training_representations",
            ),
            max_num_tokens=max_num_tokens,
            max_supervised_tokens=max_supervised_tokens,
        )


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    max_examples: int | None
    epochs: int
    lr: float
    include_previous: bool = True
    balance_by: str = "task"
    filters: CurriculumFilters = field(default_factory=CurriculumFilters)
    batch_size: int | None = None
    seq_len: int | None = None
    eval_steps: int | None = None
    save_steps: int | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CurriculumStage":
        name = str(raw.get("name", "")).strip()
        if not name:
            raise ValueError("every curriculum stage needs a non-empty name")
        max_examples = raw.get("max_examples")
        if max_examples in ("all", None):
            max_examples = None
        elif not isinstance(max_examples, int) or max_examples < 1:
            raise ValueError(f"stage {name}: max_examples must be positive or 'all'")
        epochs = raw.get("epochs", 1)
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError(f"stage {name}: epochs must be a positive integer")
        lr = raw.get("lr")
        if not isinstance(lr, (int, float)) or lr <= 0:
            raise ValueError(f"stage {name}: lr must be positive")
        balance_by = str(raw.get("balance_by", "task"))
        if balance_by not in {"task", "none"}:
            raise ValueError(f"stage {name}: balance_by must be 'task' or 'none'")

        optional_positive: dict[str, int | None] = {}
        for key in ("batch_size", "seq_len", "eval_steps", "save_steps"):
            value = raw.get(key)
            if value is not None and (not isinstance(value, int) or value < 1):
                raise ValueError(f"stage {name}: {key} must be a positive integer")
            optional_positive[key] = value
        return cls(
            name=name,
            max_examples=max_examples,
            epochs=epochs,
            lr=float(lr),
            include_previous=bool(raw.get("include_previous", True)),
            balance_by=balance_by,
            filters=CurriculumFilters.from_dict(raw.get("filters")),
            **optional_positive,
        )


@dataclass(frozen=True)
class SFTCurriculum:
    seed: int
    stages: tuple[CurriculumStage, ...]

    def stage(self, name: str) -> CurriculumStage:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(f"unknown curriculum stage {name!r}")

    def stage_index(self, name: str) -> int:
        for index, stage in enumerate(self.stages):
            if stage.name == name:
                return index
        raise KeyError(f"unknown curriculum stage {name!r}")


def load_curriculum(path: str | Path) -> SFTCurriculum:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("curriculum YAML must contain an object")
    if raw.get("version", 1) != 1:
        raise ValueError(f"unsupported curriculum version: {raw.get('version')}")
    stages_raw = raw.get("stages")
    if not isinstance(stages_raw, list) or not stages_raw:
        raise ValueError("curriculum YAML must contain at least one stage")
    stages = tuple(CurriculumStage.from_dict(item) for item in stages_raw)
    names = [stage.name for stage in stages]
    if len(names) != len(set(names)):
        raise ValueError("curriculum stage names must be unique")
    seed = raw.get("seed", 42)
    if not isinstance(seed, int):
        raise ValueError("curriculum seed must be an integer")
    return SFTCurriculum(seed=seed, stages=stages)


def load_projected_metadata(path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Load only routing metadata; prompt and response text are never scanned."""

    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError("curriculum Parquet metadata requires pyarrow") from error
    schema_names = set(pq.read_schema(path).names)
    columns = [
        name
        for name in ("example_id", "difficulty", "mode", "domain")
        if name in schema_names
    ]
    if "example_id" not in columns:
        raise ValueError(f"curriculum metadata has no example_id column: {path}")
    table = pq.read_table(path, columns=columns)
    material = table.to_pydict()
    result: dict[str, dict[str, Any]] = {}
    for row_index, example_id in enumerate(material["example_id"]):
        result[str(example_id)] = {
            name: material[name][row_index]
            for name in columns
            if name != "example_id"
        }
    return result


def _merged_example(
    example: dict[str, Any],
    metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    merged = dict(metadata.get(str(example.get("example_id", "")), {}))
    merged.update(example)
    return merged


def _eligible(example: dict[str, Any], filters: CurriculumFilters) -> bool:
    task = str(example.get("task", ""))
    if filters.tasks and task not in filters.tasks:
        return False
    if filters.exclude_tasks and task in filters.exclude_tasks:
        return False
    if filters.difficulties and str(example.get("difficulty", "")) not in filters.difficulties:
        return False
    if filters.modes and str(example.get("mode", "")) not in filters.modes:
        return False
    if (
        filters.training_representations
        and str(example.get("training_representation", ""))
        not in filters.training_representations
    ):
        return False
    if (
        filters.max_num_tokens is not None
        and int(example.get("num_tokens", 0)) > filters.max_num_tokens
    ):
        return False
    if (
        filters.max_supervised_tokens is not None
        and int(example.get("supervised_tokens", 0)) > filters.max_supervised_tokens
    ):
        return False
    return True


def _score(example: dict[str, Any], seed: int) -> bytes:
    example_id = str(example.get("example_id", ""))
    return hashlib.sha256(f"{seed}:{example_id}".encode("utf-8")).digest()


def _balanced_limit(
    examples: list[dict[str, Any]],
    limit: int,
    seed: int,
    balance_by: str,
) -> list[dict[str, Any]]:
    if len(examples) <= limit:
        return sorted(examples, key=lambda item: _score(item, seed))
    if balance_by == "none":
        return sorted(examples, key=lambda item: _score(item, seed))[:limit]

    groups: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        groups.setdefault(str(example.get("task", "unknown")), []).append(example)
    for group in groups.values():
        group.sort(key=lambda item: _score(item, seed))

    selected: list[dict[str, Any]] = []
    ordered_groups = sorted(groups)
    cursor = 0
    while len(selected) < limit and ordered_groups:
        remaining_groups: list[str] = []
        for group_name in ordered_groups:
            group = groups[group_name]
            if cursor < len(group):
                selected.append(group[cursor])
                if len(selected) == limit:
                    break
            if cursor + 1 < len(group):
                remaining_groups.append(group_name)
        ordered_groups = remaining_groups
        cursor += 1
    return selected


def select_stage_examples(
    examples: Iterable[dict[str, Any]],
    curriculum: SFTCurriculum,
    stage_name: str,
    metadata: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Select one stage in memory and retain earlier stages when requested."""

    material = list(examples)
    metadata = metadata or {}
    stage_index = curriculum.stage_index(stage_name)
    previous_ids: set[str] = set()
    if stage_index > 0 and curriculum.stages[stage_index].include_previous:
        previous = select_stage_examples(
            material,
            curriculum,
            curriculum.stages[stage_index - 1].name,
            metadata,
        )
        previous_ids = {str(item["example_id"]) for item in previous}

    stage = curriculum.stages[stage_index]
    eligible = [
        example
        for example in material
        if _eligible(_merged_example(example, metadata), stage.filters)
    ]
    by_id = {str(example["example_id"]): example for example in eligible}
    retained = [
        example
        for example in material
        if str(example.get("example_id")) in previous_ids
    ]
    retained_ids = {str(example["example_id"]) for example in retained}
    candidates = [
        example
        for example_id, example in by_id.items()
        if example_id not in retained_ids
    ]

    if stage.max_examples is None:
        selected = retained + sorted(candidates, key=lambda item: _score(item, curriculum.seed))
    else:
        remaining = max(0, stage.max_examples - len(retained))
        selected = retained + _balanced_limit(
            candidates,
            remaining,
            curriculum.seed,
            stage.balance_by,
        )
    if not selected:
        raise ValueError(f"curriculum stage {stage.name!r} selected zero examples")
    return selected

