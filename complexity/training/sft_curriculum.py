"""Deterministic, runtime-only curriculum selection for SFT shards.

The curriculum never rewrites or republishes a dataset.  It selects examples
from the existing ``examples.jsonl`` index when a stage starts. Optional row
metadata is read from the projected Parquet stored beside the token shards.
Text and structural signatures are loaded only so a stage can reject known
boilerplate and cap repeated response surfaces.
"""

from __future__ import annotations

import hashlib
from collections import Counter
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


def _task_weights(value: Any, stage_name: str) -> tuple[tuple[str, float], ...]:
    if value is None:
        return ()
    if not isinstance(value, dict) or not value:
        raise ValueError(f"stage {stage_name}: task_weights must be a non-empty object")
    result: list[tuple[str, float]] = []
    for task, weight in value.items():
        task = str(task).strip()
        if not task or not isinstance(weight, (int, float)) or weight <= 0:
            raise ValueError(
                f"stage {stage_name}: every task weight must have a name and be positive"
            )
        result.append((task, float(weight)))
    return tuple(sorted(result))


def _task_target_shares(value: Any, stage_name: str) -> tuple[tuple[str, float], ...]:
    """Parse desired supervised-token loss shares for a full task mixture."""

    if value is None:
        return ()
    if not isinstance(value, dict) or not value:
        raise ValueError(
            f"stage {stage_name}: loss_task_targets must be a non-empty object"
        )
    result: list[tuple[str, float]] = []
    for task, share in value.items():
        task = str(task).strip()
        if not task or not isinstance(share, (int, float)) or share <= 0:
            raise ValueError(
                f"stage {stage_name}: every loss task target must have a name "
                "and be positive"
            )
        result.append((task, float(share)))
    total = sum(share for _, share in result)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"stage {stage_name}: loss_task_targets must sum to 1, got {total:.12g}"
        )
    return tuple(sorted(result))


@dataclass(frozen=True)
class CurriculumFilters:
    tasks: tuple[str, ...] = ()
    exclude_tasks: tuple[str, ...] = ()
    difficulties: tuple[str, ...] = ()
    modes: tuple[str, ...] = ()
    training_representations: tuple[str, ...] = ()
    max_num_tokens: int | None = None
    max_supervised_tokens: int | None = None
    exclude_response_substrings: tuple[str, ...] = ()
    max_structure_occurrences_per_task: int | None = None
    max_opening_occurrences_per_task: int | None = None
    opening_words: int = 6
    opening_cap_exempt_tasks: tuple[str, ...] = ()

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
            "exclude_response_substrings",
            "max_structure_occurrences_per_task",
            "max_opening_occurrences_per_task",
            "opening_words",
            "opening_cap_exempt_tasks",
        }
        unknown = set(raw).difference(supported)
        if unknown:
            raise ValueError(f"unsupported curriculum filters: {sorted(unknown)}")
        max_num_tokens = raw.get("max_num_tokens")
        max_supervised_tokens = raw.get("max_supervised_tokens")
        for name, value in (
            ("max_num_tokens", max_num_tokens),
            ("max_supervised_tokens", max_supervised_tokens),
            (
                "max_structure_occurrences_per_task",
                raw.get("max_structure_occurrences_per_task"),
            ),
            (
                "max_opening_occurrences_per_task",
                raw.get("max_opening_occurrences_per_task"),
            ),
            ("opening_words", raw.get("opening_words", 6)),
        ):
            if value is not None and (not isinstance(value, int) or value < 1):
                raise ValueError(f"curriculum {name} must be a positive integer")
        return cls(
            tasks=_string_tuple(raw.get("tasks"), "filters.tasks"),
            exclude_tasks=_string_tuple(raw.get("exclude_tasks"), "filters.exclude_tasks"),
            difficulties=_string_tuple(raw.get("difficulties"), "filters.difficulties"),
            modes=_string_tuple(raw.get("modes"), "filters.modes"),
            training_representations=_string_tuple(
                raw.get("training_representations"),
                "filters.training_representations",
            ),
            max_num_tokens=max_num_tokens,
            max_supervised_tokens=max_supervised_tokens,
            exclude_response_substrings=_string_tuple(
                raw.get("exclude_response_substrings"),
                "filters.exclude_response_substrings",
            ),
            max_structure_occurrences_per_task=raw.get("max_structure_occurrences_per_task"),
            max_opening_occurrences_per_task=raw.get("max_opening_occurrences_per_task"),
            opening_words=raw.get("opening_words", 6),
            opening_cap_exempt_tasks=_string_tuple(
                raw.get("opening_cap_exempt_tasks"),
                "filters.opening_cap_exempt_tasks",
            ),
        )


@dataclass(frozen=True)
class CurriculumLossGroup:
    """Target weighted-token share for a group and optionally its tasks."""

    name: str
    tasks: tuple[str, ...]
    target_share: float
    task_target_shares: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class CurriculumLossCell:
    """Runtime loss key for one task restricted to explicit domains."""

    name: str
    task: str
    domains: tuple[str, ...]


def _loss_cells(value: Any, stage_name: str) -> tuple[CurriculumLossCell, ...]:
    if value is None:
        return ()
    if not isinstance(value, dict) or not value:
        raise ValueError(f"stage {stage_name}: loss_cells must be a non-empty object")
    cells: list[CurriculumLossCell] = []
    assigned: set[tuple[str, str]] = set()
    for raw_name, raw_cell in value.items():
        name = str(raw_name).strip()
        if not name or not isinstance(raw_cell, dict):
            raise ValueError(f"stage {stage_name}: every loss cell needs an object")
        task = str(raw_cell.get("task", "")).strip()
        domains = _string_tuple(raw_cell.get("domains"), f"loss_cells.{name}.domains")
        if not task or not domains:
            raise ValueError(
                f"stage {stage_name}: loss cell {name} requires a task and domains"
            )
        overlap = {(task, domain) for domain in domains}.intersection(assigned)
        if overlap:
            raise ValueError(
                f"stage {stage_name}: loss cells overlap on {sorted(overlap)}"
            )
        assigned.update((task, domain) for domain in domains)
        cells.append(CurriculumLossCell(name=name, task=task, domains=domains))
    return tuple(cells)


def loss_weight_key(stage: "CurriculumStage", example: dict[str, Any]) -> str:
    """Resolve the configured task×domain cell, or retain the ordinary task."""

    task = str(example.get("task", "unknown"))
    domain = str(example.get("domain", ""))
    matches = [
        cell.name
        for cell in stage.loss_cells
        if cell.task == task and domain in cell.domains
    ]
    if len(matches) > 1:  # Defensive: parser rejects this for static configs.
        raise ValueError(
            f"example {example.get('example_id')} matches multiple loss cells: {matches}"
        )
    return matches[0] if matches else task


def _group_task_target_shares(
    value: Any,
    *,
    stage_name: str,
    group_name: str,
    tasks: tuple[str, ...],
) -> tuple[tuple[str, float], ...]:
    if value is None:
        return ()
    if not isinstance(value, dict) or not value:
        raise ValueError(
            f"stage {stage_name}: loss group {group_name} "
            "task_target_shares must be a non-empty object"
        )
    result: list[tuple[str, float]] = []
    for raw_task, raw_share in value.items():
        task = str(raw_task).strip()
        if not task or not isinstance(raw_share, (int, float)) or raw_share <= 0:
            raise ValueError(
                f"stage {stage_name}: loss group {group_name} task target "
                "shares must have a name and be positive"
            )
        result.append((task, float(raw_share)))
    target_tasks = {task for task, _ in result}
    assigned_tasks = set(tasks)
    if target_tasks != assigned_tasks:
        raise ValueError(
            f"stage {stage_name}: loss group {group_name} task_target_shares "
            f"must exactly match tasks; missing={sorted(assigned_tasks - target_tasks)} "
            f"unused={sorted(target_tasks - assigned_tasks)}"
        )
    total = sum(share for _, share in result)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"stage {stage_name}: loss group {group_name} task_target_shares "
            f"must sum to 1, got {total:.12g}"
        )
    return tuple(sorted(result))


def _loss_groups(value: Any, stage_name: str) -> tuple[CurriculumLossGroup, ...]:
    if value is None:
        return ()
    if not isinstance(value, dict) or not value:
        raise ValueError(f"stage {stage_name}: loss_groups must be a non-empty object")
    groups: list[CurriculumLossGroup] = []
    assigned: set[str] = set()
    for raw_name, raw_group in value.items():
        name = str(raw_name).strip()
        if not name or not isinstance(raw_group, dict):
            raise ValueError(f"stage {stage_name}: every loss group needs an object")
        tasks = _string_tuple(raw_group.get("tasks"), f"loss_groups.{name}.tasks")
        if not tasks:
            raise ValueError(f"stage {stage_name}: loss group {name} has no tasks")
        overlap = assigned.intersection(tasks)
        if overlap:
            raise ValueError(
                f"stage {stage_name}: loss groups assign tasks more than once: "
                f"{sorted(overlap)}"
            )
        assigned.update(tasks)
        share = raw_group.get("target_share")
        if not isinstance(share, (int, float)) or not 0 < share <= 1:
            raise ValueError(
                f"stage {stage_name}: loss group {name} target_share must be in (0, 1]"
            )
        task_target_shares = _group_task_target_shares(
            raw_group.get("task_target_shares"),
            stage_name=stage_name,
            group_name=name,
            tasks=tasks,
        )
        groups.append(
            CurriculumLossGroup(name, tasks, float(share), task_target_shares)
        )
    total = sum(group.target_share for group in groups)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"stage {stage_name}: loss group target shares must sum to 1, got {total:.12g}"
        )
    return tuple(groups)


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    max_examples: int | None
    epochs: int
    lr: float
    include_previous: bool = True
    balance_by: str = "task"
    task_weights: tuple[tuple[str, float], ...] = ()
    loss_task_targets: tuple[tuple[str, float], ...] = ()
    loss_groups: tuple[CurriculumLossGroup, ...] = ()
    loss_cells: tuple[CurriculumLossCell, ...] = ()
    max_task_loss_weight: float | None = None
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
        if balance_by not in {"task", "weighted_task", "none"}:
            raise ValueError(f"stage {name}: balance_by must be 'task', 'weighted_task', or 'none'")
        task_weights = _task_weights(raw.get("task_weights"), name)
        if balance_by == "weighted_task" and not task_weights:
            raise ValueError(f"stage {name}: weighted_task balancing requires task_weights")
        if balance_by != "weighted_task" and task_weights:
            raise ValueError(f"stage {name}: task_weights require balance_by=weighted_task")
        loss_task_targets = _task_target_shares(raw.get("loss_task_targets"), name)
        loss_groups = _loss_groups(raw.get("loss_groups"), name)
        loss_cells = _loss_cells(raw.get("loss_cells"), name)
        if loss_task_targets and loss_groups:
            raise ValueError(
                f"stage {name}: use loss_task_targets or loss_groups, not both"
            )
        max_task_loss_weight = raw.get("max_task_loss_weight")
        if max_task_loss_weight is not None and (
            not isinstance(max_task_loss_weight, (int, float))
            or max_task_loss_weight <= 0
        ):
            raise ValueError(
                f"stage {name}: max_task_loss_weight must be positive"
            )

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
            task_weights=task_weights,
            loss_task_targets=loss_task_targets,
            loss_groups=loss_groups,
            loss_cells=loss_cells,
            max_task_loss_weight=(
                None
                if max_task_loss_weight is None
                else float(max_task_loss_weight)
            ),
            filters=CurriculumFilters.from_dict(raw.get("filters")),
            **optional_positive,
        )


@dataclass(frozen=True)
class CurriculumExposureGroup:
    """Expected share of planned training exposures for a task group."""

    name: str
    tasks: tuple[str, ...]
    target_share: float
    tolerance: float = 0.01
    token_target_share: float | None = None
    token_tolerance: float = 0.02


def _exposure_groups(value: Any) -> tuple[CurriculumExposureGroup, ...]:
    if value is None:
        return ()
    if not isinstance(value, dict) or not value:
        raise ValueError("curriculum exposure_groups must be a non-empty object")

    groups: list[CurriculumExposureGroup] = []
    assigned_tasks: set[str] = set()
    for raw_name, raw_group in value.items():
        name = str(raw_name).strip()
        if not name or not isinstance(raw_group, dict):
            raise ValueError("every exposure group needs a name and an object")
        tasks = _string_tuple(raw_group.get("tasks"), f"exposure_groups.{name}.tasks")
        if not tasks:
            raise ValueError(f"exposure group {name}: tasks must not be empty")
        overlap = assigned_tasks.intersection(tasks)
        if overlap:
            raise ValueError(f"exposure groups assign tasks more than once: {sorted(overlap)}")
        assigned_tasks.update(tasks)
        target_share = raw_group.get("target_share")
        tolerance = raw_group.get("tolerance", 0.01)
        token_target_share = raw_group.get("token_target_share")
        token_tolerance = raw_group.get("token_tolerance", 0.02)
        if not isinstance(target_share, (int, float)) or not 0 < target_share <= 1:
            raise ValueError(f"exposure group {name}: target_share must be in (0, 1]")
        if not isinstance(tolerance, (int, float)) or not 0 <= tolerance < 1:
            raise ValueError(f"exposure group {name}: tolerance must be in [0, 1)")
        if token_target_share is not None and (
            not isinstance(token_target_share, (int, float))
            or not 0 < token_target_share <= 1
        ):
            raise ValueError(
                f"exposure group {name}: token_target_share must be in (0, 1]"
            )
        if not isinstance(token_tolerance, (int, float)) or not 0 <= token_tolerance < 1:
            raise ValueError(
                f"exposure group {name}: token_tolerance must be in [0, 1)"
            )
        groups.append(
            CurriculumExposureGroup(
                name=name,
                tasks=tasks,
                target_share=float(target_share),
                tolerance=float(tolerance),
                token_target_share=(
                    None
                    if token_target_share is None
                    else float(token_target_share)
                ),
                token_tolerance=float(token_tolerance),
            )
        )
    if abs(sum(group.target_share for group in groups) - 1.0) > 1e-9:
        raise ValueError("curriculum exposure group target shares must sum to 1")
    token_targets = [group.token_target_share for group in groups]
    if any(target is not None for target in token_targets):
        if any(target is None for target in token_targets):
            raise ValueError(
                "every exposure group must set token_target_share when one does"
            )
        if abs(sum(float(target) for target in token_targets) - 1.0) > 1e-9:
            raise ValueError(
                "curriculum exposure group token target shares must sum to 1"
            )
    return tuple(groups)


@dataclass(frozen=True)
class SFTCurriculum:
    seed: int
    stages: tuple[CurriculumStage, ...]
    exposure_groups: tuple[CurriculumExposureGroup, ...] = ()

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
    return SFTCurriculum(
        seed=seed,
        stages=stages,
        exposure_groups=_exposure_groups(raw.get("exposure_groups")),
    )


def load_projected_metadata(path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Load metadata used by runtime curriculum routing and quality filters."""

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
        for name in (
            "example_id",
            "difficulty",
            "mode",
            "domain",
            "response",
            "structure_signature",
        )
        if name in schema_names
    ]
    if "example_id" not in columns:
        raise ValueError(f"curriculum metadata has no example_id column: {path}")
    table = pq.read_table(path, columns=columns)
    material = table.to_pydict()
    result: dict[str, dict[str, Any]] = {}
    for row_index, example_id in enumerate(material["example_id"]):
        result[str(example_id)] = {
            name: material[name][row_index] for name in columns if name != "example_id"
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
        and str(example.get("training_representation", "")) not in filters.training_representations
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
    response = str(example.get("response", "")).casefold()
    if filters.exclude_response_substrings and any(
        phrase.casefold() in response for phrase in filters.exclude_response_substrings
    ):
        return False
    return True


def _score(example: dict[str, Any], seed: int) -> bytes:
    example_id = str(example.get("example_id", ""))
    return hashlib.sha256(f"{seed}:{example_id}".encode("utf-8")).digest()


def _response_opening(response: str, words: int) -> str:
    normalized = "".join(
        character.casefold() if character.isalnum() else " " for character in response
    )
    return " ".join(normalized.split()[:words])


def _limit_repeated_surfaces(
    examples: list[dict[str, Any]],
    filters: CurriculumFilters,
    seed: int,
) -> list[dict[str, Any]]:
    """Cap repeated structures and openings independently inside each task."""

    structure_cap = filters.max_structure_occurrences_per_task
    opening_cap = filters.max_opening_occurrences_per_task
    if structure_cap is None and opening_cap is None:
        return examples

    groups: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        groups.setdefault(str(example.get("task", "unknown")), []).append(example)

    selected: list[dict[str, Any]] = []
    exempt_openings = set(filters.opening_cap_exempt_tasks)
    for task, group in sorted(groups.items()):
        structure_counts: dict[str, int] = {}
        opening_counts: dict[str, int] = {}
        for example in sorted(group, key=lambda item: _score(item, seed)):
            structure = str(example.get("structure_signature", "")).strip()
            opening = _response_opening(str(example.get("response", "")), filters.opening_words)
            if (
                structure_cap is not None
                and structure
                and structure_counts.get(structure, 0) >= structure_cap
            ):
                continue
            if (
                opening_cap is not None
                and task not in exempt_openings
                and opening
                and opening_counts.get(opening, 0) >= opening_cap
            ):
                continue
            selected.append(example)
            if structure:
                structure_counts[structure] = structure_counts.get(structure, 0) + 1
            if opening:
                opening_counts[opening] = opening_counts.get(opening, 0) + 1
    return selected


def _balanced_limit(
    examples: list[dict[str, Any]],
    limit: int,
    seed: int,
    balance_by: str,
    task_weights: tuple[tuple[str, float], ...] = (),
    retained: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    retained = retained or []
    if len(examples) <= limit and balance_by != "weighted_task":
        return sorted(examples, key=lambda item: _score(item, seed))
    if balance_by == "none":
        return sorted(examples, key=lambda item: _score(item, seed))[:limit]

    groups: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        groups.setdefault(str(example.get("task", "unknown")), []).append(example)
    for group in groups.values():
        group.sort(key=lambda item: _score(item, seed))

    if balance_by == "weighted_task":
        weights = dict(task_weights)
        groups = {name: group for name, group in groups.items() if name in weights}
        retained_counts: dict[str, int] = {
            name: sum(str(row.get("task", "unknown")) == name for row in retained)
            for name in weights
        }
        total_weight = sum(weights.values())
        selected_counts = dict(retained_counts)
        selected: list[dict[str, Any]] = []
        while len(selected) < limit:
            active = [
                name
                for name, group in groups.items()
                if selected_counts.get(name, 0) - retained_counts.get(name, 0) < len(group)
            ]
            if not active:
                break
            total_after = len(retained) + len(selected) + 1
            # Select the task furthest below its weighted target. Stable task
            # names make ties deterministic across machines and input order.
            task = max(
                active,
                key=lambda name: (
                    weights[name] * total_after / total_weight - selected_counts.get(name, 0),
                    name,
                ),
            )
            consumed = selected_counts.get(task, 0) - retained_counts.get(task, 0)
            selected.append(groups[task][consumed])
            selected_counts[task] = selected_counts.get(task, 0) + 1
        return selected

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
    eligible = []
    for example in material:
        merged = _merged_example(example, metadata)
        if _eligible(merged, stage.filters):
            eligible.append(merged)
    eligible = _limit_repeated_surfaces(
        eligible,
        stage.filters,
        curriculum.seed,
    )
    by_id = {str(example["example_id"]): example for example in eligible}
    retained = [
        _merged_example(example, metadata)
        for example in material
        if str(example.get("example_id")) in previous_ids
    ]
    retained_ids = {str(example["example_id"]) for example in retained}
    candidates = [
        example for example_id, example in by_id.items() if example_id not in retained_ids
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
            stage.task_weights,
            retained,
        )
    if not selected:
        raise ValueError(f"curriculum stage {stage.name!r} selected zero examples")
    return selected


def audit_planned_exposures(
    examples: Iterable[dict[str, Any]],
    curriculum: SFTCurriculum,
    metadata: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Audit the selected-example exposure mix across all configured stages.

    One exposure is one selected row in one epoch. This deliberately counts a
    small, high-quality adaptation set again when a stage repeats it, without
    duplicating that row in the canonical dataset.
    """

    material = list(examples)
    metadata = metadata or {}
    task_exposures: Counter[str] = Counter()
    task_supervised_token_exposures: Counter[str] = Counter()
    stage_exposures: dict[str, dict[str, Any]] = {}
    for stage in curriculum.stages:
        selected = select_stage_examples(
            material,
            curriculum,
            stage.name,
            metadata,
        )
        counts = Counter(str(row.get("task", "unknown")) for row in selected)
        exposures = {task: count * stage.epochs for task, count in sorted(counts.items())}
        task_exposures.update(exposures)
        supervised_tokens = Counter()
        for row in selected:
            supervised_tokens[str(row.get("task", "unknown"))] += int(
                row.get("supervised_tokens", 0)
            )
        task_supervised_token_exposures.update(
            {
                task: count * stage.epochs
                for task, count in supervised_tokens.items()
            }
        )
        stage_exposures[stage.name] = {
            "selected_examples": len(selected),
            "epochs": stage.epochs,
            "total_exposures": len(selected) * stage.epochs,
            "task_exposures": exposures,
        }

    total = sum(task_exposures.values())
    supervised_token_total = sum(task_supervised_token_exposures.values())
    assigned_tasks = {task for group in curriculum.exposure_groups for task in group.tasks}
    unassigned = sorted(set(task_exposures).difference(assigned_tasks))
    group_results: dict[str, dict[str, Any]] = {}
    checks: dict[str, bool] = {}
    for group in curriculum.exposure_groups:
        count = sum(task_exposures[task] for task in group.tasks)
        share = count / total if total else 0.0
        passed = abs(share - group.target_share) <= group.tolerance
        checks[f"{group.name}_within_tolerance"] = passed
        supervised_tokens = sum(
            task_supervised_token_exposures[task] for task in group.tasks
        )
        token_share = (
            supervised_tokens / supervised_token_total
            if supervised_token_total
            else 0.0
        )
        token_passed = None
        if group.token_target_share is not None:
            token_passed = (
                abs(token_share - group.token_target_share)
                <= group.token_tolerance
            )
            checks[
                f"{group.name}_supervised_tokens_within_tolerance"
            ] = token_passed
        group_results[group.name] = {
            "tasks": list(group.tasks),
            "exposures": count,
            "share": round(share, 6),
            "target_share": group.target_share,
            "tolerance": group.tolerance,
            "passed": passed,
            "supervised_token_exposures": supervised_tokens,
            "supervised_token_share": round(token_share, 6),
            "token_target_share": group.token_target_share,
            "token_tolerance": group.token_tolerance,
            "token_passed": token_passed,
        }
    if curriculum.exposure_groups:
        checks["every_selected_task_is_assigned"] = not unassigned

    return {
        "unit": "selected_examples_x_epochs",
        "total_exposures": total,
        "task_exposures": dict(sorted(task_exposures.items())),
        "supervised_token_exposures": supervised_token_total,
        "task_supervised_token_exposures": dict(
            sorted(task_supervised_token_exposures.items())
        ),
        "stage_exposures": stage_exposures,
        "groups": group_results,
        "unassigned_tasks": unassigned,
        "checks": checks,
        "passed": bool(checks) and all(checks.values()),
    }


def audit_stage_loss_targets(
    examples: Iterable[dict[str, Any]],
    curriculum: SFTCurriculum,
    stage_name: str,
    metadata: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Verify that task loss targets cover a stage without dropping rows.

    This audit uses the token counts recorded by the shard index. The dataset
    loader recomputes the final multipliers from the actually visible labels
    after sequence truncation, so its runtime report is authoritative.
    """

    material = list(examples)
    stage = curriculum.stage(stage_name)
    selected = select_stage_examples(material, curriculum, stage_name, metadata)
    token_counts: Counter[str] = Counter()
    for row in selected:
        token_counts[loss_weight_key(stage, row)] += int(
            row.get("supervised_tokens", 0)
        )
    mix = derive_task_loss_weights(stage, token_counts)
    checks = {
        "stage_keeps_every_input_example": len(selected) == len(material),
        "target_tasks_match_selected_tasks": not mix["missing_targets"]
        and not mix["unused_targets"],
        "every_selected_task_has_supervision": all(count > 0 for count in token_counts.values()),
        "weighted_shares_match_targets": mix["shares_match_targets"],
        "task_loss_weights_within_cap": mix["weights_within_cap"],
    }
    return {
        "stage": stage_name,
        "input_examples": len(material),
        "selected_examples": len(selected),
        "raw_supervised_tokens": sum(token_counts.values()),
        "raw_task_supervised_tokens": dict(sorted(token_counts.items())),
        **mix,
        "checks": checks,
        "passed": bool(stage.loss_task_targets or stage.loss_groups)
        and all(checks.values()),
    }


def derive_task_loss_weights(
    stage: CurriculumStage,
    token_counts: dict[str, int] | Counter[str],
) -> dict[str, Any]:
    """Derive stable task coefficients from raw supervised-token counts."""

    counts = {str(task): int(count) for task, count in token_counts.items()}
    selected_tasks = set(counts)
    total = sum(counts.values())
    if stage.loss_groups:
        target_tasks = {task for group in stage.loss_groups for task in group.tasks}
        weights: dict[str, float] = {}
        group_targets = {group.name: group.target_share for group in stage.loss_groups}
        group_task_targets: dict[str, dict[str, float]] = {}
        for group in stage.loss_groups:
            group_tokens = sum(counts.get(task, 0) for task in group.tasks)
            if total and group_tokens:
                nested_targets = dict(group.task_target_shares)
                group_task_targets[group.name] = nested_targets
                if nested_targets:
                    weights.update(
                        {
                            task: (group.target_share * nested_targets[task])
                            / (counts[task] / total)
                            for task in group.tasks
                            if task in counts and counts[task]
                        }
                    )
                else:
                    coefficient = group.target_share / (group_tokens / total)
                    weights.update(
                        {task: coefficient for task in group.tasks if task in counts}
                    )
    else:
        targets = dict(stage.loss_task_targets)
        target_tasks = set(targets)
        group_targets = {}
        group_task_targets = {}
        weights = {
            task: targets[task] / (counts[task] / total)
            for task in sorted(selected_tasks & target_tasks)
            if total and counts[task]
        }
    weighted_total = sum(counts[task] * weights.get(task, 0.0) for task in counts)
    weighted_task_shares = {
        task: counts[task] * weights.get(task, 0.0) / weighted_total
        for task in sorted(counts)
        if weighted_total
    }
    weighted_group_shares = {
        group.name: sum(weighted_task_shares.get(task, 0.0) for task in group.tasks)
        for group in stage.loss_groups
    }
    if stage.loss_groups:
        group_shares_match = all(
            abs(weighted_group_shares.get(name, 0.0) - share) <= 1e-9
            for name, share in group_targets.items()
        )
        nested_shares_match = all(
            abs(
                weighted_task_shares.get(task, 0.0)
                - group.target_share * nested_share
            )
            <= 1e-9
            for group in stage.loss_groups
            for task, nested_share in group.task_target_shares
        )
        shares_match = (
            target_tasks == selected_tasks
            and group_shares_match
            and nested_shares_match
        )
    else:
        targets = dict(stage.loss_task_targets)
        shares_match = target_tasks == selected_tasks and all(
            abs(weighted_task_shares.get(task, 0.0) - share) <= 1e-9
            for task, share in targets.items()
        )
    overweight_tasks = {
        task: weight
        for task, weight in weights.items()
        if stage.max_task_loss_weight is not None
        and weight > stage.max_task_loss_weight + 1e-12
    }
    return {
        "loss_task_targets": dict(stage.loss_task_targets),
        "loss_group_targets": group_targets,
        "loss_group_task_targets": group_task_targets,
        "max_task_loss_weight": stage.max_task_loss_weight,
        "task_loss_weights": dict(sorted(weights.items())),
        "weighted_task_shares": weighted_task_shares,
        "weighted_group_shares": weighted_group_shares,
        "missing_targets": sorted(selected_tasks - target_tasks),
        "unused_targets": sorted(target_tasks - selected_tasks),
        "shares_match_targets": shares_match,
        "overweight_tasks": dict(sorted(overweight_tasks.items())),
        "weights_within_cap": not overweight_tasks,
    }
