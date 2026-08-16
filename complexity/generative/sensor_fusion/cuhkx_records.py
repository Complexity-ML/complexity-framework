"""Filesystem indexing for extracted CUHK-X Small Model Track clips."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

CUHKX_DIRECTORY_MODALITIES: Mapping[str, str] = {
    "Depth_Color": "depth",
    "IR": "ir",
    "Thermal": "thermal",
    "IMU": "imu",
    "Radar": "radar",
    "Skeleton": "skeleton",
}

CUHKX_TRAIN_USERS = tuple(range(1, 10)) + tuple(range(16, 25))


@dataclass(frozen=True)
class CUHKXRecord:
    """One synchronized activity trial across the available sensor folders."""

    action_id: int
    action_name: str
    user_id: int
    trial_id: str
    paths: Mapping[str, Path]

    @property
    def key(self) -> tuple[int, int, str]:
        return self.action_id, self.user_id, self.trial_id


@dataclass(frozen=True)
class CUHKXTestRecord:
    """One unlabeled official test clip in the order defined by test.csv."""

    index: int
    path: str
    sample_id: str
    paths: Mapping[str, Path]


def _parse_action(folder: str) -> tuple[int, str]:
    prefix, separator, name = folder.partition("_")
    if not separator or not prefix.isdigit():
        raise ValueError(f"invalid CUHK-X action directory: {folder!r}")
    action_id = int(prefix)
    if not 0 <= action_id < 40:
        raise ValueError(f"CUHK-X action id must be in [0, 39], got {action_id}")
    return action_id, name


def _parse_user(folder: str) -> int:
    if not folder.startswith("user") or not folder[4:].isdigit():
        raise ValueError(f"invalid CUHK-X user directory: {folder!r}")
    return int(folder[4:])


def build_cuhkx_records(root: str | Path) -> List[CUHKXRecord]:
    """Index an extracted ``HAR/data`` tree without reading clip contents."""

    root = Path(root)
    if (root / "HAR" / "data").is_dir():
        root = root / "HAR" / "data"
    if not root.is_dir():
        raise FileNotFoundError(f"CUHK-X HAR/data directory not found: {root}")

    indexed: Dict[tuple[int, int, str], dict] = {}
    for directory_name, modality in CUHKX_DIRECTORY_MODALITIES.items():
        modality_root = root / directory_name
        if not modality_root.is_dir():
            continue
        for action_path in sorted(path for path in modality_root.iterdir() if path.is_dir()):
            action_id, action_name = _parse_action(action_path.name)
            for user_path in sorted(path for path in action_path.iterdir() if path.is_dir()):
                user_id = _parse_user(user_path.name)
                for trial_path in sorted(path for path in user_path.iterdir() if path.is_dir()):
                    if not any(path.is_file() for path in trial_path.rglob("*")):
                        continue
                    key = (action_id, user_id, trial_path.name)
                    entry = indexed.setdefault(
                        key,
                        {
                            "action_id": action_id,
                            "action_name": action_name,
                            "user_id": user_id,
                            "trial_id": trial_path.name,
                            "paths": {},
                        },
                    )
                    entry["paths"][modality] = trial_path
    return [CUHKXRecord(**indexed[key]) for key in sorted(indexed)]


def save_cuhkx_manifest(records: Iterable[CUHKXRecord], path: str | Path) -> None:
    """Persist the expensive filesystem index without copying any sensor data."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "action_id": record.action_id,
            "action_name": record.action_name,
            "user_id": record.user_id,
            "trial_id": record.trial_id,
            "paths": {name: str(value) for name, value in record.paths.items()},
        }
        for record in records
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def load_cuhkx_manifest(path: str | Path) -> List[CUHKXRecord]:
    """Read a local manifest and ensure every referenced trial still exists."""

    path = Path(path)
    payload = json.loads(path.read_text())
    records = [
        CUHKXRecord(
            action_id=int(item["action_id"]),
            action_name=str(item["action_name"]),
            user_id=int(item["user_id"]),
            trial_id=str(item["trial_id"]),
            paths={name: Path(value) for name, value in item["paths"].items()},
        )
        for item in payload
    ]
    missing = [path for record in records for path in record.paths.values() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"CUHK-X manifest references a missing trial: {missing[0]}")
    return records


def build_cuhkx_test_records(
    root: str | Path,
    test_csv: str | Path,
) -> List[CUHKXTestRecord]:
    """Index official unlabeled clips without consulting sample predictions."""

    root = Path(root)
    test_csv = Path(test_csv)
    with test_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["path", "prediction"]:
            raise ValueError("CUHK-X test.csv must have exactly path,prediction columns")
        rows = list(reader)
    if not rows:
        raise ValueError("CUHK-X test.csv is empty")

    records = []
    seen = set()
    for index, row in enumerate(rows):
        relative = row["path"]
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"unsafe CUHK-X test path: {relative!r}")
        if row.get("prediction", "").strip():
            raise ValueError("test.csv must be unlabeled; refusing prediction leakage")
        sample_id = Path(relative.rstrip("/")).name
        if not sample_id.startswith("SM_test_") or not sample_id[8:].isdigit():
            raise ValueError(f"invalid CUHK-X test sample id: {sample_id!r}")
        if sample_id in seen:
            raise ValueError(f"duplicate CUHK-X test sample id: {sample_id}")
        seen.add(sample_id)
        sample_root = root / relative
        if not sample_root.is_dir():
            raise FileNotFoundError(f"CUHK-X test clip not found: {sample_root}")
        paths = {
            modality: sample_root / directory_name
            for directory_name, modality in CUHKX_DIRECTORY_MODALITIES.items()
            if (sample_root / directory_name).is_dir()
            and any(path.is_file() for path in (sample_root / directory_name).rglob("*"))
        }
        if not paths:
            raise ValueError(f"CUHK-X test clip has no supported modalities: {sample_root}")
        records.append(
            CUHKXTestRecord(
                index=index,
                path=relative,
                sample_id=sample_id,
                paths=paths,
            )
        )
    return records


def subject_disjoint_records(
    records: Iterable[CUHKXRecord],
    *,
    split: str,
    validation_users: tuple[int, ...] = (8, 9, 23, 24),
) -> List[CUHKXRecord]:
    """Create a reproducible train/validation split with disjoint people."""

    records = list(records)
    validation = set(validation_users)
    official_train = set(CUHKX_TRAIN_USERS)
    if not validation or not validation.issubset(official_train):
        raise ValueError("validation_users must be a non-empty subset of CUHK-X train users")
    if split == "train":
        return [
            record
            for record in records
            if record.user_id in official_train and record.user_id not in validation
        ]
    if split == "validation":
        return [record for record in records if record.user_id in validation]
    if split == "all":
        return records
    raise ValueError("split must be train, validation, or all")
