"""PyTorch dataset and collation for extracted CUHK-X HAR trials."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional

import torch
from torch.utils.data import Dataset

from .config import SENSOR_MODALITIES
from .cuhkx_records import (
    CUHKXRecord,
    CUHKXTestRecord,
    build_cuhkx_records,
    build_cuhkx_test_records,
    subject_disjoint_records,
)
from .preprocessing import (
    CUHKXPreprocessingConfig,
    NoUsableModalityError,
    load_imu_sequence,
    load_radar_sequence,
    load_skeleton_sequence,
    load_visual_clip,
)


class _CUHKXModalities:
    preprocessing: CUHKXPreprocessingConfig
    visual_horizontal_flip: float = 0.0
    visual_crop_jitter: float = 0.0

    def _initialize_caches(self) -> None:
        # Dataset workers are persistent, so deterministic sensor parsing is
        # paid once rather than once per epoch. Visual tensors stay stochastic;
        # only their file index is cached in preprocessing.py.
        self._static_sensor_cache: dict[tuple[str, Path], torch.Tensor] = {}

    def _empty(self, modality: str) -> torch.Tensor:
        config = self.preprocessing
        if modality == "depth":
            return torch.zeros(3, config.clip_frames, config.image_size, config.image_size)
        if modality == "ir":
            return torch.zeros(1, config.clip_frames, config.image_size, config.image_size)
        if modality == "thermal":
            return torch.zeros(3, config.clip_frames, config.image_size, config.image_size)
        if modality == "imu":
            return torch.zeros(config.sensor_steps, config.imu_features)
        if modality == "radar":
            return torch.zeros(config.sensor_steps, config.radar_features)
        if modality == "skeleton":
            return torch.zeros(config.sensor_steps, config.skeleton_joints, 3)
        raise ValueError(f"unsupported modality: {modality}")

    def _load(
        self,
        modality: str,
        path: Path,
        *,
        augmentation_seed: int | None = None,
    ) -> torch.Tensor:
        if modality == "depth":
            return load_visual_clip(
                path,
                channels=3,
                config=self.preprocessing,
                augmentation_seed=augmentation_seed,
                horizontal_flip_probability=self.visual_horizontal_flip,
                crop_jitter=self.visual_crop_jitter,
            )
        if modality == "ir":
            return load_visual_clip(
                path,
                channels=1,
                config=self.preprocessing,
                augmentation_seed=augmentation_seed,
                horizontal_flip_probability=self.visual_horizontal_flip,
                crop_jitter=self.visual_crop_jitter,
            )
        if modality == "thermal":
            return load_visual_clip(
                path,
                channels=3,
                config=self.preprocessing,
                augmentation_seed=augmentation_seed,
                horizontal_flip_probability=self.visual_horizontal_flip,
                crop_jitter=self.visual_crop_jitter,
            )
        if modality in {"imu", "radar", "skeleton"}:
            cache = getattr(self, "_static_sensor_cache", None)
            if cache is None:
                self._initialize_caches()
                cache = self._static_sensor_cache
            key = (modality, path)
            if key not in cache:
                if modality == "imu":
                    cache[key] = load_imu_sequence(path, self.preprocessing)
                elif modality == "radar":
                    cache[key] = load_radar_sequence(path, self.preprocessing)
                else:
                    cache[key] = load_skeleton_sequence(path, self.preprocessing)
            return cache[key]
        raise ValueError(f"unsupported modality: {modality}")

    def _load_modalities(
        self,
        paths: Mapping[str, Path],
        *,
        augmentation_seed: int | None = None,
    ) -> tuple[dict, dict]:
        inputs: Dict[str, torch.Tensor] = {}
        modality_mask: Dict[str, torch.Tensor] = {}
        for modality in SENSOR_MODALITIES:
            path = paths.get(modality)
            present = path is not None
            if present:
                try:
                    inputs[modality] = self._load(
                        modality,
                        path,
                        augmentation_seed=augmentation_seed,
                    )
                except NoUsableModalityError:
                    present = False
                    inputs[modality] = self._empty(modality)
            else:
                inputs[modality] = self._empty(modality)
            modality_mask[modality] = torch.tensor(present, dtype=torch.bool)
        return inputs, modality_mask


class CUHKXSmallTrackDataset(_CUHKXModalities, Dataset):
    """Load synchronized CUHK-X modalities with explicit missing-sensor masks."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "train",
        preprocessing: Optional[CUHKXPreprocessingConfig] = None,
        validation_users: tuple[int, ...] = (8, 9, 23, 24),
        records: Optional[List[CUHKXRecord]] = None,
        training_augmentation: bool = False,
        seed: int = 42,
        visual_horizontal_flip: float = 0.0,
        visual_crop_jitter: float = 0.0,
    ):
        self.root = Path(root)
        self.preprocessing = preprocessing or CUHKXPreprocessingConfig()
        self.training_augmentation = bool(training_augmentation)
        self.seed = int(seed)
        self.visual_horizontal_flip = float(visual_horizontal_flip)
        self.visual_crop_jitter = float(visual_crop_jitter)
        self._initialize_caches()
        if split != "train" and self.training_augmentation:
            raise ValueError("training augmentation is only valid for the train split")
        indexed = records if records is not None else build_cuhkx_records(self.root)
        self.records = subject_disjoint_records(
            indexed,
            split=split,
            validation_users=validation_users,
        )
        self.loss_weights = torch.ones(len(self.records), dtype=torch.float32)

    def set_loss_weights(self, weights: torch.Tensor) -> None:
        weights = torch.as_tensor(weights, dtype=torch.float32, device="cpu")
        if weights.shape != (len(self.records),):
            raise ValueError("loss weights must match the dataset length")
        if not torch.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("loss weights must be finite and positive")
        self.loss_weights = weights

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int | tuple[int, int]) -> dict:
        epoch = 0
        if isinstance(index, tuple):
            index, epoch = index
        record = self.records[index]
        augmentation_seed = None
        if self.training_augmentation:
            augmentation_seed = self.seed + epoch * len(self.records) + index
        inputs, modality_mask = self._load_modalities(
            record.paths,
            augmentation_seed=augmentation_seed,
        )
        return {
            "inputs": inputs,
            "modality_mask": modality_mask,
            "label": torch.tensor(record.action_id, dtype=torch.long),
            "loss_weight": self.loss_weights[index],
            "metadata": {
                "action_name": record.action_name,
                "user_id": record.user_id,
                "trial_id": record.trial_id,
            },
        }


class CUHKXSmallTrackTestDataset(_CUHKXModalities, Dataset):
    """Load unlabeled official test clips strictly in test.csv order."""

    def __init__(
        self,
        root: str | Path,
        test_csv: str | Path,
        *,
        preprocessing: Optional[CUHKXPreprocessingConfig] = None,
        records: Optional[List[CUHKXTestRecord]] = None,
    ):
        self.root = Path(root)
        self.test_csv = Path(test_csv)
        self.preprocessing = preprocessing or CUHKXPreprocessingConfig()
        self._initialize_caches()
        self.records = records or build_cuhkx_test_records(self.root, self.test_csv)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        inputs, modality_mask = self._load_modalities(record.paths)
        return {
            "inputs": inputs,
            "modality_mask": modality_mask,
            "index": torch.tensor(record.index, dtype=torch.long),
            "path": record.path,
            "sample_id": record.sample_id,
        }


def collate_cuhkx(samples: List[Mapping]) -> dict:
    if not samples:
        raise ValueError("cannot collate an empty CUHK-X batch")
    return {
        "inputs": {
            modality: torch.stack([sample["inputs"][modality] for sample in samples])
            for modality in SENSOR_MODALITIES
        },
        "modality_mask": {
            modality: torch.stack(
                [sample["modality_mask"][modality] for sample in samples]
            )
            for modality in SENSOR_MODALITIES
        },
        "labels": torch.stack([sample["label"] for sample in samples]),
        "loss_weights": torch.stack([sample["loss_weight"] for sample in samples]),
        "metadata": [sample["metadata"] for sample in samples],
    }


def collate_cuhkx_test(samples: List[Mapping]) -> dict:
    if not samples:
        raise ValueError("cannot collate an empty CUHK-X test batch")
    return {
        "inputs": {
            modality: torch.stack([sample["inputs"][modality] for sample in samples])
            for modality in SENSOR_MODALITIES
        },
        "modality_mask": {
            modality: torch.stack(
                [sample["modality_mask"][modality] for sample in samples]
            )
            for modality in SENSOR_MODALITIES
        },
        "indices": torch.stack([sample["index"] for sample in samples]),
        "paths": [sample["path"] for sample in samples],
        "sample_ids": [sample["sample_id"] for sample in samples],
    }
