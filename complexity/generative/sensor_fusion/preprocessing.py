"""Local preprocessing of CUHK-X visual, inertial, radar, and pose streams."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError


class NoUsableModalityError(FileNotFoundError):
    """Raised when a modality folder contains no usable sensor observation."""


class NoUsableVisualFramesError(NoUsableModalityError):
    """Raised when a visual modality contains no decodable frame."""


@dataclass(frozen=True)
class CUHKXPreprocessingConfig:
    version: int = 1
    image_size: int = 112
    clip_frames: int = 16
    sensor_steps: int = 64
    imu_features: int = 45
    radar_features: int = 16
    skeleton_joints: int = 17

    def __post_init__(self) -> None:
        if self.version not in (1, 2, 3):
            raise ValueError("CUHK-X preprocessing version must be 1, 2, or 3")
        if min(
            self.image_size,
            self.clip_frames,
            self.sensor_steps,
            self.imu_features,
            self.radar_features,
            self.skeleton_joints,
        ) <= 0:
            raise ValueError("all preprocessing dimensions must be positive")
        if self.imu_features != 45:
            raise ValueError("CUHK-X IMU preprocessing emits five devices x nine features")
        if self.radar_features != 16:
            raise ValueError("CUHK-X radar preprocessing emits sixteen frame statistics")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "CUHKXPreprocessingConfig":
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown CUHK-X preprocessing fields: {unknown}")
        return cls(**values)


def _sample_indices(
    length: int,
    count: int,
    *,
    random_seed: int | None = None,
) -> np.ndarray:
    if length <= 0:
        raise ValueError("cannot sample an empty sequence")
    if random_seed is not None:
        generator = np.random.default_rng(random_seed)
        boundaries = np.linspace(0, length, count + 1)
        indices = []
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            lower = min(int(math.floor(start)), length - 1)
            upper = min(max(int(math.ceil(stop)), lower + 1), length)
            indices.append(int(generator.integers(lower, upper)))
        return np.asarray(indices, dtype=np.int64)
    return np.rint(np.linspace(0, length - 1, count)).astype(np.int64)


def _zscore(values: np.ndarray, *, axis: int = 0) -> np.ndarray:
    mean = values.mean(axis=axis, keepdims=True)
    std = values.std(axis=axis, keepdims=True)
    return np.nan_to_num((values - mean) / np.maximum(std, 1e-6)).astype(np.float32)


@lru_cache(maxsize=16_384)
def _visual_frame_paths(folder: Path) -> tuple[Path, ...]:
    """Index one clip once per persistent data-loader worker."""

    extensions = {".png", ".jpg", ".jpeg"}
    return tuple(
        sorted(path for path in folder.rglob("*") if path.suffix.lower() in extensions)
    )


def load_visual_clip(
    folder: Path,
    *,
    channels: int,
    config: CUHKXPreprocessingConfig,
    augmentation_seed: int | None = None,
    horizontal_flip_probability: float = 0.0,
    crop_jitter: float = 0.0,
) -> torch.Tensor:
    if not 0.0 <= horizontal_flip_probability <= 1.0:
        raise ValueError("horizontal_flip_probability must be in [0, 1]")
    if not 0.0 <= crop_jitter <= 0.5:
        raise ValueError("crop_jitter must be in [0, 0.5]")
    paths = _visual_frame_paths(folder)
    if not paths:
        raise NoUsableVisualFramesError(f"no visual frames found in {folder}")
    frames = []
    mode = "L" if channels == 1 else "RGB"
    decoded: dict[int, np.ndarray] = {}
    invalid: set[int] = set()
    generator = np.random.default_rng(augmentation_seed)
    flip = (
        augmentation_seed is not None
        and generator.random() < horizontal_flip_probability
    )
    centering = (
        float(generator.uniform(0.5 - crop_jitter, 0.5 + crop_jitter)),
        float(generator.uniform(0.5 - crop_jitter, 0.5 + crop_jitter)),
    )

    def decode_nearest(target: int) -> np.ndarray:
        # Some official CUHK-X test clips contain zero-filled files with a PNG
        # suffix. Search symmetrically around the requested temporal position so
        # one bad frame does not invalidate an otherwise usable modality.
        candidates = sorted(
            range(len(paths)),
            key=lambda index: (abs(index - target), index),
        )
        for index in candidates:
            if index in decoded:
                return decoded[index]
            if index in invalid:
                continue
            try:
                with Image.open(paths[index]) as image:
                    image = ImageOps.fit(
                        image.convert(mode),
                        (config.image_size, config.image_size),
                        method=Image.Resampling.BILINEAR,
                        centering=centering,
                    )
                    if flip:
                        image = ImageOps.mirror(image)
                    array = np.array(image, dtype=np.float32, copy=True) / 255.0
            except (OSError, UnidentifiedImageError):
                invalid.add(index)
                continue
            decoded[index] = array
            return array
        raise NoUsableVisualFramesError(f"no decodable visual frames found in {folder}")

    for index in _sample_indices(
        len(paths),
        config.clip_frames,
        random_seed=augmentation_seed,
    ):
        array = decode_nearest(int(index))
        if channels == 1:
            array = array[..., None]
        frames.append(torch.from_numpy(array).permute(2, 0, 1))
    return (torch.stack(frames, dim=1) - 0.5) / 0.5


_IMU_DEVICE_ORDER = ("WTLA", "WTRA", "WTC", "WTLL", "WTRL")


def _resample_rows(
    rows: Sequence[Sequence[float]],
    count: int,
    width: int,
    *,
    standardize: bool = True,
) -> np.ndarray:
    if not rows:
        return np.zeros((count, width), dtype=np.float32)
    values = np.asarray(rows, dtype=np.float32)
    positions = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    targets = np.linspace(0.0, 1.0, count, dtype=np.float32)
    output = np.stack(
        [np.interp(targets, positions, values[:, column]) for column in range(width)],
        axis=1,
    )
    return _zscore(output) if standardize else output.astype(np.float32)


def load_imu_sequence(folder: Path, config: CUHKXPreprocessingConfig) -> torch.Tensor:
    by_device: dict[str, list[tuple[str, list[float]]]] = {
        name: [] for name in _IMU_DEVICE_ORDER
    }
    for path in sorted(folder.glob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                if len(row) < 11:
                    continue
                device = row[1].split("(", 1)[0].strip()
                if device not in by_device:
                    continue
                try:
                    # Acceleration, angular velocity, and Euler angles.
                    by_device[device].append(
                        (row[0].strip(), [float(value) for value in row[2:11]])
                    )
                except ValueError:
                    continue
    if not any(by_device.values()):
        raise NoUsableModalityError(f"no usable IMU rows found in {folder}")
    streams = []
    for name in _IMU_DEVICE_ORDER:
        ordered = [
            values
            for _, values in sorted(by_device[name], key=lambda item: item[0])
        ]
        stream = _resample_rows(
            ordered,
            config.sensor_steps,
            9,
            standardize=config.version == 1,
        )
        if config.version >= 2:
            # Acceleration (g), angular velocity (degree/s), and Euler angles
            # use fixed physical scales. Per-clip z-scoring erases posture and
            # movement magnitude, both of which discriminate HAR classes.
            scales = np.asarray(
                (4, 4, 4, 500, 500, 500, 180, 180, 180),
                dtype=np.float32,
            )
            stream = np.clip(stream / scales, -4.0, 4.0)
        streams.append(stream)
    return torch.from_numpy(np.concatenate(streams, axis=1))


def _radar_frame_features(rows: Sequence[Sequence[str]]) -> np.ndarray:
    points = []
    for row in rows:
        if len(row) < 9:
            continue
        try:
            points.append([float(value) for value in row[3:9]])
        except ValueError:
            continue
    if not points:
        return np.zeros(16, dtype=np.float32)
    values = np.asarray(points, dtype=np.float32)
    xyz = values[:, :3]
    count = np.asarray([math.log1p(len(points))], dtype=np.float32)
    summary = np.concatenate(
        (
            count,
            values.mean(axis=0),
            values.std(axis=0),
            np.asarray(
                [
                    values[:, 4].max(),
                    np.linalg.norm(xyz, axis=1).mean(),
                    np.abs(values[:, 3]).mean(),
                ],
                dtype=np.float32,
            ),
        )
    )
    return summary.astype(np.float32)


def load_radar_sequence(folder: Path, config: CUHKXPreprocessingConfig) -> torch.Tensor:
    frames: dict[str, list[list[str]]] = {}
    for path in sorted(folder.glob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    frames.setdefault(row[1], []).append(row)
    if not frames:
        raise NoUsableModalityError(f"no usable radar frames found in {folder}")
    keys = sorted(
        frames,
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    )
    values = np.stack([_radar_frame_features(frames[key]) for key in keys])
    values = values[_sample_indices(len(values), config.sensor_steps)]
    if config.version == 1:
        values = _zscore(values)
    else:
        scales = np.asarray(
            (6, 5, 5, 5, 5, 300, 1000, 5, 5, 5, 5, 300, 1000, 300, 5, 5),
            dtype=np.float32,
        )
        values = np.clip(values / scales, -4.0, 4.0).astype(np.float32)
    return torch.from_numpy(values)


def _first_pose(path: Path, joints: int) -> np.ndarray:
    try:
        with path.open(encoding="utf-8") as handle:
            people = json.load(handle)
    except (json.JSONDecodeError, OSError):
        people = []
    if not people or "keypoints" not in people[0]:
        return np.zeros((joints, 3), dtype=np.float32)
    values = np.asarray(people[0]["keypoints"], dtype=np.float32)
    output = np.zeros((joints, 3), dtype=np.float32)
    height = min(joints, values.shape[0])
    width = min(3, values.shape[1] if values.ndim == 2 else 0)
    if height and width:
        output[:height, :width] = values[:height, :width]
    return output


def load_skeleton_sequence(folder: Path, config: CUHKXPreprocessingConfig) -> torch.Tensor:
    paths = sorted(folder.rglob("*.json"))
    if not paths:
        raise NoUsableModalityError(f"no skeleton frames found in {folder}")
    poses = np.stack(
        [
            _first_pose(paths[int(index)], config.skeleton_joints)
            for index in _sample_indices(len(paths), config.sensor_steps)
        ]
    )
    if not np.any(poses):
        raise NoUsableModalityError(f"no usable skeleton poses found in {folder}")
    roots = poses[:, :1].copy()
    local_poses = poses - roots
    frame_scales = np.sqrt(np.square(local_poses).sum(axis=-1).mean(axis=1))
    if config.version >= 3:
        # Keep subject motion while removing camera-position and body-size bias.
        # The old per-frame root subtraction made walking, sitting, falling and
        # jumping trajectories indistinguishable at the root joint.
        valid_scales = frame_scales[frame_scales > 1e-6]
        clip_scale = float(np.median(valid_scales)) if valid_scales.size else 1.0
        root_trajectory = roots - roots[:1]
        poses = (local_poses + root_trajectory) / max(clip_scale, 1e-6)
    else:
        poses = local_poses / np.maximum(frame_scales[:, None, None], 1e-6)
    return torch.from_numpy(np.nan_to_num(poses).astype(np.float32))
