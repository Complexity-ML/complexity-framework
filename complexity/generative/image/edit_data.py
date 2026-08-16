"""WebDataset triplet reader for instruction-guided image editing."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

PathLike = Union[str, Path]
_IMAGE_SUFFIXES = ("webp", "jpg", "jpeg", "png")


def image_payload_to_tensor(payload: bytes, image_size: int) -> torch.Tensor:
    with Image.open(io.BytesIO(payload)) as image:
        image = image.convert("RGB")
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).div_(127.5).sub_(1.0)


def _member_parts(name: str) -> tuple[str, str] | None:
    """Return ``(sample_id, role)`` for the documented triplet schema."""

    for role in ("source", "target"):
        for suffix in _IMAGE_SUFFIXES:
            marker = f".{role}.{suffix}"
            if name.endswith(marker):
                return name[: -len(marker)], role
    if name.endswith(".txt"):
        return name[:-4], "instruction"
    if name.endswith(".json"):
        return name[:-5], "metadata"
    return None


class AtlasImageEditTarDataset(IterableDataset):
    """Read ``source image + instruction + target image`` TAR samples.

    Each sample uses four members with one shared identifier::

        000001.source.webp
        000001.target.webp
        000001.txt
        000001.json
    """

    def __init__(
        self,
        shards: Sequence[PathLike],
        image_size: int = 256,
        *,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.shards = tuple(Path(path) for path in shards)
        self.image_size = int(image_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        if not self.shards:
            raise ValueError("at least one edit shard is required")
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be in [0, world_size)")

    def _assigned_shards(self, worker_id: int, worker_count: int) -> tuple[Path, ...]:
        rank_shards = self.shards[self.rank :: self.world_size]
        return rank_shards[worker_id::worker_count]

    def _worker_shards(self) -> Iterable[Path]:
        worker = get_worker_info()
        if worker is None:
            worker_id, worker_count = 0, 1
        else:
            worker_id, worker_count = worker.id, worker.num_workers
        return self._assigned_shards(worker_id, worker_count)

    def __iter__(self) -> Iterator[Dict[str, object]]:
        for shard in self._worker_shards():
            if not shard.is_file():
                raise FileNotFoundError(shard)
            pending: Dict[str, Dict[str, bytes]] = {}
            with tarfile.open(shard, "r:*") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    parts = _member_parts(member.name)
                    if parts is None:
                        continue
                    sample_id, role = parts
                    stream = archive.extractfile(member)
                    if stream is None:
                        continue
                    row = pending.setdefault(sample_id, {})
                    row[role] = stream.read()
                    if {"source", "target", "instruction", "metadata"} <= row.keys():
                        yield {
                            "sample_id": sample_id,
                            "source_pixel_values": image_payload_to_tensor(
                                row["source"], self.image_size
                            ),
                            "target_pixel_values": image_payload_to_tensor(
                                row["target"], self.image_size
                            ),
                            "instruction": row["instruction"].decode("utf-8").strip(),
                            "metadata": json.loads(row["metadata"]),
                        }
                        del pending[sample_id]


def collate_atlas_image_edits(rows: List[Dict[str, object]]) -> Dict[str, object]:
    if not rows:
        raise ValueError("cannot collate an empty edit batch")
    return {
        "source_pixel_values": torch.stack([row["source_pixel_values"] for row in rows]),
        "target_pixel_values": torch.stack([row["target_pixel_values"] for row in rows]),
        "instructions": [str(row["instruction"]) for row in rows],
        "sample_ids": [str(row["sample_id"]) for row in rows],
        "metadata": [row["metadata"] for row in rows],
    }
