"""Streaming reader for Complexity Atlas Images WebDataset shards."""

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


def _image_tensor(payload: bytes, image_size: int) -> torch.Tensor:
    with Image.open(io.BytesIO(payload)) as image:
        image = image.convert("RGB")
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).div_(127.5).sub_(1.0)


class AtlasImageTarDataset(IterableDataset):
    """Read ``.webp + .txt + .json`` samples without extracting TAR shards."""

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
            raise ValueError("at least one image shard is required")
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be in [0, world_size)")

    def _assigned_shards(self, worker_id: int, worker_count: int) -> tuple[Path, ...]:
        """Partition ranks first so DataLoader workers cannot starve a DDP rank."""

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
                    if not member.isfile() or "." not in member.name:
                        continue
                    sample_id, suffix = member.name.rsplit(".", 1)
                    if suffix not in {"webp", "jpg", "jpeg", "png", "txt", "json"}:
                        continue
                    stream = archive.extractfile(member)
                    if stream is None:
                        continue
                    row = pending.setdefault(sample_id, {})
                    row[suffix] = stream.read()
                    image_suffix = next(
                        (name for name in ("webp", "jpg", "jpeg", "png") if name in row),
                        None,
                    )
                    if image_suffix and "txt" in row and "json" in row:
                        yield {
                            "sample_id": sample_id,
                            "pixel_values": _image_tensor(row[image_suffix], self.image_size),
                            "caption": row["txt"].decode("utf-8").strip(),
                            "metadata": json.loads(row["json"]),
                        }
                        del pending[sample_id]


def collate_atlas_images(rows: List[Dict[str, object]]) -> Dict[str, object]:
    if not rows:
        raise ValueError("cannot collate an empty image batch")
    return {
        "pixel_values": torch.stack([row["pixel_values"] for row in rows]),
        "captions": [str(row["caption"]) for row in rows],
        "sample_ids": [str(row["sample_id"]) for row in rows],
        "metadata": [row["metadata"] for row in rows],
    }
