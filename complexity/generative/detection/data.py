"""Datasets for training ``TRHashObjectDetector``.

Both datasets yield ``(pixel_values, targets)`` pairs where ``targets`` is a
``[N, 5]`` tensor of ``(cx, cy, w, h, class_id)`` normalized to ``[0, 1]`` --
directly consumable by ``TRHashObjectDetector.compute_loss``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, IterableDataset

from .augmentations import DetectionAugmenter, letterbox
from .image_io import IMAGE_BACKENDS, load_rgb_image

SHAPE_CLASSES = ("rectangle", "ellipse", "triangle")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _bound_and_split_stream(
    stream: Any,
    *,
    local_examples: int,
    rank: int,
    world_size: int,
    splitter: Any,
) -> Any:
    """Bound a lazy HF stream globally before attaching its rank split."""

    bounded_stream = stream.take(local_examples * world_size)
    return splitter(bounded_stream, rank=rank, world_size=world_size)


def _normalize_image(image: Image.Image) -> torch.Tensor:
    pixels = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0
    return (pixels - 0.5) / 0.5


class SyntheticShapesDataset(Dataset):
    """Procedurally generated shape-detection task: no external data required.

    Each sample draws 1-3 random shapes (rectangle/ellipse/triangle, one class
    per shape type) at random positions on a plain background. The dataset is
    seeded so a given index always yields the same image across epochs.
    """

    def __init__(
        self,
        length: int,
        image_size: int = 224,
        seed: int = 0,
        *,
        resample_each_epoch: bool = False,
    ):
        self.length = length
        self.image_size = image_size
        self.seed = seed
        self.resample_each_epoch = resample_each_epoch
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Select a deterministic fresh sample stream for a training epoch."""

        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        epoch_offset = self.epoch * self.length if self.resample_each_epoch else 0
        rng = random.Random(self.seed + epoch_offset + index)
        size = self.image_size
        image = Image.new("RGB", (size, size), color=(230, 230, 230))
        draw = ImageDraw.Draw(image)

        num_shapes = rng.randint(1, 3)
        boxes: List[List[float]] = []
        for _ in range(num_shapes):
            class_id = rng.randrange(len(SHAPE_CLASSES))
            w = rng.uniform(0.15, 0.4) * size
            h = rng.uniform(0.15, 0.4) * size
            cx = rng.uniform(w / 2 + 2, size - w / 2 - 2)
            cy = rng.uniform(h / 2 + 2, size - h / 2 - 2)
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            color = tuple(rng.randrange(40, 220) for _ in range(3))

            shape = SHAPE_CLASSES[class_id]
            if shape == "rectangle":
                draw.rectangle([x1, y1, x2, y2], fill=color)
            elif shape == "ellipse":
                draw.ellipse([x1, y1, x2, y2], fill=color)
            else:
                draw.polygon([(cx, y1), (x1, y2), (x2, y2)], fill=color)

            boxes.append([cx / size, cy / size, w / size, h / size, float(class_id)])

        pixel_values = _normalize_image(image)
        targets = torch.tensor(boxes, dtype=torch.float32)
        return pixel_values, targets


class CocoDetectionDataset(Dataset):
    """Minimal COCO-format loader.

    Expects a COCO-style JSON with ``images`` (``id``, ``file_name``) and
    ``annotations`` (``image_id``, ``bbox`` as ``[x, y, w, h]`` in pixels,
    ``category_id``). Category ids are remapped to contiguous
    ``0..num_classes-1`` indices in first-seen order. Images are resized
    with a letterbox transform that preserves aspect ratio.
    """

    def __init__(
        self,
        annotations_path: Path,
        images_dir: Path,
        image_size: int = 224,
        *,
        augmentation: str | None = None,
        augmentation_backend: str = "native",
        image_backend: str = "pillow",
        return_metadata: bool = False,
        seed: int = 0,
        mosaic_probability: float = 0.0,
        mixup_probability: float = 0.0,
        copy_paste_probability: float = 0.0,
        random_erasing_probability: float = 0.0,
        total_epochs: int = 0,
        close_mosaic_epochs: int = 0,
        mosaic_tiles: int = 4,
        mosaic_canvas_size: int = 0,
    ):
        if image_backend not in IMAGE_BACKENDS:
            raise ValueError(f"unsupported image backend: {image_backend}")
        if return_metadata and (
            augmentation is not None
            or any(
                (
                    mosaic_probability,
                    mixup_probability,
                    copy_paste_probability,
                    random_erasing_probability,
                )
            )
        ):
            raise ValueError("metadata-returning COCO datasets cannot apply augmentation")
        self.annotations_path = Path(annotations_path)
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.image_backend = image_backend
        self.return_metadata = return_metadata
        self.augmenter = DetectionAugmenter(
            image_size,
            mode=augmentation,
            backend=augmentation_backend,
            seed=seed,
            mosaic_probability=mosaic_probability,
            mixup_probability=mixup_probability,
            copy_paste_probability=copy_paste_probability,
            random_erasing_probability=random_erasing_probability,
            total_epochs=total_epochs,
            close_mosaic_epochs=close_mosaic_epochs,
            mosaic_tiles=mosaic_tiles,
            mosaic_canvas_size=mosaic_canvas_size,
        )
        coco: Dict[str, Any] = json.loads(self.annotations_path.read_text())

        self.images: Dict[int, Dict[str, Any]] = {item["id"]: item for item in coco["images"]}
        self.image_ids: List[int] = list(self.images.keys())

        category_ids = sorted(
            {int(category["id"]) for category in coco.get("categories", [])}
            or {int(annotation["category_id"]) for annotation in coco["annotations"]}
        )
        self.category_to_class = {
            category_id: index for index, category_id in enumerate(category_ids)
        }
        self.class_to_category = tuple(category_ids)
        self.num_classes = len(category_ids)

        self.annotations_by_image: Dict[int, List[Dict[str, Any]]] = {
            image_id: [] for image_id in self.image_ids
        }
        for annotation in coco["annotations"]:
            self.annotations_by_image[annotation["image_id"]].append(annotation)

    def __len__(self) -> int:
        return len(self.image_ids)

    def set_epoch(self, epoch: int) -> None:
        self.augmenter.set_epoch(epoch)

    def _load_raw(self, index: int) -> tuple[Image.Image, torch.Tensor]:
        image_id = self.image_ids[index]
        meta = self.images[image_id]
        image = load_rgb_image(self.images_dir / meta["file_name"], self.image_backend)
        original_w, original_h = image.size

        boxes: List[List[float]] = []
        for annotation in self.annotations_by_image[image_id]:
            x, y, w, h = annotation["bbox"]
            cx = (x + w / 2) / original_w
            cy = (y + h / 2) / original_h
            nw = w / original_w
            nh = h / original_h
            class_id = self.category_to_class[annotation["category_id"]]
            boxes.append([cx, cy, nw, nh, float(class_id)])

        targets = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty(0, 5)
        return image, targets

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        if self.return_metadata:
            image, targets = self._load_raw(index)
            original_width, original_height = image.size
            scale = min(self.image_size / original_width, self.image_size / original_height)
            resized_width = max(1, round(original_width * scale))
            resized_height = max(1, round(original_height * scale))
            image, targets = letterbox(image, targets, self.image_size)
            metadata = {
                "image_id": self.image_ids[index],
                "original_width": original_width,
                "original_height": original_height,
                "image_size": self.image_size,
                "scale": scale,
                "left": (self.image_size - resized_width) // 2,
                "top": (self.image_size - resized_height) // 2,
            }
            return _normalize_image(image), targets, metadata
        image, targets = self.augmenter(index, len(self), self._load_raw)
        return _normalize_image(image), targets


class HuggingFaceDetectionDataset(IterableDataset):
    """Stream image detection rows from a Hugging Face parquet dataset.

    The stream is deterministically shuffled per epoch and split across DDP
    ranks before decoding. This avoids materializing large detection corpora
    twice (download archive plus extracted images) while keeping an exact,
    fixed number of examples on every rank.
    """

    def __init__(
        self,
        dataset_id: str,
        split: str,
        *,
        num_examples: int,
        num_classes: int,
        image_size: int,
        rank: int = 0,
        world_size: int = 1,
        image_column: str = "image",
        annotations_column: str = "annotations",
        metadata_file_glob: str | None = None,
        category_id_offset: int = 1,
        augmentation: str | None = None,
        seed: int = 0,
        shuffle_buffer: int = 10_000,
    ):
        super().__init__()
        if num_examples < world_size:
            raise ValueError("HF detection split is smaller than the DDP world size")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be within the DDP world size")
        if augmentation not in {None, "light", "strong"}:
            raise ValueError("augmentation must be light, strong, or None")
        if shuffle_buffer <= 0:
            raise ValueError("shuffle_buffer must be positive")
        self.dataset_id = dataset_id
        self.split = split
        self.total_examples = num_examples
        self.local_examples = num_examples // world_size
        self.num_classes = num_classes
        self.image_size = image_size
        self.rank = rank
        self.world_size = world_size
        self.image_column = image_column
        self.annotations_column = annotations_column
        self.metadata_file_glob = metadata_file_glob
        self.category_id_offset = category_id_offset
        self.augmentation = augmentation
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self.epoch = 0

    def __len__(self) -> int:
        return self.local_examples

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def _repository_file_uri(self, file_glob: str) -> str:
        root = Path(self.dataset_id)
        if root.is_dir():
            return str(root / file_glob)
        return f"hf://datasets/{self.dataset_id}/{file_glob.lstrip('/')}"

    def _stream(self, *, metadata_only: bool = False):
        try:
            from datasets import Image as HFImage
            from datasets import load_dataset
            from datasets.distributed import split_dataset_by_node
        except ImportError as error:  # pragma: no cover - optional dependency guard
            raise RuntimeError(
                "Hugging Face detection streaming requires `pip install datasets`"
            ) from error
        if metadata_only:
            if self.metadata_file_glob:
                parquet_uri = self._repository_file_uri(self.metadata_file_glob)
                return load_dataset(
                    "parquet",
                    data_files=[parquet_uri],
                    split="train",
                    streaming=True,
                    columns=["width", "height", self.annotations_column],
                ).take(self.total_examples)
            stream = load_dataset(self.dataset_id, split=self.split, streaming=True)
            stream = stream.cast_column(self.image_column, HFImage(decode=False))
            stream = stream.remove_columns(self.image_column)
            return stream.take(self.total_examples)
        root = Path(self.dataset_id)
        if root.is_dir():
            stream = load_dataset(
                "parquet",
                data_files=[self._repository_file_uri(f"data/{self.split}-*.parquet")],
                split="train",
                streaming=True,
            )
        else:
            stream = load_dataset(self.dataset_id, split=self.split, streaming=True)
        stream = stream.shuffle(
            seed=self.seed + self.epoch,
            buffer_size=self.shuffle_buffer,
        )
        # Hugging Face applies the distributed split lazily at iteration time.
        # Calling ``take(local_examples)`` after attaching that split still
        # limits the *global* stream first, leaving only local/world examples
        # on each rank. Bound the global stream to an exactly divisible size,
        # then attach the rank split so every rank yields ``local_examples``.
        return _bound_and_split_stream(
            stream,
            local_examples=self.local_examples,
            rank=self.rank,
            world_size=self.world_size,
            splitter=split_dataset_by_node,
        )

    def _targets(self, row: Dict[str, Any]) -> torch.Tensor:
        annotations = row[self.annotations_column]
        if isinstance(annotations, str):
            annotations = json.loads(annotations)
        width = float(row["width"])
        height = float(row["height"])
        boxes = []
        for annotation in annotations:
            x, y, box_width, box_height = map(float, annotation["bbox"])
            x1 = min(max(x, 0.0), width)
            y1 = min(max(y, 0.0), height)
            x2 = min(max(x + box_width, 0.0), width)
            y2 = min(max(y + box_height, 0.0), height)
            if x2 <= x1 or y2 <= y1:
                continue
            class_id = int(annotation["category_id"]) - self.category_id_offset
            if not 0 <= class_id < self.num_classes:
                raise ValueError(f"category ID maps outside configured classes: {class_id}")
            boxes.append(
                (
                    ((x1 + x2) * 0.5) / width,
                    ((y1 + y2) * 0.5) / height,
                    (x2 - x1) / width,
                    (y2 - y1) / height,
                    float(class_id),
                )
            )
        return torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty(0, 5)

    def _augment(
        self,
        image: Image.Image,
        targets: torch.Tensor,
        sample_index: int,
    ) -> tuple[Image.Image, torch.Tensor]:
        if self.augmentation is None:
            return image, targets
        rng = random.Random(
            self.seed + self.epoch * self.total_examples + self.rank + sample_index * self.world_size
        )
        if rng.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if targets.numel():
                targets = targets.clone()
                targets[:, 0] = 1.0 - targets[:, 0]
        jitter = (0.65, 1.35) if self.augmentation == "strong" else (0.8, 1.2)
        image = ImageEnhance.Color(
            ImageEnhance.Contrast(
                ImageEnhance.Brightness(image).enhance(rng.uniform(*jitter))
            ).enhance(rng.uniform(*jitter))
        ).enhance(rng.uniform(*jitter))
        if self.augmentation == "strong" and rng.random() < 0.1:
            image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.1, 1.2)))
        return image, targets

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        worker = torch.utils.data.get_worker_info()
        if worker is not None:
            raise RuntimeError("HF detection streaming currently requires --workers 0")
        for index, row in enumerate(self._stream()):
            image = row[self.image_column].convert("RGB")
            targets = self._targets(row)
            image, targets = self._augment(image, targets, index)
            image, targets = letterbox(image, targets, self.image_size)
            yield _normalize_image(image), targets

    def object_bucket_counts(self) -> torch.Tensor:
        """Count class x size x density buckets without decoding image bytes."""

        counts = torch.zeros(self.num_classes, 3, 3, dtype=torch.float64)
        for row in self._stream(metadata_only=True):
            targets = self._targets(row)
            if not len(targets):
                continue
            density_bin = int(
                torch.bucketize(torch.tensor(len(targets)), torch.tensor((3, 10)))
            )
            size_bins = torch.bucketize(
                targets[:, 2:4].prod(dim=-1),
                torch.tensor((0.02, 0.15)),
            )
            classes = targets[:, 4].long()
            density_bins = torch.full_like(classes, density_bin)
            counts.index_put_(
                (classes, size_bins, density_bins),
                torch.ones_like(classes, dtype=counts.dtype),
                accumulate=True,
            )
        return counts


class CocoVideoDetectionDataset(Dataset):
    """COCO-Video loader returning synchronized clips around each key frame.

    The annotation file follows COCO detection and adds ``video_id`` plus
    ``frame_id`` to each image record. Targets belong to the center frame.
    Temporal neighbors are clamped at sequence boundaries, so every sample has
    a stable ``[T, 3, H, W]`` shape.
    """

    def __init__(
        self,
        annotations_path: Path,
        images_dir: Path,
        image_size: int = 224,
        *,
        clip_frames: int = 5,
        frame_stride: int = 1,
        augmentation: str | None = None,
        seed: int = 0,
    ):
        if clip_frames < 3 or clip_frames % 2 == 0:
            raise ValueError("clip_frames must be an odd integer of at least 3")
        if frame_stride <= 0:
            raise ValueError("frame_stride must be positive")
        if augmentation not in {None, "light", "strong"}:
            raise ValueError("augmentation must be light, strong, or None")
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.clip_frames = clip_frames
        self.frame_stride = frame_stride
        self.augmentation = augmentation
        self.seed = seed
        self.epoch = 0

        coco: Dict[str, Any] = json.loads(Path(annotations_path).read_text())
        image_records = coco["images"]
        if any("video_id" not in item for item in image_records):
            raise ValueError("COCO-Video images require video_id")
        if any("frame_id" not in item for item in image_records):
            raise ValueError("COCO-Video images require frame_id")

        self.images: Dict[int, Dict[str, Any]] = {
            int(item["id"]): item for item in image_records
        }
        self.image_ids: List[int] = [int(item["id"]) for item in image_records]
        category_ids = sorted(
            {annotation["category_id"] for annotation in coco["annotations"]}
        )
        self.category_to_class = {
            category_id: index for index, category_id in enumerate(category_ids)
        }
        self.num_classes = len(category_ids)
        self.annotations_by_image: Dict[int, List[Dict[str, Any]]] = {
            image_id: [] for image_id in self.image_ids
        }
        for annotation in coco["annotations"]:
            self.annotations_by_image[int(annotation["image_id"])].append(annotation)

        video_frames: Dict[int, List[int]] = {}
        for image_id in self.image_ids:
            video_id = int(self.images[image_id]["video_id"])
            video_frames.setdefault(video_id, []).append(image_id)
        self.video_frames = {
            video_id: sorted(
                frame_ids,
                key=lambda image_id: int(self.images[image_id]["frame_id"]),
            )
            for video_id, frame_ids in video_frames.items()
        }
        self.frame_positions = {
            image_id: position
            for frame_ids in self.video_frames.values()
            for position, image_id in enumerate(frame_ids)
        }

    def __len__(self) -> int:
        return len(self.image_ids)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def _target_tensor(self, image_id: int) -> torch.Tensor:
        meta = self.images[image_id]
        width = float(meta["width"])
        height = float(meta["height"])
        boxes = []
        for annotation in self.annotations_by_image[image_id]:
            x, y, box_width, box_height = annotation["bbox"]
            boxes.append(
                (
                    (x + box_width / 2) / width,
                    (y + box_height / 2) / height,
                    box_width / width,
                    box_height / height,
                    float(self.category_to_class[annotation["category_id"]]),
                )
            )
        return torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty(0, 5)

    def _clip_image_ids(self, center_id: int) -> List[int]:
        meta = self.images[center_id]
        frames = self.video_frames[int(meta["video_id"])]
        center = self.frame_positions[center_id]
        radius = self.clip_frames // 2
        return [
            frames[min(max(center + offset * self.frame_stride, 0), len(frames) - 1)]
            for offset in range(-radius, radius + 1)
        ]

    def _synchronized_augmentation(
        self,
        frames: List[Image.Image],
        targets: torch.Tensor,
        index: int,
    ) -> tuple[List[Image.Image], torch.Tensor]:
        if self.augmentation is None:
            return frames, targets
        rng = random.Random(self.seed + self.epoch * len(self) + index)
        if rng.random() < 0.5:
            frames = [frame.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for frame in frames]
            if targets.numel():
                targets = targets.clone()
                targets[:, 0] = 1.0 - targets[:, 0]
        jitter = (0.65, 1.35) if self.augmentation == "strong" else (0.8, 1.2)
        brightness = rng.uniform(*jitter)
        contrast = rng.uniform(*jitter)
        color = rng.uniform(*jitter)
        frames = [
            ImageEnhance.Color(
                ImageEnhance.Contrast(
                    ImageEnhance.Brightness(frame).enhance(brightness)
                ).enhance(contrast)
            ).enhance(color)
            for frame in frames
        ]
        if self.augmentation == "strong" and rng.random() < 0.1:
            frames = [ImageEnhance.Color(frame).enhance(0.0) for frame in frames]
        if self.augmentation == "strong" and rng.random() < 0.1:
            radius = rng.uniform(0.1, 1.2)
            frames = [frame.filter(ImageFilter.GaussianBlur(radius)) for frame in frames]
        return frames, targets

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        center_id = self.image_ids[index]
        frames = [
            Image.open(self.images_dir / self.images[image_id]["file_name"]).convert("RGB")
            for image_id in self._clip_image_ids(center_id)
        ]
        targets = self._target_tensor(center_id)
        frames, targets = self._synchronized_augmentation(frames, targets, index)
        center = len(frames) // 2
        processed = []
        for frame_index, frame in enumerate(frames):
            frame_targets = targets if frame_index == center else torch.empty(0, 5)
            frame, transformed = letterbox(frame, frame_targets, self.image_size)
            if frame_index == center:
                targets = transformed
            processed.append(_normalize_image(frame))
        return torch.stack(processed), targets


class YoloDetectionDataset(Dataset):
    """YOLO-format image/label folders with letterbox and light augmentation."""

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        image_size: int = 224,
        *,
        augmentation: str | None = None,
        augmentation_backend: str = "native",
        image_backend: str = "pillow",
        seed: int = 0,
        mosaic_probability: float = 0.0,
        mixup_probability: float = 0.0,
        copy_paste_probability: float = 0.0,
        random_erasing_probability: float = 0.0,
        total_epochs: int = 0,
        close_mosaic_epochs: int = 0,
        mosaic_tiles: int = 4,
        mosaic_canvas_size: int = 0,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.image_size = image_size
        if image_backend not in IMAGE_BACKENDS:
            raise ValueError(f"unsupported image backend: {image_backend}")
        self.image_backend = image_backend
        self.augmenter = DetectionAugmenter(
            image_size,
            mode=augmentation,
            backend=augmentation_backend,
            seed=seed,
            mosaic_probability=mosaic_probability,
            mixup_probability=mixup_probability,
            copy_paste_probability=copy_paste_probability,
            random_erasing_probability=random_erasing_probability,
            total_epochs=total_epochs,
            close_mosaic_epochs=close_mosaic_epochs,
            mosaic_tiles=mosaic_tiles,
            mosaic_canvas_size=mosaic_canvas_size,
        )
        self.image_paths = sorted(
            path
            for path in self.images_dir.rglob("*")
            if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.image_paths:
            raise ValueError(f"no images found in {self.images_dir}")
        self.label_paths = [
            self.labels_dir / path.relative_to(self.images_dir).with_suffix(".txt")
            for path in self.image_paths
        ]
        class_ids = set()
        for label_path in self.label_paths:
            if not label_path.exists():
                continue
            for line in label_path.read_text().splitlines():
                if line.strip():
                    class_ids.add(int(line.split()[0]))
        self.num_classes = max(class_ids, default=-1) + 1

    def set_epoch(self, epoch: int) -> None:
        self.augmenter.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_targets(self, path: Path) -> torch.Tensor:
        if not path.exists():
            return torch.empty(0, 5)
        boxes = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            class_id, cx, cy, width, height = map(float, line.split()[:5])
            boxes.append((cx, cy, width, height, class_id))
        return torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty(0, 5)

    def _load_raw(self, index: int) -> tuple[Image.Image, torch.Tensor]:
        return (
            load_rgb_image(self.image_paths[index], self.image_backend),
            self._load_targets(self.label_paths[index]),
        )

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, targets = self.augmenter(index, len(self), self._load_raw)
        return _normalize_image(image), targets


def collate_detection(batch: List[tuple[Any, ...]]) -> tuple[Any, ...]:
    pixel_values = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    if len(batch[0]) == 3:
        return pixel_values, targets, [item[2] for item in batch]
    return pixel_values, targets
