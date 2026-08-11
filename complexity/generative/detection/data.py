"""Datasets for training ``TRHashObjectDetector``.

Both datasets yield ``(pixel_values, targets)`` pairs where ``targets`` is a
``[N, 5]`` tensor of ``(cx, cy, w, h, class_id)`` normalized to ``[0, 1]`` --
directly consumable by ``TRHashObjectDetector.compute_loss``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance
from torch.utils.data import Dataset

SHAPE_CLASSES = ("rectangle", "ellipse", "triangle")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _normalize_image(image: Image.Image) -> torch.Tensor:
    pixels = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0
    return (pixels - 0.5) / 0.5


def _letterbox(
    image: Image.Image,
    boxes: torch.Tensor,
    image_size: int,
) -> Tuple[Image.Image, torch.Tensor]:
    """Resize without distortion and remap normalized ``cxcywh`` boxes."""

    original_w, original_h = image.size
    scale = min(image_size / original_w, image_size / original_h)
    resized_w = max(1, round(original_w * scale))
    resized_h = max(1, round(original_h * scale))
    left = (image_size - resized_w) // 2
    top = (image_size - resized_h) // 2
    resized = image.resize((resized_w, resized_h), Image.BILINEAR)
    canvas = Image.new("RGB", (image_size, image_size), (114, 114, 114))
    canvas.paste(resized, (left, top))

    if boxes.numel():
        boxes = boxes.clone()
        boxes[:, 0] = (boxes[:, 0] * original_w * scale + left) / image_size
        boxes[:, 1] = (boxes[:, 1] * original_h * scale + top) / image_size
        boxes[:, 2] = boxes[:, 2] * original_w * scale / image_size
        boxes[:, 3] = boxes[:, 3] * original_h * scale / image_size
    return canvas, boxes


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

    def __init__(self, annotations_path: Path, images_dir: Path, image_size: int = 224):
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        coco: Dict[str, Any] = json.loads(Path(annotations_path).read_text())

        self.images: Dict[int, Dict[str, Any]] = {item["id"]: item for item in coco["images"]}
        self.image_ids: List[int] = list(self.images.keys())

        category_ids = sorted({annotation["category_id"] for annotation in coco["annotations"]})
        self.category_to_class = {category_id: index for index, category_id in enumerate(category_ids)}
        self.num_classes = len(category_ids)

        self.annotations_by_image: Dict[int, List[Dict[str, Any]]] = {
            image_id: [] for image_id in self.image_ids
        }
        for annotation in coco["annotations"]:
            self.annotations_by_image[annotation["image_id"]].append(annotation)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_id = self.image_ids[index]
        meta = self.images[image_id]
        image = Image.open(self.images_dir / meta["file_name"]).convert("RGB")
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
        image, targets = _letterbox(image, targets, self.image_size)
        pixel_values = _normalize_image(image)
        return pixel_values, targets


class YoloDetectionDataset(Dataset):
    """Ultralytics-style image/label folders with letterbox and light augmentation."""

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        image_size: int = 224,
        *,
        augment: bool = False,
        seed: int = 0,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.image_size = image_size
        self.augment = augment
        self.seed = seed
        self.epoch = 0
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
        self.epoch = epoch

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

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.image_paths[index]).convert("RGB")
        targets = self._load_targets(self.label_paths[index])
        rng = random.Random(self.seed + self.epoch * len(self) + index)
        if self.augment:
            if rng.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if targets.numel():
                    targets[:, 0] = 1.0 - targets[:, 0]
            image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.8, 1.2))
            image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.8, 1.2))
            image = ImageEnhance.Color(image).enhance(rng.uniform(0.8, 1.2))
        image, targets = _letterbox(image, targets, self.image_size)
        return _normalize_image(image), targets


def collate_detection(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    pixel_values = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return pixel_values, targets
