"""Composable box-aware augmentations shared by COCO and YOLO datasets."""

from __future__ import annotations

import math
import random
from collections.abc import Callable

import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

RawLoader = Callable[[int], tuple[Image.Image, torch.Tensor]]


def random_scale_translate(
    image: Image.Image,
    targets: torch.Tensor,
    rng: random.Random,
) -> tuple[Image.Image, torch.Tensor]:
    width, height = image.size
    scale = rng.uniform(0.75, 1.25)
    resized_w = max(1, round(width * scale))
    resized_h = max(1, round(height * scale))
    resized = image.resize((resized_w, resized_h), Image.BILINEAR)
    canvas = Image.new("RGB", (width, height), (114, 114, 114))
    paste_x = (
        -rng.randint(0, resized_w - width)
        if resized_w >= width
        else rng.randint(0, width - resized_w)
    )
    paste_y = (
        -rng.randint(0, resized_h - height)
        if resized_h >= height
        else rng.randint(0, height - resized_h)
    )
    source_x = max(-paste_x, 0)
    source_y = max(-paste_y, 0)
    crop = resized.crop(
        (
            source_x,
            source_y,
            min(source_x + width, resized_w),
            min(source_y + height, resized_h),
        )
    )
    canvas.paste(crop, (max(paste_x, 0), max(paste_y, 0)))
    if not targets.numel():
        return canvas, targets
    classes = targets[:, 4:]
    dimensions = targets.new_tensor((width, height))
    centers = targets[:, :2] * dimensions
    sizes = targets[:, 2:4] * dimensions
    top_left = (centers - sizes / 2) * scale + targets.new_tensor((paste_x, paste_y))
    bottom_right = (centers + sizes / 2) * scale + targets.new_tensor((paste_x, paste_y))
    top_left = torch.maximum(top_left, targets.new_zeros(2))
    bottom_right = torch.minimum(bottom_right, dimensions)
    sizes = bottom_right - top_left
    keep = (sizes[:, 0] >= 2.0) & (sizes[:, 1] >= 2.0)
    remapped = torch.cat(
        (((top_left + bottom_right) / 2) / dimensions, sizes / dimensions, classes),
        dim=-1,
    )
    return canvas, remapped[keep]


def letterbox(
    image: Image.Image,
    boxes: torch.Tensor,
    image_size: int,
) -> tuple[Image.Image, torch.Tensor]:
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


class DetectionAugmenter:
    def __init__(
        self,
        image_size: int,
        *,
        mode: str | None,
        seed: int,
        mosaic_probability: float,
        mixup_probability: float,
        copy_paste_probability: float,
        random_erasing_probability: float,
        total_epochs: int,
        close_mosaic_epochs: int,
    ):
        if mode not in {None, "light", "strong"}:
            raise ValueError("augmentation must be light, strong, or None")
        probabilities = (
            mosaic_probability,
            mixup_probability,
            copy_paste_probability,
            random_erasing_probability,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("augmentation probabilities must be in [0, 1]")
        if total_epochs < 0 or close_mosaic_epochs < 0:
            raise ValueError("augmentation epoch counts must be non-negative")
        self.image_size = image_size
        self.mode = mode
        self.seed = seed
        self.mosaic_probability = float(mosaic_probability)
        self.mixup_probability = float(mixup_probability)
        self.copy_paste_probability = float(copy_paste_probability)
        self.random_erasing_probability = float(random_erasing_probability)
        self.total_epochs = int(total_epochs)
        self.close_mosaic_epochs = int(close_mosaic_epochs)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _mosaic_enabled(self) -> bool:
        if self.mosaic_probability <= 0.0:
            return False
        if not self.total_epochs or not self.close_mosaic_epochs:
            return True
        return self.epoch < max(self.total_epochs - self.close_mosaic_epochs, 0)

    def _mosaic(
        self,
        index: int,
        length: int,
        load_raw: RawLoader,
        rng: random.Random,
    ) -> tuple[Image.Image, torch.Tensor]:
        tile_size = self.image_size // 2
        canvas = Image.new("RGB", (self.image_size, self.image_size), (114, 114, 114))
        indices = [index] + [rng.randrange(length) for _ in range(3)]
        all_targets = []
        for tile_index, source_index in enumerate(indices):
            image, targets = load_raw(source_index)
            image, targets = letterbox(image, targets, tile_size)
            col, row = tile_index % 2, tile_index // 2
            canvas.paste(image, (col * tile_size, row * tile_size))
            if targets.numel():
                targets = targets.clone()
                targets[:, 0] = (targets[:, 0] + col) / 2
                targets[:, 1] = (targets[:, 1] + row) / 2
                targets[:, 2:4] /= 2
                all_targets.append(targets)
        return canvas, torch.cat(all_targets) if all_targets else torch.empty(0, 5)

    def _mixup(self, image, targets, length, load_raw, rng):
        donor, donor_targets = load_raw(rng.randrange(length))
        donor, donor_targets = letterbox(donor, donor_targets, self.image_size)
        image = Image.blend(image, donor, rng.uniform(0.35, 0.65))
        if donor_targets.numel():
            targets = torch.cat((targets, donor_targets)) if targets.numel() else donor_targets
        return image, targets

    def _copy_paste(self, image, targets, length, load_raw, rng):
        donor, donor_targets = load_raw(rng.randrange(length))
        donor, donor_targets = letterbox(donor, donor_targets, self.image_size)
        selected = []
        for target in donor_targets:
            if rng.random() >= 0.5:
                continue
            cx, cy, width, height = target[:4].tolist()
            x1 = max(0, round((cx - width / 2) * self.image_size))
            y1 = max(0, round((cy - height / 2) * self.image_size))
            x2 = min(self.image_size, round((cx + width / 2) * self.image_size))
            y2 = min(self.image_size, round((cy + height / 2) * self.image_size))
            if x2 > x1 and y2 > y1:
                image.paste(donor.crop((x1, y1, x2, y2)), (x1, y1))
                selected.append(target)
        if selected:
            pasted = torch.stack(selected)
            targets = torch.cat((targets, pasted)) if targets.numel() else pasted
        return image, targets

    def _erase(self, image: Image.Image, rng: random.Random) -> Image.Image:
        area = self.image_size**2 * rng.uniform(0.02, 0.12)
        aspect = rng.uniform(0.4, 2.5)
        width = min(self.image_size, max(1, round(math.sqrt(area * aspect))))
        height = min(self.image_size, max(1, round(math.sqrt(area / aspect))))
        left = rng.randint(0, self.image_size - width)
        top = rng.randint(0, self.image_size - height)
        fill = tuple(rng.randint(80, 160) for _ in range(3))
        ImageDraw.Draw(image).rectangle((left, top, left + width, top + height), fill=fill)
        return image

    def __call__(
        self,
        index: int,
        length: int,
        load_raw: RawLoader,
    ) -> tuple[Image.Image, torch.Tensor]:
        rng = random.Random(self.seed + self.epoch * length + index)
        if self._mosaic_enabled() and rng.random() < self.mosaic_probability:
            image, targets = self._mosaic(index, length, load_raw, rng)
        else:
            image, targets = load_raw(index)
        if self.mode:
            if rng.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if targets.numel():
                    targets[:, 0] = 1.0 - targets[:, 0]
            jitter = (0.65, 1.35) if self.mode == "strong" else (0.8, 1.2)
            image = ImageEnhance.Brightness(image).enhance(rng.uniform(*jitter))
            image = ImageEnhance.Contrast(image).enhance(rng.uniform(*jitter))
            image = ImageEnhance.Color(image).enhance(rng.uniform(*jitter))
            if self.mode == "strong":
                image, targets = random_scale_translate(image, targets, rng)
                if rng.random() < 0.10:
                    image = ImageEnhance.Color(image).enhance(0.0)
                if rng.random() < 0.10:
                    image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.1, 1.2)))
        image, targets = letterbox(image, targets, self.image_size)
        if self.mixup_probability and rng.random() < self.mixup_probability:
            image, targets = self._mixup(image, targets, length, load_raw, rng)
        if self.copy_paste_probability and rng.random() < self.copy_paste_probability:
            image, targets = self._copy_paste(image, targets, length, load_raw, rng)
        if self.random_erasing_probability and rng.random() < self.random_erasing_probability:
            image = self._erase(image, rng)
        return image, targets
