"""Composable box-aware augmentations shared by COCO and YOLO datasets."""

from __future__ import annotations

import math
import random
from collections.abc import Callable

import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from .albumentations_backend import apply_albumentations, build_transform

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


def crop_mosaic_canvas(
    image: Image.Image,
    targets: torch.Tensor,
    *,
    output_size: int,
    left: int,
    top: int,
    min_visible_fraction: float = 0.20,
) -> tuple[Image.Image, torch.Tensor]:
    """Crop a larger Mosaic canvas and remap normalized ``cxcywh`` targets."""

    canvas_width, canvas_height = image.size
    if canvas_width != canvas_height:
        raise ValueError("Mosaic crop requires a square canvas")
    if output_size > canvas_width:
        raise ValueError("Mosaic crop cannot exceed the canvas size")
    if not 0 <= left <= canvas_width - output_size:
        raise ValueError("Mosaic crop left offset is outside the canvas")
    if not 0 <= top <= canvas_height - output_size:
        raise ValueError("Mosaic crop top offset is outside the canvas")
    cropped = image.crop((left, top, left + output_size, top + output_size))
    if not targets.numel():
        return cropped, targets

    classes = targets[:, 4:]
    canvas_dimensions = targets.new_tensor((canvas_width, canvas_height))
    centers = targets[:, :2] * canvas_dimensions
    sizes = targets[:, 2:4] * canvas_dimensions
    original_area = (sizes[:, 0] * sizes[:, 1]).clamp_min(1e-6)
    offset = targets.new_tensor((left, top))
    top_left = centers - sizes / 2 - offset
    bottom_right = centers + sizes / 2 - offset
    crop_dimensions = targets.new_tensor((output_size, output_size))
    top_left = torch.maximum(top_left, targets.new_zeros(2))
    bottom_right = torch.minimum(bottom_right, crop_dimensions)
    clipped_sizes = bottom_right - top_left
    visible_area = clipped_sizes[:, 0].clamp_min(0) * clipped_sizes[:, 1].clamp_min(
        0
    )
    keep = (
        (clipped_sizes[:, 0] >= 2.0)
        & (clipped_sizes[:, 1] >= 2.0)
        & (visible_area / original_area >= min_visible_fraction)
    )
    remapped = torch.cat(
        (
            ((top_left + bottom_right) / 2) / crop_dimensions,
            clipped_sizes / crop_dimensions,
            classes,
        ),
        dim=-1,
    )
    return cropped, remapped[keep]


class DetectionAugmenter:
    def __init__(
        self,
        image_size: int,
        *,
        mode: str | None,
        backend: str = "native",
        seed: int,
        mosaic_probability: float,
        mixup_probability: float,
        copy_paste_probability: float,
        random_erasing_probability: float,
        total_epochs: int,
        close_mosaic_epochs: int,
        mosaic_tiles: int = 4,
        mosaic_canvas_size: int = 0,
    ):
        if mode not in {None, "light", "strong"}:
            raise ValueError("augmentation must be light, strong, or None")
        if backend not in {"native", "albumentations"}:
            raise ValueError("augmentation backend must be native or albumentations")
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
        mosaic_grid = math.isqrt(mosaic_tiles)
        if mosaic_grid < 2 or mosaic_grid * mosaic_grid != mosaic_tiles:
            raise ValueError("mosaic_tiles must be a perfect square of at least 4")
        resolved_canvas_size = int(mosaic_canvas_size or image_size)
        if resolved_canvas_size < image_size:
            raise ValueError("mosaic_canvas_size must be at least image_size")
        if resolved_canvas_size % mosaic_grid:
            raise ValueError("mosaic_canvas_size must be divisible by the Mosaic grid")
        self.image_size = image_size
        self.mode = mode
        self.backend = backend
        self.transform = build_transform(mode) if mode and backend == "albumentations" else None
        self.seed = seed
        self.mosaic_probability = float(mosaic_probability)
        self.mixup_probability = float(mixup_probability)
        self.copy_paste_probability = float(copy_paste_probability)
        self.random_erasing_probability = float(random_erasing_probability)
        self.total_epochs = int(total_epochs)
        self.close_mosaic_epochs = int(close_mosaic_epochs)
        self.mosaic_tiles = int(mosaic_tiles)
        self.mosaic_grid = mosaic_grid
        self.mosaic_canvas_size = resolved_canvas_size
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
        grid = self.mosaic_grid
        canvas_size = self.mosaic_canvas_size
        tile_size = canvas_size // grid
        canvas = Image.new("RGB", (canvas_size, canvas_size), (114, 114, 114))
        indices = [index] + [rng.randrange(length) for _ in range(self.mosaic_tiles - 1)]
        all_targets = []
        for tile_index, source_index in enumerate(indices):
            image, targets = load_raw(source_index)
            image, targets = letterbox(image, targets, tile_size)
            col, row = tile_index % grid, tile_index // grid
            canvas.paste(image, (col * tile_size, row * tile_size))
            if targets.numel():
                targets = targets.clone()
                targets[:, 0] = (targets[:, 0] + col) / grid
                targets[:, 1] = (targets[:, 1] + row) / grid
                targets[:, 2:4] /= grid
                all_targets.append(targets)
        combined = torch.cat(all_targets) if all_targets else torch.empty(0, 5)
        if canvas_size == self.image_size:
            return canvas, combined
        crop_left = rng.randint(0, canvas_size - self.image_size)
        crop_top = rng.randint(0, canvas_size - self.image_size)
        return crop_mosaic_canvas(
            canvas,
            combined,
            output_size=self.image_size,
            left=crop_left,
            top=crop_top,
        )

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
        if self.mode and self.backend == "native":
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
        elif self.transform is not None:
            image, targets = apply_albumentations(
                self.transform,
                image,
                targets,
                seed=self.seed + self.epoch * length + index,
            )
        image, targets = letterbox(image, targets, self.image_size)
        if self.mixup_probability and rng.random() < self.mixup_probability:
            image, targets = self._mixup(image, targets, length, load_raw, rng)
        if self.copy_paste_probability and rng.random() < self.copy_paste_probability:
            image, targets = self._copy_paste(image, targets, length, load_raw, rng)
        if self.random_erasing_probability and rng.random() < self.random_erasing_probability:
            image = self._erase(image, rng)
        return image, targets
