"""Export ``SyntheticShapesDataset`` samples in Ultralytics YOLO format."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from complexity.generative.detection import SyntheticShapesDataset
from complexity.generative.detection.data import SHAPE_CLASSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument("--train-seed", type=int, default=200_003)
    parser.add_argument("--validation-seed", type=int, default=1_000_000)
    return parser.parse_args()


def export_split(
    output: Path,
    split: str,
    *,
    samples: int,
    image_size: int,
    seed: int,
) -> None:
    images_dir = output / "images" / split
    labels_dir = output / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    dataset = SyntheticShapesDataset(samples, image_size=image_size, seed=seed)

    for index in range(samples):
        pixel_values, targets = dataset[index]
        pixels = (
            ((pixel_values * 0.5 + 0.5) * 255.0)
            .clamp(0, 255)
            .byte()
            .permute(1, 2, 0)
            .numpy()
        )
        stem = f"{index:06d}"
        Image.fromarray(pixels).save(images_dir / f"{stem}.png")
        lines = [
            f"{int(class_id)} {cx:.8f} {cy:.8f} {width:.8f} {height:.8f}"
            for cx, cy, width, height, class_id in targets.tolist()
        ]
        (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")

    print(f"exported {samples} {split} samples (seed={seed})")


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    export_split(
        output,
        "train",
        samples=args.train_samples,
        image_size=args.image_size,
        seed=args.train_seed,
    )
    export_split(
        output,
        "val",
        samples=args.validation_samples,
        image_size=args.image_size,
        seed=args.validation_seed,
    )
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(SHAPE_CLASSES))
    (output / "data.yaml").write_text(
        f"path: {output}\ntrain: images/train\nval: images/val\nnames:\n{names}\n"
    )
    print(f"dataset config: {output / 'data.yaml'}")


if __name__ == "__main__":
    main()
