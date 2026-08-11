"""Download Pascal VOC 2007+2012 and prepare a combined YOLO-format dataset."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

VOC_NAMES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)
ASSET_ROOT = "https://github.com/ultralytics/assets/releases/download/v0.0.0"
ARCHIVES = (
    f"{ASSET_ROOT}/VOCtrainval_06-Nov-2007.zip",
    f"{ASSET_ROOT}/VOCtest_06-Nov-2007.zip",
    f"{ASSET_ROOT}/VOCtrainval_11-May-2012.zip",
)
SPLITS = (
    ("2012", "train", "train"),
    ("2012", "val", "train"),
    ("2007", "train", "train"),
    ("2007", "val", "train"),
    ("2007", "test", "val"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/VOC"))
    parser.add_argument("--download-threads", type=int, default=3)
    return parser.parse_args()


def convert_box(width: int, height: int, coordinates: list[float]) -> tuple[float, ...]:
    xmin, xmax, ymin, ymax = coordinates
    center_x = (xmin + xmax) / 2.0 - 1.0
    center_y = (ymin + ymax) / 2.0 - 1.0
    return (
        center_x / width,
        center_y / height,
        (xmax - xmin) / width,
        (ymax - ymin) / height,
    )


def convert_annotation(annotation: Path) -> list[str]:
    root = ET.parse(annotation).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"missing image size in {annotation}")
    width = int(size.findtext("width", "0"))
    height = int(size.findtext("height", "0"))
    lines = []
    for object_node in root.iter("object"):
        name = object_node.findtext("name")
        if name not in VOC_NAMES or int(object_node.findtext("difficult", "0")) == 1:
            continue
        box = object_node.find("bndbox")
        if box is None:
            continue
        coordinates = [
            float(box.findtext(axis, "0"))
            for axis in ("xmin", "xmax", "ymin", "ymax")
        ]
        normalized = convert_box(width, height, coordinates)
        lines.append(" ".join(str(value) for value in (VOC_NAMES.index(name), *normalized)))
    return lines


def prepare_split(source: Path, output: Path, year: str, split: str, target: str) -> int:
    image_ids = (
        source / f"VOC{year}" / "ImageSets" / "Main" / f"{split}.txt"
    ).read_text().split()
    images_dir = output / "images" / target
    labels_dir = output / "labels" / target
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for image_id in image_ids:
        source_image = source / f"VOC{year}" / "JPEGImages" / f"{image_id}.jpg"
        destination_image = images_dir / source_image.name
        if source_image.exists() and not destination_image.exists():
            source_image.replace(destination_image)
        annotation = source / f"VOC{year}" / "Annotations" / f"{image_id}.xml"
        (labels_dir / f"{image_id}.txt").write_text(
            "\n".join(convert_annotation(annotation)) + "\n"
        )
    return len(image_ids)


def main() -> None:
    args = parse_args()
    try:
        from ultralytics.utils.downloads import download
    except ImportError as error:
        raise RuntimeError("VOC preparation requires ultralytics") from error

    output = args.output.resolve()
    download(
        list(ARCHIVES),
        dir=output / "downloads",
        threads=args.download_threads,
        exist_ok=True,
    )
    source = output / "downloads" / "VOCdevkit"
    counts = {"train": 0, "val": 0}
    for year, split, target in SPLITS:
        counts[target] += prepare_split(source, output, year, split, target)

    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(VOC_NAMES))
    (output / "data.yaml").write_text(
        f"path: {output}\ntrain: images/train\nval: images/val\nnames:\n{names}\n"
    )
    print(f"prepared Pascal VOC: {counts['train']} train, {counts['val']} validation")
    print(f"dataset config: {output / 'data.yaml'}")


if __name__ == "__main__":
    main()
