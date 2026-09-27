#!/usr/bin/env python3
"""Benchmark Pillow and OpenCV decoding on a COCO detection dataset."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from complexity.generative.detection.data import CocoDetectionDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def _measure(
    dataset: CocoDetectionDataset,
    indices: list[int],
    repeats: int,
) -> dict[str, float]:
    elapsed = []
    for _ in range(repeats):
        started = time.perf_counter()
        for index in indices:
            dataset._load_raw(index)
        elapsed.append(time.perf_counter() - started)
    median = statistics.median(elapsed)
    return {
        "median_seconds": median,
        "images_per_second": len(indices) / median,
        "minimum_seconds": min(elapsed),
    }


def main() -> None:
    args = parse_args()
    if args.samples <= 0 or args.repeats <= 0:
        raise ValueError("samples and repeats must be positive")
    datasets = {
        backend: CocoDetectionDataset(
            args.annotations,
            args.images,
            image_backend=backend,
        )
        for backend in ("pillow", "opencv")
    }
    indices = list(range(min(args.samples, len(datasets["pillow"]))))
    if not indices:
        raise ValueError("dataset contains no images")

    pillow_image, pillow_targets = datasets["pillow"]._load_raw(indices[0])
    opencv_image, opencv_targets = datasets["opencv"]._load_raw(indices[0])
    if pillow_image.size != opencv_image.size or not pillow_targets.equal(opencv_targets):
        raise RuntimeError("decoder backends disagree on image geometry or targets")
    pixel_error = np.abs(
        np.asarray(pillow_image, dtype=np.int16) - np.asarray(opencv_image, dtype=np.int16)
    )

    results = {
        backend: _measure(dataset, indices, args.repeats)
        for backend, dataset in datasets.items()
    }
    results["opencv_speedup"] = (
        results["opencv"]["images_per_second"]
        / results["pillow"]["images_per_second"]
    )
    results["parity"] = {
        "first_image_max_pixel_error": int(pixel_error.max()),
        "first_image_mean_pixel_error": float(pixel_error.mean()),
        "targets_equal": True,
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
