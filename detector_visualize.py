"""Visualize TR-Hash detector predictions beside synthetic ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file

from complexity.generative.detection import (
    SyntheticShapesDataset,
    TRHashDetectorConfig,
    TRHashObjectDetector,
)

CLASS_NAMES = ("rectangle", "ellipse", "triangle")
COLORS = ((220, 40, 40), (40, 120, 220), (40, 170, 70))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Checkpoint directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/tr_hash_yolo_result.png"),
        help="Destination PNG (default: %(default)s)",
    )
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--iou-threshold", type=float, default=0.4)
    return parser.parse_args()


def draw_label(
    draw: ImageDraw.ImageDraw,
    position: tuple[float, float],
    text: str,
    color: tuple[int, int, int],
) -> None:
    """Draw readable text over a solid white label background."""

    x, y = position
    left, top, right, bottom = draw.textbbox((x, y), text)
    draw.rectangle((left - 1, top - 1, right + 1, bottom + 1), fill="white")
    draw.text((x, y), text, fill=color)


def main() -> None:
    args = parse_args()
    config_path = args.checkpoint / "config.json"
    weights_path = args.checkpoint / "model.safetensors"
    config = TRHashDetectorConfig.from_dict(json.loads(config_path.read_text()))

    model = TRHashObjectDetector(config).eval()
    model.load_state_dict(load_file(str(weights_path)))
    print(
        f"loaded checkpoint: {model.num_parameters() / 1e6:.2f}M params, "
        f"grid={config.grid_size}x{config.grid_size}"
    )

    dataset = SyntheticShapesDataset(
        length=args.samples,
        image_size=config.image_size,
        seed=args.seed,
    )
    panels: list[Image.Image] = []

    for index in range(args.samples):
        pixel_values, gt_targets = dataset[index]
        with torch.inference_mode():
            detections = model.predict(
                pixel_values.unsqueeze(0),
                objectness_threshold=args.threshold,
                iou_threshold=args.iou_threshold,
            )[0]

        raw = (
            ((pixel_values * 0.5 + 0.5) * 255.0)
            .clamp(0, 255)
            .byte()
            .permute(1, 2, 0)
            .numpy()
        )
        image = Image.fromarray(raw)
        size = config.image_size

        gt_panel = image.copy()
        gt_draw = ImageDraw.Draw(gt_panel)
        for cx, cy, width, height, class_id in gt_targets.tolist():
            box = (
                (cx - width / 2) * size,
                (cy - height / 2) * size,
                (cx + width / 2) * size,
                (cy + height / 2) * size,
            )
            gt_draw.rectangle(box, outline=COLORS[int(class_id)], width=2)
        draw_label(gt_draw, (4, 4), "ground truth", (0, 0, 0))

        pred_panel = image.copy()
        pred_draw = ImageDraw.Draw(pred_panel)
        for box, score, label in zip(
            detections["boxes"], detections["scores"], detections["labels"]
        ):
            box_pixels = tuple((box * size).tolist())
            class_id = int(label)
            color = COLORS[class_id]
            pred_draw.rectangle(box_pixels, outline=color, width=2)
            draw_label(
                pred_draw,
                (box_pixels[0] + 2, max(16, box_pixels[1] - 11)),
                f"{CLASS_NAMES[class_id]} {float(score):.2f}",
                color,
            )

        detection_count = detections["boxes"].shape[0]
        draw_label(pred_draw, (4, 4), f"predicted ({detection_count})", (0, 0, 0))

        combined = Image.new("RGB", (size * 2 + 8, size), "white")
        combined.paste(gt_panel, (0, 0))
        combined.paste(pred_panel, (size + 8, 0))
        panels.append(combined)
        print(
            f"image {index}: {len(gt_targets)} ground-truth boxes, "
            f"{detection_count} detections"
        )

    gap = 4
    grid = Image.new(
        "RGB",
        (panels[0].width, panels[0].height * len(panels) + gap * (len(panels) - 1)),
        "white",
    )
    y = 0
    for panel in panels:
        grid.paste(panel, (0, y))
        y += panel.height + gap

    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.output)
    print(f"saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
