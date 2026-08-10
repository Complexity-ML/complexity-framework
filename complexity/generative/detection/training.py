"""Training loop for ``TRHashObjectDetector``.

Single-device (CPU/MPS/CUDA), non-distributed -- the detector is small
enough that a training farm is not the point here; this is for proving the
model actually learns, and for fine-tuning on a real COCO-format dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from .config import TRHashDetectorConfig
from .data import CocoDetectionDataset, SyntheticShapesDataset, collate_detection
from .model import TRHashObjectDetector

LOGGER = logging.getLogger("tr_hash_detector")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=None, help="COCO-format JSON")
    parser.add_argument("--images", type=Path, default=None, help="Directory of images for --annotations")
    parser.add_argument("--synthetic-samples", type=int, default=512, help="Used when --annotations is omitted")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--vision-hidden-size", type=int, default=192)
    parser.add_argument("--vision-layers", type=int, default=4)
    parser.add_argument("--vision-heads", type=int, default=6)
    parser.add_argument("--vision-num-experts", type=int, default=4)
    parser.add_argument("--vision-top-k", type=int, default=2)
    parser.add_argument("--vision-expert-width", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--log-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=0, help="0 disables periodic checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Override auto-detected device")
    return parser.parse_args()


def resolve_device(override: str | None) -> torch.device:
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(output: Path, model: TRHashObjectDetector, config: TRHashDetectorConfig, step: int) -> None:
    target = output / f"step_{step:06d}"
    target.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    save_file(state, str(target / "model.safetensors"))
    (target / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    LOGGER.info("Checkpoint saved: %s", target)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    if args.annotations is not None:
        if args.images is None:
            raise ValueError("--images is required alongside --annotations")
        dataset = CocoDetectionDataset(args.annotations, args.images, image_size=args.image_size)
        num_classes = dataset.num_classes
        LOGGER.info("COCO dataset: %d images, %d classes", len(dataset), num_classes)
    else:
        dataset = SyntheticShapesDataset(args.synthetic_samples, image_size=args.image_size, seed=args.seed)
        num_classes = args.num_classes
        LOGGER.info("Synthetic dataset: %d images, %d classes", len(dataset), num_classes)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_detection,
    )

    config = TRHashDetectorConfig(
        image_size=args.image_size,
        patch_size=args.patch_size,
        vision_hidden_size=args.vision_hidden_size,
        vision_layers=args.vision_layers,
        vision_heads=args.vision_heads,
        vision_num_experts=args.vision_num_experts,
        vision_top_k=args.vision_top_k,
        vision_expert_width=args.vision_expert_width,
        num_classes=num_classes,
    )
    model = TRHashObjectDetector(config).to(device)
    LOGGER.info("Model: %.2fM parameters", model.num_parameters() / 1e6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.jsonl"

    step = 0
    running_loss = 0.0
    started = time.monotonic()
    for epoch in range(args.epochs):
        for pixel_values, targets in loader:
            pixel_values = pixel_values.to(device, non_blocking=True)
            targets = [target.to(device, non_blocking=True) for target in targets]

            raw = model(pixel_values)
            losses = model.compute_loss(raw, targets)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            running_loss += float(losses["loss"].detach())
            if step % args.log_steps == 0:
                average_loss = running_loss / args.log_steps
                elapsed = time.monotonic() - started
                LOGGER.info(
                    "epoch=%d step=%d loss=%.4f obj=%.4f box=%.4f cls=%.4f elapsed=%.1fs",
                    epoch,
                    step,
                    average_loss,
                    float(losses["objectness_loss"].detach()),
                    float(losses["box_loss"].detach()),
                    float(losses["class_loss"].detach()),
                    elapsed,
                )
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps({"step": step, "epoch": epoch, "loss": average_loss}) + "\n")
                running_loss = 0.0
            if args.save_steps and step % args.save_steps == 0:
                save_checkpoint(args.output, model, config, step)

    save_checkpoint(args.output, model, config, step)
    LOGGER.info("Training complete: %d steps over %d epochs", step, args.epochs)


if __name__ == "__main__":
    main()
