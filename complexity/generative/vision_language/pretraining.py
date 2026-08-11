"""Supervised pretraining for a TR-Hash vision tower on real images."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from .vision_tower import TRHashVisionClassifier, TRHashVisionTowerConfig

LOGGER = logging.getLogger("tr_hash_vision_pretraining")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cifar10", action="store_true")
    source.add_argument("--image-folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/vision"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--expert-width", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--expert-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_device(override: str | None) -> torch.device:
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_datasets(args: argparse.Namespace):
    try:
        from torchvision import datasets, transforms
    except ImportError as error:
        raise RuntimeError("vision pretraining requires torchvision") from error

    train_transform = transforms.Compose(
        (
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        )
    )
    validation_transform = transforms.Compose(
        (
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        )
    )
    if args.cifar10:
        train = datasets.CIFAR10(
            args.data_root, train=True, transform=train_transform, download=True
        )
        validation = datasets.CIFAR10(
            args.data_root, train=False, transform=validation_transform, download=True
        )
        return train, validation, 10

    train = datasets.ImageFolder(args.image_folder / "train", transform=train_transform)
    validation = datasets.ImageFolder(
        args.image_folder / "val", transform=validation_transform
    )
    if train.class_to_idx != validation.class_to_idx:
        raise ValueError("train and validation ImageFolder classes differ")
    return train, validation, len(train.classes)


def save_tower(
    output: Path,
    model: TRHashVisionClassifier,
    config: TRHashVisionTowerConfig,
    *,
    epoch: int,
    accuracy: float,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in model.tower.state_dict().items()
    }
    save_file(state, str(output / "tower.safetensors"))
    config_values = asdict(config)
    config_values["precision"] = config.precision.value
    (output / "config.json").write_text(
        json.dumps(
            {"tower": config_values, "epoch": epoch, "validation_accuracy": accuracy},
            indent=2,
        )
        + "\n"
    )


@torch.inference_mode()
def validate(
    model: TRHashVisionClassifier, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    correct = total = 0
    for pixels, labels in loader:
        pixels = pixels.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        predictions = model(pixels)["logits"].argmax(-1)
        correct += int((predictions == labels).sum().cpu())
        total += len(labels)
    model.train()
    return correct / max(total, 1)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    train_dataset, validation_dataset, num_classes = build_datasets(args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    precision = "fp32" if device.type == "mps" else "bf16"
    config = TRHashVisionTowerConfig(
        image_size=args.image_size,
        patch_size=args.patch_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        num_experts=args.num_experts,
        top_k=args.top_k,
        expert_width=args.expert_width,
        precision=precision,
    )
    model = TRHashVisionClassifier(config, num_classes).to(device)
    if args.expert_lr_multiplier <= 0.0:
        raise ValueError("--expert-lr-multiplier must be positive")
    expert_parameters = []
    base_parameters = []
    for name, parameter in model.named_parameters():
        target = expert_parameters if ".mlp.expert_" in name else base_parameters
        target.append(parameter)
    optimizer = torch.optim.AdamW(
        (
            {"params": base_parameters, "lr": args.lr},
            {
                "params": expert_parameters,
                "lr": args.lr * args.expert_lr_multiplier,
            },
        ),
        weight_decay=args.weight_decay,
        foreach=False if device.type == "mps" else None,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs * len(train_loader), 1)
    )
    best_accuracy = -1.0
    started = time.monotonic()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for pixels, labels in train_loader:
            pixels = pixels.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            loss = model(pixels, labels=labels)["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                foreach=False if device.type == "mps" else None,
            )
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.detach())
        accuracy = validate(model, validation_loader, device)
        LOGGER.info(
            "epoch=%d loss=%.4f val_accuracy=%.4f elapsed=%.1fs",
            epoch,
            running_loss / max(len(train_loader), 1),
            accuracy,
            time.monotonic() - started,
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_tower(args.output / "best", model, config, epoch=epoch, accuracy=accuracy)
    save_tower(args.output / "last", model, config, epoch=args.epochs - 1, accuracy=accuracy)


if __name__ == "__main__":
    main()
