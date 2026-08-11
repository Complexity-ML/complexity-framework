"""Supervised pretraining for a TR-Hash vision tower on real images."""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..detection.config import TRHashDetectorConfig
from ..detection.hierarchical_tower import HierarchicalTRHashVisionClassifier
from .vision_tower import TRHashVisionClassifier, TRHashVisionTowerConfig

LOGGER = logging.getLogger("tr_hash_vision_pretraining")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cifar10", action="store_true")
    source.add_argument("--image-folder", type=Path)
    source.add_argument(
        "--hf-dataset",
        help="Hugging Face image-classification dataset, for example clane9/imagenet-100",
    )
    parser.add_argument("--hf-train-split", default="train")
    parser.add_argument("--hf-validation-split", default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/vision"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--architecture-version", type=int, choices=(5, 6), default=5)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--expert-width", type=int, default=48)
    parser.add_argument("--stage-depths", type=int, nargs="+", default=(1, 1, 2))
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--expert-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--log-steps", type=int, default=25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class HuggingFaceImageDataset(Dataset):
    """Apply torchvision transforms lazily to an Arrow-backed image dataset."""

    def __init__(self, dataset, transform: Callable):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        example = self.dataset[index]
        image = example["image"].convert("RGB")
        return self.transform(image), int(example["label"])


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

    interpolation = transforms.InterpolationMode.BICUBIC
    train_transform = transforms.Compose(
        (
            transforms.RandomResizedCrop(
                args.image_size, scale=(0.5, 1.0), interpolation=interpolation
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        )
    )
    validation_transform = transforms.Compose(
        (
            transforms.Resize(
                round(args.image_size / 0.875), interpolation=interpolation
            ),
            transforms.CenterCrop(args.image_size),
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

    if args.hf_dataset:
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise RuntimeError(
                "Hugging Face pretraining requires `datasets` and `hf_xet`"
            ) from error
        train_source = load_dataset(
            args.hf_dataset,
            split=args.hf_train_split,
            cache_dir=str(args.data_root),
        )
        validation_source = load_dataset(
            args.hf_dataset,
            split=args.hf_validation_split,
            cache_dir=str(args.data_root),
        )
        label_feature = train_source.features["label"]
        class_names = getattr(label_feature, "names", None)
        num_classes = (
            len(class_names)
            if class_names
            else max(train_source.unique("label")) + 1
        )
        return (
            HuggingFaceImageDataset(train_source, train_transform),
            HuggingFaceImageDataset(validation_source, validation_transform),
            num_classes,
        )

    train = datasets.ImageFolder(args.image_folder / "train", transform=train_transform)
    validation = datasets.ImageFolder(
        args.image_folder / "val", transform=validation_transform
    )
    if train.class_to_idx != validation.class_to_idx:
        raise ValueError("train and validation ImageFolder classes differ")
    return train, validation, len(train.classes)


def save_tower(
    output: Path,
    model,
    config,
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
    if isinstance(config, TRHashDetectorConfig):
        config_values = config.to_dict()
    else:
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
    model: TRHashVisionClassifier,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool = False,
    show_progress: bool = False,
) -> float:
    model.eval()
    correct = total = 0
    progress = tqdm(
        loader,
        desc="vision validation",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
        disable=False if show_progress else True,
    )
    for pixels, labels in progress:
        pixels = pixels.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if use_amp
            else nullcontext()
        )
        with autocast:
            predictions = model(pixels)["logits"].argmax(-1)
        correct += int((predictions == labels).sum().cpu())
        total += len(labels)
    model.train()
    return correct / max(total, 1)


def cosine_schedule(
    step: int, *, warmup_steps: int, total_steps: int, min_ratio: float
) -> float:
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


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
    LOGGER.info(
        "Vision dataset: %d train, %d validation, %d classes",
        len(train_dataset),
        len(validation_dataset),
        num_classes,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=device.type == "cuda",
    )
    precision = "fp32" if device.type == "mps" else "bf16"
    if args.architecture_version == 6:
        config = TRHashDetectorConfig(
            architecture_version=6,
            image_size=args.image_size,
            patch_size=args.patch_size,
            vision_hidden_size=args.hidden_size,
            vision_layers=args.layers,
            vision_heads=args.heads,
            vision_num_experts=args.num_experts,
            vision_top_k=args.top_k,
            vision_expert_width=args.expert_width,
            vision_stage_depths=tuple(args.stage_depths),
            vision_window_size=args.window_size,
            vision_precision=precision,
            num_classes=num_classes,
        )
        model = HierarchicalTRHashVisionClassifier(config, num_classes).to(device)
    else:
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
    LOGGER.info(
        "Vision model: %.2fM parameters on %s",
        sum(parameter.numel() for parameter in model.parameters()) / 1e6,
        device,
    )
    if args.expert_lr_multiplier <= 0.0:
        raise ValueError("--expert-lr-multiplier must be positive")
    expert_parameters = []
    base_parameters = []
    for name, parameter in model.named_parameters():
        target = expert_parameters if ".mlp.expert_" in name else base_parameters
        target.append(parameter)
    optimizer_options = {
        "weight_decay": args.weight_decay,
        "foreach": False if device.type == "mps" else None,
    }
    if device.type == "cuda":
        optimizer_options["fused"] = True
    optimizer = torch.optim.AdamW(
        (
            {"params": base_parameters, "lr": args.lr},
            {
                "params": expert_parameters,
                "lr": args.lr * args.expert_lr_multiplier,
            },
        ),
        **optimizer_options,
    )
    total_steps = max(args.epochs * len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_schedule(
            step,
            warmup_steps=min(args.warmup_steps, total_steps),
            total_steps=total_steps,
            min_ratio=args.min_lr_ratio,
        ),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.jsonl"
    use_amp = device.type == "cuda"
    best_accuracy = -1.0
    started = time.monotonic()
    step = 0
    running_loss = 0.0
    for epoch in range(args.epochs):
        progress = tqdm(
            train_loader,
            desc=f"vision train {epoch + 1}/{args.epochs}",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            disable=False,
        )
        for pixels, labels in progress:
            pixels = pixels.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            autocast = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if use_amp
                else nullcontext()
            )
            with autocast:
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
            step += 1
            running_loss += float(loss.detach())
            if step % args.log_steps == 0:
                average_loss = running_loss / args.log_steps
                record = {
                    "step": step,
                    "epoch": epoch,
                    "loss": average_loss,
                    "lr": scheduler.get_last_lr()[0],
                    "expert_lr": scheduler.get_last_lr()[1],
                    "elapsed": time.monotonic() - started,
                }
                LOGGER.info(
                    "epoch=%d step=%d/%d loss=%.4f lr=%.2e expert_lr=%.2e elapsed=%.1fs",
                    epoch,
                    step,
                    total_steps,
                    average_loss,
                    record["lr"],
                    record["expert_lr"],
                    record["elapsed"],
                )
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                progress.set_postfix(
                    loss=f"{average_loss:.4f}",
                    lr=f"{record['lr']:.2e}",
                    expert_lr=f"{record['expert_lr']:.2e}",
                )
                running_loss = 0.0
        accuracy = validate(
            model,
            validation_loader,
            device,
            use_amp=use_amp,
            show_progress=True,
        )
        LOGGER.info(
            "validation epoch=%d val_accuracy=%.4f elapsed=%.1fs",
            epoch,
            accuracy,
            time.monotonic() - started,
        )
        with metrics_path.open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "step": step,
                        "epoch": epoch,
                        "validation_accuracy": accuracy,
                        "elapsed": time.monotonic() - started,
                    }
                )
                + "\n"
            )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_tower(args.output / "best", model, config, epoch=epoch, accuracy=accuracy)
    save_tower(args.output / "last", model, config, epoch=args.epochs - 1, accuracy=accuracy)


if __name__ == "__main__":
    main()
