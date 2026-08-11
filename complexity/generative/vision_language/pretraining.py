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
from safetensors.torch import load_file, save_file
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler
from tqdm import tqdm

from ..detection.checkpointing import load_training_state, save_training_state
from ..detection.config import TRHashDetectorConfig
from ..detection.distributed import DistributedContext
from ..detection.hierarchical_tower import HierarchicalTRHashVisionClassifier
from ..detection.training import TqdmLoggingHandler, vision_backend_summary
from .vision_tower import TRHashVisionClassifier, TRHashVisionTowerConfig

LOGGER = logging.getLogger("tr_hash_vision_pretraining")


class ResumableBatchSampler:
    """Skip consumed batches at the index level without decoding their images."""

    def __init__(self, batch_sampler: BatchSampler) -> None:
        self.batch_sampler = batch_sampler
        self.start_batch = 0

    def set_start_batch(self, start_batch: int) -> None:
        if not 0 <= start_batch <= len(self.batch_sampler):
            raise ValueError("start_batch is outside the epoch")
        self.start_batch = start_batch

    def __iter__(self):
        iterator = iter(self.batch_sampler)
        for _ in range(self.start_batch):
            next(iterator)
        yield from iterator

    def __len__(self) -> int:
        # Keep the full epoch length: the training cursor and tqdm initial value
        # account for the skipped prefix separately.
        return len(self.batch_sampler)


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
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="exactly resume model, optimizer, scheduler, cursor and RNG state",
    )
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
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--require-triton",
        action="store_true",
        help="fail instead of silently using the PyTorch TR-Hash fallback",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class HuggingFaceImageDataset(Dataset):
    """Apply torchvision transforms lazily to an Arrow-backed image dataset."""

    def __init__(self, dataset, transform: Callable, *, seed: int = 0):
        self.dataset = dataset
        self.transform = transform
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        example = self.dataset[index]
        image = example["image"].convert("RGB")
        # Make stochastic crops/jitter a pure function of epoch+index. This is
        # required for an exact mid-epoch resume with DataLoader workers.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed + self.epoch * len(self) + index)
            pixels = self.transform(image)
        return pixels, int(example["label"])


class TorchvisionImageDataset(Dataset):
    """Apply transforms deterministically to a raw torchvision-style dataset."""

    def __init__(self, dataset, transform: Callable, *, seed: int = 0):
        self.dataset = dataset
        self.transform = transform
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, label = self.dataset[index]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed + self.epoch * len(self) + index)
            pixels = self.transform(image.convert("RGB"))
        return pixels, int(label)


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
        train_source = datasets.CIFAR10(
            args.data_root, train=True, transform=None, download=True
        )
        validation_source = datasets.CIFAR10(
            args.data_root, train=False, transform=None, download=True
        )
        return (
            TorchvisionImageDataset(train_source, train_transform, seed=args.seed),
            TorchvisionImageDataset(validation_source, validation_transform),
            10,
        )

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
        # Derive the head from labels actually present in the training split.
        # Some repacks append a metadata-only ``none`` class for the unlabeled
        # test split, which must not create a spurious 1001st training class.
        num_classes = max(train_source.unique("label")) + 1
        return (
            HuggingFaceImageDataset(train_source, train_transform, seed=args.seed),
            HuggingFaceImageDataset(validation_source, validation_transform),
            num_classes,
        )

    train_source = datasets.ImageFolder(args.image_folder / "train")
    validation_source = datasets.ImageFolder(args.image_folder / "val")
    if train_source.class_to_idx != validation_source.class_to_idx:
        raise ValueError("train and validation ImageFolder classes differ")
    return (
        TorchvisionImageDataset(train_source, train_transform, seed=args.seed),
        TorchvisionImageDataset(validation_source, validation_transform),
        len(train_source.classes),
    )


def _config_values(config) -> dict:
    if isinstance(config, TRHashDetectorConfig):
        return config.to_dict()
    values = asdict(config)
    values["precision"] = config.precision.value
    return values


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
    (output / "config.json").write_text(
        json.dumps(
            {
                "tower": _config_values(config),
                "epoch": epoch,
                "validation_accuracy": accuracy,
            },
            indent=2,
        )
        + "\n"
    )


def save_vision_checkpoint(
    output: Path,
    model: torch.nn.Module,
    config,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    batch_in_epoch: int,
    step: int,
    best_accuracy: float,
    running_loss: float,
    running_loss_steps: int,
    total_epochs: int,
    steps_per_epoch: int,
    training_options: dict[str, object],
    distributed_rng_states,
    accuracy: float,
) -> None:
    """Write transfer weights plus the exact state required for continuation."""

    metadata_epoch = epoch - 1 if batch_in_epoch == 0 and epoch > 0 else epoch
    save_tower(output, model, config, epoch=metadata_epoch, accuracy=accuracy)
    model_state = {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
    }
    save_file(model_state, str(output / "model.safetensors"))
    save_training_state(
        output,
        optimizer,
        scheduler,
        epoch=epoch,
        batch_in_epoch=batch_in_epoch,
        step=step,
        best_map50=best_accuracy,
        running_losses={"classification": running_loss},
        running_loss_steps=running_loss_steps,
        total_epochs=total_epochs,
        steps_per_epoch=steps_per_epoch,
        training_options=training_options,
        distributed_rng_states=distributed_rng_states,
    )


@torch.inference_mode()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool = False,
    show_progress: bool = False,
    distributed: DistributedContext | None = None,
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
    if distributed is not None and distributed.enabled:
        counts = distributed.all_gather_objects((correct, total))
        correct = sum(item[0] for item in counts)
        total = sum(item[1] for item in counts)
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
    distributed = DistributedContext.initialize(resolve_device(args.device))
    logging.basicConfig(
        level=logging.INFO if distributed.is_main else logging.ERROR,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[TqdmLoggingHandler()],
        force=True,
    )
    torch.manual_seed(args.seed + distributed.rank)
    device = distributed.device
    # Hugging Face/datasets serializes cache writes with file locks.  Let every
    # rank resolve the local dataset before the first NCCL collective so a large
    # initial download cannot time out a process-group barrier.
    train_dataset, validation_dataset, num_classes = build_datasets(args)
    if distributed.is_main:
        LOGGER.info(
            "Vision dataset: %d train, %d validation, %d classes",
            len(train_dataset),
            len(validation_dataset),
            num_classes,
        )
    train_sampler = distributed.train_sampler(train_dataset, args.seed)
    shuffle_generator = torch.Generator()
    sample_sampler = train_sampler or RandomSampler(
        train_dataset,
        generator=shuffle_generator,
    )
    train_batch_sampler = ResumableBatchSampler(
        BatchSampler(sample_sampler, args.batch_size, drop_last=False)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=args.workers,
        # Recreate workers so train_dataset.set_epoch() reaches worker copies.
        persistent_workers=False,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        sampler=distributed.eval_sampler(validation_dataset),
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
    if args.resume is not None:
        saved_config = json.loads((args.resume / "config.json").read_text())["tower"]
        current_config = json.loads(json.dumps(_config_values(config)))
        if saved_config != current_config:
            raise ValueError("exact resume requires an unchanged vision model config")
        model.load_state_dict(load_file(str(args.resume / "model.safetensors")))
    if distributed.is_main:
        LOGGER.info(
            "Vision model: %.2fM parameters on %s (%d GPU%s)",
            sum(parameter.numel() for parameter in model.parameters()) / 1e6,
            device,
            distributed.world_size,
            "s" if distributed.world_size != 1 else "",
        )
        backend = vision_backend_summary(
            model,
            device.type,
            require_triton=args.require_triton,
        )
        LOGGER.info("TR-Hash vision backend: %s", backend["selected_backend"])
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
    if args.log_steps <= 0 or args.eval_every <= 0 or args.save_steps < 0:
        raise ValueError("log/eval steps must be positive and save steps non-negative")
    training_options = {
        "batch_size": args.batch_size,
        "dataset_size": len(train_dataset),
        "seed": args.seed,
        "lr": args.lr,
        "expert_lr_multiplier": args.expert_lr_multiplier,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "min_lr_ratio": args.min_lr_ratio,
        "model_config": _config_values(config),
        "dataset": args.hf_dataset or str(args.image_folder or "cifar10"),
    }
    start_epoch = 0
    start_batch = 0
    step = 0
    best_accuracy = -1.0
    last_accuracy = -1.0
    running_loss = 0.0
    running_loss_steps = 0
    if args.resume is not None:
        resume_state = load_training_state(
            args.resume,
            optimizer,
            scheduler,
            total_epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            training_options=training_options,
            rank=distributed.rank,
            world_size=distributed.world_size,
            device=device,
        )
        start_epoch = int(resume_state["epoch"])
        start_batch = int(resume_state["batch_in_epoch"])
        step = int(resume_state["step"])
        best_accuracy = float(resume_state["best_map50"])
        last_accuracy = best_accuracy
        running_loss = float(
            resume_state.get("running_losses", {}).get("classification", 0.0)
        )
        running_loss_steps = int(resume_state.get("running_loss_steps", 0))
        if not 0 <= start_epoch <= args.epochs:
            raise ValueError(f"invalid resumed epoch cursor: {start_epoch}")
        if not 0 <= start_batch <= len(train_loader):
            raise ValueError(f"invalid resumed batch cursor: {start_batch}")
        if distributed.is_main:
            LOGGER.info(
                "Resumed exactly from %s: epoch=%d batch=%d step=%d",
                args.resume,
                start_epoch,
                start_batch,
                step,
            )

    training_model = distributed.wrap(model)
    if distributed.is_main:
        args.output.mkdir(parents=True, exist_ok=True)
    distributed.barrier()
    metrics_path = args.output / "metrics.jsonl"
    use_amp = device.type == "cuda"
    started = time.monotonic()

    def write_checkpoint(
        target: Path,
        *,
        epoch: int,
        batch_in_epoch: int,
        accuracy: float,
    ) -> None:
        distributed_rng_states = distributed.gather_rng_states()
        if distributed.is_main:
            save_vision_checkpoint(
                target,
                model,
                config,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                batch_in_epoch=batch_in_epoch,
                step=step,
                best_accuracy=best_accuracy,
                running_loss=running_loss,
                running_loss_steps=running_loss_steps,
                total_epochs=args.epochs,
                steps_per_epoch=len(train_loader),
                training_options=training_options,
                distributed_rng_states=distributed_rng_states,
                accuracy=accuracy,
            )
            LOGGER.info("Vision checkpoint saved: %s", target)
        distributed.barrier()

    for epoch in range(start_epoch, args.epochs):
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        else:
            shuffle_generator.manual_seed(args.seed + epoch)
        batches_to_skip = start_batch if epoch == start_epoch else 0
        train_batch_sampler.set_start_batch(batches_to_skip)
        loader_iterator = iter(train_loader)
        progress = tqdm(
            loader_iterator,
            desc=f"vision train {epoch + 1}/{args.epochs}",
            unit="batch",
            total=len(train_loader),
            initial=batches_to_skip,
            dynamic_ncols=True,
            leave=False,
            disable=not distributed.is_main,
        )
        for batch_index, (pixels, labels) in enumerate(
            progress,
            start=batches_to_skip,
        ):
            pixels = pixels.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            autocast = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if use_amp
                else nullcontext()
            )
            with autocast:
                loss = training_model(pixels, labels=labels)["loss"]
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
            running_loss_steps += 1
            if step % args.log_steps == 0:
                average_loss = distributed.mean_scalars(
                    {"loss": running_loss / max(running_loss_steps, 1)}
                )["loss"]
                record = {
                    "step": step,
                    "epoch": epoch,
                    "loss": average_loss,
                    "lr": scheduler.get_last_lr()[0],
                    "expert_lr": scheduler.get_last_lr()[1],
                    "elapsed": time.monotonic() - started,
                }
                if distributed.is_main:
                    with metrics_path.open("a") as handle:
                        handle.write(json.dumps(record) + "\n")
                    progress.set_postfix(
                        loss=f"{average_loss:.4f}",
                        lr=f"{record['lr']:.2e}",
                        expert_lr=f"{record['expert_lr']:.2e}",
                    )
                running_loss = 0.0
                running_loss_steps = 0
            if args.save_steps and step % args.save_steps == 0:
                next_batch = batch_index + 1
                next_epoch = epoch
                if next_batch == len(train_loader):
                    next_epoch += 1
                    next_batch = 0
                write_checkpoint(
                    args.output / f"step_{step:07d}",
                    epoch=next_epoch,
                    batch_in_epoch=next_batch,
                    accuracy=last_accuracy,
                )
        start_batch = 0
        should_evaluate = (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs
        if should_evaluate:
            last_accuracy = validate(
                model,
                validation_loader,
                device,
                use_amp=use_amp,
                show_progress=distributed.is_main,
                distributed=distributed,
            )
            if distributed.is_main:
                LOGGER.info(
                    "validation epoch=%d val_accuracy=%.4f elapsed=%.1fs",
                    epoch,
                    last_accuracy,
                    time.monotonic() - started,
                )
                with metrics_path.open("a") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "step": step,
                                "epoch": epoch,
                                "validation_accuracy": last_accuracy,
                                "elapsed": time.monotonic() - started,
                            }
                        )
                        + "\n"
                    )
            if last_accuracy > best_accuracy:
                best_accuracy = last_accuracy
                write_checkpoint(
                    args.output / "best",
                    epoch=epoch + 1,
                    batch_in_epoch=0,
                    accuracy=last_accuracy,
                )
    write_checkpoint(
        args.output / "last",
        epoch=args.epochs,
        batch_in_epoch=0,
        accuracy=last_accuracy,
    )
    distributed.close()


if __name__ == "__main__":
    main()
