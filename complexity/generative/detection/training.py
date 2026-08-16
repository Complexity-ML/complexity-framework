"""Single-device or DDP training loop for ``TRHashObjectDetector``."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from contextlib import nullcontext
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, IterableDataset, Subset
from tqdm import tqdm

from ...training.finetuning import (
    VISION_SUPERVISED_FINETUNING,
    validate_full_parameter_finetuning,
)
from ...training.musgd import MuSGD, build_musgd_parameter_groups, named_learning_rates
from ...training.packing import resolve_packed_epoch_schedule
from ...training.tensorboard import TensorBoardMetricWriter
from .checkpointing import load_training_state, save_training_state
from .coco_evaluation import detections_to_coco, evaluate_coco_predictions
from .config import TRHashDetectorConfig
from .data import (
    CocoDetectionDataset,
    CocoVideoDetectionDataset,
    HuggingFaceDetectionDataset,
    SyntheticShapesDataset,
    YoloDetectionDataset,
    collate_detection,
)
from .distributed import DistributedContext
from .ema import ModelEMA
from .metrics import DetectionMetricsAccumulator
from .model import TRHashObjectDetector
from .provenance import (
    NATIVE_DETECTOR_IMPLEMENTATION,
    PROVENANCE_FORMAT_VERSION,
    read_detector_provenance,
    validate_native_random_init_provenance,
)

LOGGER = logging.getLogger("tr_hash_detector")
TRITON_BACKENDS = frozenset({"fused_cuda", "cggr"})


class TqdmLoggingHandler(logging.StreamHandler):
    """Write log records without corrupting an active tqdm progress bar."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=self.stream)
        except Exception:
            self.handleError(record)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=None, help="COCO-format JSON")
    parser.add_argument(
        "--images", type=Path, default=None, help="Directory of images for --annotations"
    )
    parser.add_argument("--validation-annotations", type=Path, default=None)
    parser.add_argument("--validation-images", type=Path, default=None)
    parser.add_argument(
        "--hf-detection-dataset",
        default=None,
        help="Hugging Face dataset ID streamed as image detection parquet",
    )
    parser.add_argument("--hf-detection-train-split", default="train")
    parser.add_argument("--hf-detection-validation-split", default="val")
    parser.add_argument("--hf-detection-train-examples", type=int, default=0)
    parser.add_argument("--hf-detection-validation-examples", type=int, default=0)
    parser.add_argument("--hf-detection-num-classes", type=int, default=0)
    parser.add_argument("--hf-detection-image-column", default="image")
    parser.add_argument("--hf-detection-annotations-column", default="annotations")
    parser.add_argument(
        "--hf-detection-metadata-file-glob",
        default=None,
        help=(
            "Repository-relative parquet glob for annotation-only projection; "
            "avoids reading embedded image bytes while building object weights"
        ),
    )
    parser.add_argument("--hf-detection-category-offset", type=int, default=1)
    parser.add_argument("--hf-detection-shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--video-annotations", type=Path, default=None)
    parser.add_argument("--video-images", type=Path, default=None)
    parser.add_argument("--validation-video-annotations", type=Path, default=None)
    parser.add_argument("--validation-video-images", type=Path, default=None)
    parser.add_argument("--video-clip-frames", type=int, default=5)
    parser.add_argument("--video-frame-stride", type=int, default=1)
    parser.add_argument("--yolo-images", type=Path, default=None)
    parser.add_argument("--yolo-labels", type=Path, default=None)
    parser.add_argument("--validation-yolo-images", type=Path, default=None)
    parser.add_argument("--validation-yolo-labels", type=Path, default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--synthetic-samples", type=int, default=512, help="Used when --annotations is omitted"
    )
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--validation-seed", type=int, default=1_000_000)
    parser.add_argument(
        "--fixed-synthetic-data",
        action="store_true",
        help="Reuse identical synthetic samples each epoch instead of resampling",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--training-purpose",
        choices=("detector-pretraining", VISION_SUPERVISED_FINETUNING),
        default="detector-pretraining",
        help="explicitly separate random-init pretraining from full-model vision SFT",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Exactly resume model, optimizer, scheduler and data cursor from a checkpoint",
    )
    parser.add_argument(
        "--backbone-checkpoint",
        type=Path,
        default=None,
        help="Directory containing a pretrained tower.safetensors",
    )
    parser.add_argument(
        "--detector-checkpoint",
        type=Path,
        default=None,
        help="Transfer a v6 detector checkpoint, optionally onto new classes",
    )
    parser.add_argument(
        "--class-map",
        type=Path,
        default=None,
        help="JSON object mapping target class IDs to source class IDs",
    )
    parser.add_argument(
        "--require-random-init",
        action="store_true",
        help=(
            "reject external detector/backbone weights and only resume checkpoints "
            "whose provenance records native random initialization"
        ),
    )
    parser.add_argument(
        "--provenance-dataset",
        default="unspecified",
        help="stable dataset identifier persisted in every checkpoint provenance file",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--architecture-version",
        type=int,
        choices=(6, 7, 8),
        default=6,
        help="detector architecture version",
    )
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--vision-hidden-size", type=int, default=192)
    parser.add_argument("--vision-layers", type=int, default=4)
    parser.add_argument("--vision-heads", type=int, default=6)
    parser.add_argument("--vision-num-experts", type=int, default=4)
    parser.add_argument("--vision-top-k", type=int, default=2)
    parser.add_argument("--vision-shared-width", type=int, default=96)
    parser.add_argument("--vision-expert-width", type=int, default=48)
    parser.add_argument(
        "--vision-stage-depths",
        type=int,
        nargs="+",
        default=(1, 1, 2),
        help="TR-Hash block counts for the hierarchical stages",
    )
    parser.add_argument("--vision-window-size", type=int, default=8)
    parser.add_argument("--single-scale", action="store_true")
    parser.add_argument(
        "--neck-mode",
        choices=("baseline", "fpn", "pan"),
        default="pan",
        help="cross-scale feature fusion used before the prediction heads",
    )
    parser.add_argument(
        "--neck-normalized-fusion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use positive per-channel normalized weights for adjacent-scale fusion",
    )
    parser.add_argument(
        "--neck-repeats",
        type=int,
        default=1,
        help="number of consecutive cross-scale fusion passes",
    )
    parser.add_argument(
        "--p2-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable the stride-8-equivalent fine prediction level",
    )
    parser.add_argument("--static-assignment", action="store_true")
    parser.add_argument("--assignment-top-k", type=int, default=5)
    parser.add_argument(
        "--vision-precision",
        choices=("auto", "fp32", "bf16", "fp16"),
        default="auto",
        help="auto selects fp32 on MPS and bf16 elsewhere",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="training batch size per device",
    )
    parser.add_argument(
        "--nominal-batch-size",
        type=int,
        default=0,
        help=(
            "target global batch used for automatic accumulation and weight-decay "
            "normalization; 0 disables normalization"
        ),
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=0,
        help="microbatches per optimizer step; 0 derives it from --nominal-batch-size",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=0,
        help="validation batch size per device; 0 reuses --batch-size",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--optimizer", choices=("musgd",), required=True)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--musgd-muon-weight", type=float, default=0.2)
    parser.add_argument("--musgd-sgd-weight", type=float, default=1.0)
    parser.add_argument(
        "--backbone-lr-multiplier",
        type=float,
        default=1.0,
        help="multiply the LR of vision tower parameters during detector fine-tuning",
    )
    parser.add_argument("--expert-lr-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--one-to-one-lr-multiplier",
        type=float,
        default=1.5,
        help="multiply the LR of the lightweight NMS-free output branch",
    )
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="EMA decay used for validation/export; 0 disables EMA",
    )
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument(
        "--warmup-epochs",
        type=float,
        default=None,
        help="derive warmup optimizer steps from this many epochs (overrides warmup-steps)",
    )
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--eval-confidence", type=float, default=0.20)
    parser.add_argument(
        "--eval-backend",
        choices=("auto", "internal", "pycocotools", "faster"),
        default="auto",
        help="auto uses official accelerated COCOeval for COCO validation",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="validate every N epochs and always after the final epoch",
    )
    parser.add_argument(
        "--eval-max-detections",
        type=int,
        default=100,
        help="maximum detections retained per validation image",
    )
    parser.add_argument("--box-loss-weight", type=float, default=5.0)
    parser.add_argument("--quality-loss-weight", type=float, default=1.0)
    parser.add_argument("--box-l1-weight", type=float, default=0.25)
    parser.add_argument("--box-iou-weight", type=float, default=1.0)
    parser.add_argument(
        "--level-adapters",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable identity-initialized residual adapters on every FPN level",
    )
    parser.add_argument("--level-adapter-ratio", type=float, default=0.25)
    parser.add_argument(
        "--class-level-hash-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="route class x FPN-level identities through a TR-Hash gate",
    )
    parser.add_argument("--class-level-gate-temperature", type=float, default=1.0)
    parser.add_argument(
        "--object-weighting",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="weight positives by class x object-size x scene-density frequency",
    )
    parser.add_argument("--object-weighting-beta", type=float, default=0.999)
    parser.add_argument("--object-weighting-max", type=float, default=4.0)
    parser.add_argument("--level-aux-loss-weight", type=float, default=0.0)
    parser.add_argument("--gate-calibration-loss-weight", type=float, default=0.0)
    parser.add_argument("--object-contrastive-loss-weight", type=float, default=0.0)
    parser.add_argument("--object-contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--object-contrastive-dim", type=int, default=64)
    parser.add_argument(
        "--video-motion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="accept [B,T,3,H,W] clips and inject temporal motion into P2-P5",
    )
    parser.add_argument("--video-motion-hidden-size", type=int, default=64)
    parser.add_argument("--video-motion-scale-init", type=float, default=0.1)
    parser.add_argument("--reg-max", type=int, default=16)
    parser.add_argument(
        "--head-hidden-size",
        type=int,
        default=0,
        help="0 uses half the backbone width, with a minimum of 32",
    )
    parser.add_argument(
        "--head-spatial-mixing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="apply independent residual depthwise spatial mixing before box/class heads",
    )
    parser.add_argument(
        "--regression-logit-scale",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="learn a positive DFL-logit scale independently for every pyramid level",
    )
    parser.add_argument("--dfl-loss-weight", type=float, default=0.5)
    parser.add_argument("--quality-focal-beta", type=float, default=2.0)
    parser.add_argument(
        "--end-to-end",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="train a one-to-one branch for NMS-free inference",
    )
    parser.add_argument("--one-to-one-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--one-to-one-loss-start",
        type=float,
        default=0.25,
        help="initial one-to-one loss weight, ramped to --one-to-one-loss-weight",
    )
    parser.add_argument(
        "--one-to-one-shared-gradient-scale",
        type=float,
        default=0.25,
        help="fraction of one-to-one gradient propagated into shared detector features",
    )
    parser.add_argument(
        "--augmentation",
        choices=("light", "strong"),
        default="strong",
    )
    parser.add_argument(
        "--augmentation-backend",
        choices=("native", "albumentations"),
        default="native",
        help="box-aware per-image augmentation backend",
    )
    parser.add_argument(
        "--image-backend",
        choices=("pillow", "opencv"),
        default="opencv",
        help="image decoder for random-access COCO/YOLO datasets",
    )
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument(
        "--mosaic-tiles",
        type=int,
        default=4,
        help=(
            "source images composited per Mosaic canvas; must be a perfect "
            "square >= 4 (4 = 2x2, 9 = 3x3, 16 = 4x4, ...). Purely a visual "
            "augmentation knob -- does not affect the step schedule"
        ),
    )
    parser.add_argument(
        "--mosaic-canvas-size",
        type=int,
        default=0,
        help=(
            "square Mosaic construction size before a random image-size crop; "
            "0 uses image-size directly"
        ),
    )
    parser.add_argument(
        "--mosaic-packed-epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "let --packed-epochs shorten Mosaic-active epochs; final "
            "close-Mosaic epochs always retain the full loader cardinality"
        ),
    )
    parser.add_argument(
        "--packed-epochs",
        type=int,
        default=1,
        help=(
            "divide the natural, un-packed step count by this factor while "
            "Mosaic is active (independent of --mosaic-tiles); 1 = no "
            "reduction, the natural full-pass step count"
        ),
    )
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--copy-paste", type=float, default=0.0)
    parser.add_argument("--random-erasing", type=float, default=0.0)
    parser.add_argument(
        "--close-mosaic-epochs",
        type=int,
        default=0,
        help="disable Mosaic for the final N epochs",
    )
    parser.add_argument(
        "--multi-scale-min",
        type=int,
        default=0,
        help="minimum square training resolution; 0 disables input resizing",
    )
    parser.add_argument(
        "--multi-scale-max",
        type=int,
        default=0,
        help="maximum square training resolution; 0 uses --image-size",
    )
    parser.add_argument(
        "--multi-scale-step",
        type=int,
        default=0,
        help="resolution interval; 0 uses --patch-size",
    )
    parser.add_argument("--no-stal", action="store_true")
    parser.add_argument("--no-progressive-loss", action="store_true")
    parser.add_argument("--log-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=0, help="0 disables periodic checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Override auto-detected device")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA BF16 autocast")
    parser.add_argument(
        "--require-triton",
        action="store_true",
        help="Fail at startup instead of silently using the PyTorch TR-Hash fallback",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("none", "default", "reduce-overhead", "max-autotune"),
        default="none",
        help="compile the detector forward/backward graph with TorchInductor/Triton",
    )
    return parser.parse_args()


def resolve_initialization_provenance(args: argparse.Namespace) -> Dict[str, object]:
    """Validate initialization policy and return immutable checkpoint provenance."""

    training_purpose = getattr(args, "training_purpose", "detector-pretraining")
    if training_purpose == VISION_SUPERVISED_FINETUNING:
        validate_full_parameter_finetuning(training_purpose)
        if args.detector_checkpoint is None:
            raise ValueError(
                "vision-supervised-finetuning requires --detector-checkpoint"
            )
        if args.resume is not None or args.backbone_checkpoint is not None:
            raise ValueError(
                "vision-supervised-finetuning resets optimizer/scheduler and therefore "
                "forbids --resume and --backbone-checkpoint"
            )
        if args.require_random_init:
            raise ValueError(
                "vision-supervised-finetuning cannot be combined with --require-random-init"
            )

    initialization_modes = (
        args.resume,
        args.backbone_checkpoint,
        args.detector_checkpoint,
    )
    if sum(value is not None for value in initialization_modes) > 1:
        raise ValueError(
            "--resume, --backbone-checkpoint, and --detector-checkpoint are mutually exclusive"
        )
    if args.class_map is not None and args.detector_checkpoint is None:
        raise ValueError("--class-map requires --detector-checkpoint")
    if args.require_random_init and (
        args.backbone_checkpoint is not None or args.detector_checkpoint is not None
    ):
        raise ValueError(
            "--require-random-init forbids --backbone-checkpoint and --detector-checkpoint"
        )
    if args.require_random_init and args.provenance_dataset == "unspecified":
        raise ValueError("--require-random-init requires an explicit --provenance-dataset")

    if args.resume is not None:
        provenance_path = args.resume / "provenance.json"
        if not provenance_path.is_file():
            if args.require_random_init:
                raise ValueError("strict random-init resume requires checkpoint provenance.json")
            return {
                "format_version": PROVENANCE_FORMAT_VERSION,
                "implementation": NATIVE_DETECTOR_IMPLEMENTATION,
                "initialization": "legacy-resume",
                "external_checkpoint": None,
                "dataset": args.provenance_dataset,
            }
        provenance = read_detector_provenance(args.resume)
        if args.require_random_init:
            validate_native_random_init_provenance(provenance, dataset=args.provenance_dataset)
        saved_dataset = provenance.get("dataset")
        if args.provenance_dataset != "unspecified" and saved_dataset != args.provenance_dataset:
            raise ValueError("checkpoint provenance dataset does not match --provenance-dataset")
        return dict(provenance)

    initialization = "random"
    external_checkpoint = None
    if args.detector_checkpoint is not None:
        initialization = "detector-transfer"
        external_checkpoint = str(args.detector_checkpoint)
    elif args.backbone_checkpoint is not None:
        initialization = "backbone-transfer"
        external_checkpoint = str(args.backbone_checkpoint)
    return {
        "format_version": PROVENANCE_FORMAT_VERSION,
        "implementation": NATIVE_DETECTOR_IMPLEMENTATION,
        "initialization": initialization,
        "external_checkpoint": external_checkpoint,
        "dataset": args.provenance_dataset,
    }


def vision_backend_summary(
    model: TRHashObjectDetector,
    device_type: str,
    *,
    require_triton: bool = False,
) -> dict:
    """Resolve and optionally enforce the execution backend for every vision block."""

    summaries = [block.mlp.capability_summary(device_type) for block in model.tower.blocks]
    selected = {summary["selected_backend"] for summary in summaries}
    if len(selected) != 1:
        raise RuntimeError(
            "vision blocks selected inconsistent TR-Hash backends: " + ", ".join(sorted(selected))
        )
    summary = summaries[0]
    if require_triton and summary["selected_backend"] not in TRITON_BACKENDS:
        reasons = "; ".join(summary["backend_reasons"]) or "no backend reason reported"
        raise RuntimeError(
            "Triton is required for this run, but the vision tower selected "
            f"{summary['selected_backend']}: {reasons}"
        )
    return summary


def resolve_device(override: str | None) -> torch.device:
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resize_detector_inputs(
    pixel_values: torch.Tensor,
    size: tuple[int, int],
) -> torch.Tensor:
    """Resize image batches or video batches without mixing time and channels."""

    if pixel_values.ndim == 4:
        return F.interpolate(
            pixel_values,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
    if pixel_values.ndim == 5:
        batch, frames, channels, height, width = pixel_values.shape
        resized = F.interpolate(
            pixel_values.reshape(batch * frames, channels, height, width),
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        return resized.reshape(batch, frames, channels, *size)
    raise ValueError("detector inputs must be image [B,C,H,W] or video [B,T,C,H,W]")


def _statistics_targets(dataset, index: int) -> torch.Tensor:
    if isinstance(dataset, Subset):
        return _statistics_targets(dataset.dataset, int(dataset.indices[index]))
    if isinstance(dataset, (CocoDetectionDataset, CocoVideoDetectionDataset)):
        image_id = dataset.image_ids[index]
        meta = dataset.images[image_id]
        width = meta.get("width")
        height = meta.get("height")
        if width is None or height is None:
            with Image.open(dataset.images_dir / meta["file_name"]) as image:
                width, height = image.size
        boxes = []
        for annotation in dataset.annotations_by_image[image_id]:
            _, _, box_width, box_height = annotation["bbox"]
            boxes.append(
                (
                    0.0,
                    0.0,
                    box_width / width,
                    box_height / height,
                    float(dataset.category_to_class[annotation["category_id"]]),
                )
            )
        return torch.tensor(boxes) if boxes else torch.empty(0, 5)
    if isinstance(dataset, YoloDetectionDataset):
        return dataset._load_targets(dataset.label_paths[index])
    return dataset[index][1]


def build_object_weight_table(
    dataset,
    num_classes: int,
    *,
    beta: float,
    max_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate effective-number weights over class x size x density buckets."""

    counts = torch.zeros(num_classes, 3, 3, dtype=torch.float64)
    for index in range(len(dataset)):
        targets = _statistics_targets(dataset, index)
        if not len(targets):
            continue
        density_bin = int(torch.bucketize(torch.tensor(len(targets)), torch.tensor((3, 10))))
        size_bins = torch.bucketize(targets[:, 2:4].prod(dim=-1), torch.tensor((0.02, 0.15)))
        classes = targets[:, 4].long()
        density_bins = torch.full_like(classes, density_bin)
        values = torch.ones_like(classes, dtype=counts.dtype)
        counts.index_put_((classes, size_bins, density_bins), values, accumulate=True)
    return object_weight_table_from_counts(counts, beta=beta, max_weight=max_weight)


def object_weight_table_from_counts(
    counts: torch.Tensor,
    *,
    beta: float,
    max_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert class x size x density counts into effective-number weights."""

    counts = counts.to(dtype=torch.float64, device="cpu")
    observed = counts > 0
    weights = torch.ones_like(counts)
    if observed.any():
        if beta:
            weights[observed] = (1.0 - beta) / (1.0 - beta ** counts[observed])
        else:
            weights[observed] = 1.0
        weights[observed] /= weights[observed].mean()
        weights.clamp_(max=max_weight)
    return weights.float(), counts


def load_object_bucket_count_cache(
    path: Path,
    *,
    dataset_id: str,
    split: str,
    num_examples: int,
    num_classes: int,
    category_id_offset: int,
    metadata_file_glob: str | None,
) -> torch.Tensor | None:
    """Load cached streamed annotation statistics when their identity matches."""

    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    expected = {
        "format_version": 1,
        "dataset_id": dataset_id,
        "split": split,
        "num_examples": num_examples,
        "num_classes": num_classes,
        "category_id_offset": category_id_offset,
        "metadata_file_glob": metadata_file_glob,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        return None
    counts = torch.tensor(payload.get("counts", ()), dtype=torch.float64)
    expected_shape = (num_classes, 3, 3)
    if tuple(counts.shape) != expected_shape:
        raise ValueError(
            f"invalid object bucket cache shape {tuple(counts.shape)}; "
            f"expected {expected_shape}"
        )
    return counts


def save_object_bucket_count_cache(
    path: Path,
    counts: torch.Tensor,
    *,
    dataset_id: str,
    split: str,
    num_examples: int,
    num_classes: int,
    category_id_offset: int,
    metadata_file_glob: str | None,
) -> None:
    """Atomically persist streamed annotation statistics for exact resume."""

    payload = {
        "format_version": 1,
        "dataset_id": dataset_id,
        "split": split,
        "num_examples": num_examples,
        "num_classes": num_classes,
        "category_id_offset": category_id_offset,
        "metadata_file_glob": metadata_file_glob,
        "counts": counts.to(dtype=torch.float64, device="cpu").tolist(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    temporary.replace(path)


def save_checkpoint(
    output: Path,
    model: TRHashObjectDetector,
    config: TRHashDetectorConfig,
    step: int,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    batch_in_epoch: int,
    best_map50: float,
    best_end_to_end_map50: float,
    running_losses: Dict[str, float],
    running_loss_steps: int,
    total_epochs: int,
    steps_per_epoch: int,
    training_options: Dict[str, object],
    provenance: Mapping[str, object],
    name: str | None = None,
    validation_metrics: Dict[str, Any] | None = None,
    distributed_rng_states: Sequence[Mapping[str, torch.Tensor]] | None = None,
    ema_model: TRHashObjectDetector | None = None,
) -> None:
    target = output / (name or f"step_{step:06d}")
    target.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    save_file(state, str(target / "model.safetensors"))
    tower_state = {
        name: value.detach().cpu().contiguous() for name, value in model.tower.state_dict().items()
    }
    save_file(tower_state, str(target / "tower.safetensors"))
    if ema_model is not None:
        ema_state = {
            name: value.detach().cpu().contiguous()
            for name, value in ema_model.state_dict().items()
        }
        save_file(ema_state, str(target / "ema.safetensors"))
        ema_tower_state = {
            name: value.detach().cpu().contiguous()
            for name, value in ema_model.tower.state_dict().items()
        }
        save_file(ema_tower_state, str(target / "ema_tower.safetensors"))
    (target / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    (target / "provenance.json").write_text(json.dumps(dict(provenance), indent=2) + "\n")
    if validation_metrics is not None:
        (target / "validation.json").write_text(json.dumps(validation_metrics, indent=2) + "\n")
    save_training_state(
        target,
        optimizer,
        scheduler,
        epoch=epoch,
        batch_in_epoch=batch_in_epoch,
        step=step,
        best_map50=best_map50,
        best_end_to_end_map50=best_end_to_end_map50,
        running_losses=running_losses,
        running_loss_steps=running_loss_steps,
        total_epochs=total_epochs,
        steps_per_epoch=steps_per_epoch,
        training_options=training_options,
        distributed_rng_states=distributed_rng_states,
    )
    LOGGER.info("Checkpoint saved: %s", target)


def load_pretrained_tower(model: TRHashObjectDetector, checkpoint: Path) -> None:
    """Load compatible tower parameters and resize learned patch positions.

    Routing tables are deterministic buffers derived from the target image
    grid, so they are deliberately regenerated instead of copied.
    """

    state = load_file(str(checkpoint / "tower.safetensors"))
    parameters = dict(model.tower.named_parameters())
    position_name = "position_embedding"
    if position_name in state and state[position_name].shape != parameters[position_name].shape:
        old_positions = state[position_name]
        old_grid = math.isqrt(old_positions.shape[1])
        new_grid = math.isqrt(parameters[position_name].shape[1])
        if (
            old_grid**2 != old_positions.shape[1]
            or new_grid**2 != parameters[position_name].shape[1]
        ):
            raise ValueError("vision position embeddings must form square patch grids")
        state[position_name] = (
            F.interpolate(
                old_positions.reshape(1, old_grid, old_grid, old_positions.shape[-1])
                .permute(0, 3, 1, 2)
                .float(),
                size=(new_grid, new_grid),
                mode="bicubic",
                align_corners=False,
            )
            .permute(0, 2, 3, 1)
            .reshape(1, new_grid**2, old_positions.shape[-1])
            .to(old_positions.dtype)
        )
    for name, parameter in parameters.items():
        source = state.get(name)
        if (
            source is not None
            and name.startswith(("position_rows.", "position_cols."))
            and source.shape != parameter.shape
        ):
            if source.ndim != 3 or source.shape[:2] != parameter.shape[:2]:
                continue
            state[name] = F.interpolate(
                source.float(),
                size=parameter.shape[-1],
                mode="linear",
                align_corners=False,
            ).to(source.dtype)

    compatible = {
        name: state[name]
        for name, parameter in parameters.items()
        if name in state and state[name].shape == parameter.shape
    }
    skipped = sorted(set(parameters) - set(compatible))
    model.tower.load_state_dict(compatible, strict=False)
    LOGGER.info(
        "Loaded %d pretrained tower parameters from %s%s",
        len(compatible),
        checkpoint,
        f" (skipped incompatible: {', '.join(skipped)})" if skipped else "",
    )


def load_class_mapping(path: Path | None) -> Dict[int, int] | None:
    if path is None:
        return None
    values = json.loads(path.read_text())
    if not isinstance(values, dict):
        raise ValueError("class map must be a JSON object of target_id: source_id")
    try:
        return {int(target): int(source) for target, source in values.items()}
    except (TypeError, ValueError) as error:
        raise ValueError("class map IDs must be integers") from error


def load_pretrained_detector(
    model: TRHashObjectDetector,
    checkpoint: Path,
    *,
    class_mapping: Dict[int, int] | None = None,
) -> None:
    """Transfer a v6 detector and optionally remap classification rows."""

    source_config = TRHashDetectorConfig.from_dict(
        json.loads((checkpoint / "config.json").read_text())
    )
    source_weights = (
        checkpoint / "ema.safetensors"
        if (checkpoint / "ema.safetensors").is_file()
        else checkpoint / "model.safetensors"
    )
    source_state = load_file(str(source_weights))
    parameters = dict(model.named_parameters())

    position_name = "tower.position_embedding"
    if (
        position_name in source_state
        and position_name in parameters
        and source_state[position_name].shape != parameters[position_name].shape
    ):
        old_positions = source_state[position_name]
        old_grid = math.isqrt(old_positions.shape[1])
        new_grid = math.isqrt(parameters[position_name].shape[1])
        if (
            old_grid**2 != old_positions.shape[1]
            or new_grid**2 != parameters[position_name].shape[1]
        ):
            raise ValueError("vision position embeddings must form square patch grids")
        source_state[position_name] = (
            F.interpolate(
                old_positions.reshape(1, old_grid, old_grid, old_positions.shape[-1])
                .permute(0, 3, 1, 2)
                .float(),
                size=(new_grid, new_grid),
                mode="bicubic",
                align_corners=False,
            )
            .permute(0, 2, 3, 1)
            .reshape(1, new_grid**2, old_positions.shape[-1])
            .to(old_positions.dtype)
        )
    for name, parameter in parameters.items():
        source = source_state.get(name)
        if (
            source is not None
            and name.startswith(("tower.position_rows.", "tower.position_cols."))
            and source.shape != parameter.shape
        ):
            if source.ndim != 3 or source.shape[:2] != parameter.shape[:2]:
                continue
            source_state[name] = F.interpolate(
                source.float(),
                size=parameter.shape[-1],
                mode="linear",
                align_corners=False,
            ).to(source.dtype)

    output_parameter_names = {
        f"head.classification_heads.{level}.3.{suffix}"
        for level in range(len(model.config.grid_sizes))
        for suffix in ("weight", "bias")
    }
    pyramid_compatible = source_config.grid_sizes == model.config.grid_sizes
    compatible = {}
    adapted_outputs = []
    for name, parameter in parameters.items():
        source = source_state.get(name)
        if source is None:
            continue
        if not pyramid_compatible and name.startswith("head."):
            continue
        is_output = name in output_parameter_names
        if not is_output and source.shape == parameter.shape:
            compatible[name] = source
            continue
        if (
            not is_output
            or source.ndim != parameter.ndim
            or source.shape[1:] != parameter.shape[1:]
        ):
            continue
        adapted = parameter.detach().cpu().clone()
        if class_mapping is None and source_config.num_classes == model.config.num_classes:
            adapted.copy_(source)
        elif class_mapping is not None:
            for target_class, source_class in class_mapping.items():
                if not 0 <= target_class < model.config.num_classes:
                    raise ValueError(f"target class ID out of range: {target_class}")
                if not 0 <= source_class < source_config.num_classes:
                    raise ValueError(f"source class ID out of range: {source_class}")
                adapted[target_class].copy_(source[source_class])
        compatible[name] = adapted
        adapted_outputs.append(name)

    model.load_state_dict(compatible, strict=False)
    skipped = sorted(set(parameters) - set(compatible))
    LOGGER.info(
        "Transferred detector from %s: %d parameter tensors, %d adapted output tensors%s",
        checkpoint,
        len(compatible),
        len(adapted_outputs),
        f" (left initialized: {', '.join(skipped)})" if skipped else "",
    )


def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    centers, sizes = boxes[:, :2], boxes[:, 2:]
    return torch.cat((centers - sizes / 2, centers + sizes / 2), dim=-1)


def _match_image_detections(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    target_boxes: torch.Tensor,
    target_labels: torch.Tensor,
    num_classes: int,
) -> Dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Greedily match one image's detections after computing IoUs once.

    Only predictions that overlap any target by at least 0.5 enter the small
    greedy loop. All other predictions are known false positives immediately.
    """

    matches_by_class = {}
    for class_id in range(num_classes):
        prediction_mask = labels == class_id
        class_scores = scores[prediction_mask]
        class_boxes = boxes[prediction_mask]
        order = torch.argsort(class_scores, descending=True)
        class_scores = class_scores[order]
        class_boxes = class_boxes[order]
        matches = torch.zeros(len(class_scores), dtype=torch.float32)
        class_targets = target_boxes[target_labels == class_id]

        if len(class_boxes) and len(class_targets):
            top_left = torch.maximum(class_boxes[:, None, :2], class_targets[None, :, :2])
            bottom_right = torch.minimum(class_boxes[:, None, 2:], class_targets[None, :, 2:])
            intersections = (bottom_right - top_left).clamp_min(0).prod(-1)
            prediction_areas = (class_boxes[:, 2:] - class_boxes[:, :2]).clamp_min(0).prod(-1)
            target_areas = (class_targets[:, 2:] - class_targets[:, :2]).clamp_min(0).prod(-1)
            ious = intersections / (
                prediction_areas[:, None] + target_areas[None, :] - intersections
            ).clamp_min(1e-9)
            candidate_predictions = torch.nonzero(
                (ious >= 0.5).any(dim=1), as_tuple=False
            ).flatten()
            used_targets = torch.zeros(len(class_targets), dtype=torch.bool)
            for prediction_index in candidate_predictions.tolist():
                available_ious = ious[prediction_index].masked_fill(used_targets, -1.0)
                best_iou, target_index = available_ious.max(dim=0)
                if float(best_iou) >= 0.5:
                    matches[prediction_index] = 1.0
                    used_targets[int(target_index)] = True

        matches_by_class[class_id] = (class_scores, matches)
    return matches_by_class


def _average_precision_from_matches(
    scores: torch.Tensor,
    true_positives: torch.Tensor,
    total_ground_truth: int,
) -> float:
    if total_ground_truth == 0:
        return float("nan")
    if not len(scores):
        return 0.0
    order = torch.argsort(scores, descending=True)
    true_positive_cumulative = true_positives[order].cumsum(0)
    false_positive_cumulative = (1.0 - true_positives[order]).cumsum(0)
    recall = true_positive_cumulative / total_ground_truth
    precision = true_positive_cumulative / (
        true_positive_cumulative + false_positive_cumulative
    ).clamp_min(1e-9)
    recall = torch.cat((torch.tensor([0.0]), recall, torch.tensor([1.0])))
    precision = torch.cat((torch.tensor([1.0]), precision, torch.tensor([0.0])))
    precision = torch.cummax(precision.flip(0), dim=0).values.flip(0)
    changing = torch.nonzero(recall[1:] != recall[:-1], as_tuple=False).flatten()
    return float(((recall[changing + 1] - recall[changing]) * precision[changing + 1]).sum())


def verify_loader_cardinality(
    loader: DataLoader,
    processed_batches: int,
    distributed: DistributedContext | None,
    *,
    phase: str,
    expected_batches: int | None = None,
) -> None:
    """Refuse silently truncated iterable streams on any DDP rank."""

    expected_batches = len(loader) if expected_batches is None else expected_batches
    local = {
        "rank": 0 if distributed is None else distributed.rank,
        "processed": processed_batches,
        "expected": expected_batches,
    }
    reports = [local] if distributed is None else distributed.all_gather_objects(local)
    invalid = [report for report in reports if report["processed"] != report["expected"]]
    if invalid:
        details = ", ".join(
            f"rank {report['rank']}: {report['processed']}/{report['expected']}"
            for report in invalid
        )
        raise RuntimeError(f"{phase} loader ended before its declared cardinality ({details})")


@torch.inference_mode()
def evaluate_detector(
    model: TRHashObjectDetector,
    loader: DataLoader,
    device: torch.device,
    *,
    confidence_threshold: float,
    use_amp: bool = False,
    show_progress: bool = False,
    max_detections: int = 100,
    distributed: DistributedContext | None = None,
    end_to_end: bool = False,
    coco_annotations_path: Path | None = None,
    coco_category_ids: Sequence[int] | None = None,
    coco_backend: str = "auto",
) -> Dict[str, Any]:
    model.eval()
    official_coco = coco_annotations_path is not None
    metrics = (
        None
        if official_coco
        else DetectionMetricsAccumulator(
            model.config.num_classes,
            model.config.image_size,
        )
    )
    coco_predictions: list[dict[str, Any]] = []
    coco_image_ids: list[int] = []

    progress = tqdm(
        loader,
        desc="detector validation",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
        disable=False if show_progress else True,
    )
    processed_batches = 0
    for batch in progress:
        pixel_values, targets = batch[:2]
        sample_metadata = batch[2] if len(batch) == 3 else None
        processed_batches += 1
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
        with autocast:
            model_inputs = pixel_values.to(
                device,
                non_blocking=device.type == "cuda",
            )
            if end_to_end:
                detections = model.predict_end_to_end(
                    model_inputs,
                    confidence_threshold=0.001,
                    max_detections=max_detections,
                )
            else:
                detections = model.predict(
                    model_inputs,
                    confidence_threshold=0.001,
                    iou_threshold=0.5,
                    postprocess_on_cpu=device.type == "mps",
                    max_detections=max_detections,
                    nms_free=False,
                )
        if metrics is not None:
            for detection, image_targets in zip(detections, targets):
                metrics.update(
                    detection["boxes"],
                    detection["scores"],
                    detection["labels"],
                    image_targets,
                )
        if official_coco:
            if sample_metadata is None or coco_category_ids is None:
                raise ValueError("official COCO evaluation requires sample metadata and categories")
            coco_predictions.extend(
                detections_to_coco(detections, sample_metadata, coco_category_ids)
            )
            coco_image_ids.extend(int(sample["image_id"]) for sample in sample_metadata)
    verify_loader_cardinality(
        loader,
        processed_batches,
        distributed,
        phase="validation",
    )
    if distributed is not None and distributed.enabled:
        if metrics is not None:
            states = distributed.all_gather_objects(metrics.state_dict())
            metrics = DetectionMetricsAccumulator(
                model.config.num_classes,
                model.config.image_size,
            )
            for state in states:
                metrics.merge_state_dict(state)
        if official_coco:
            prediction_parts = distributed.all_gather_objects(coco_predictions)
            image_id_parts = distributed.all_gather_objects(coco_image_ids)
            coco_predictions = [record for part in prediction_parts for record in part]
            coco_image_ids = [image_id for part in image_id_parts for image_id in part]
    computed: Dict[str, Any] = metrics.compute(confidence_threshold) if metrics else {}
    if official_coco:
        official = (
            evaluate_coco_predictions(
                coco_annotations_path,
                coco_predictions,
                coco_image_ids,
                backend=coco_backend,
                max_detections=max_detections,
                confidence_threshold=confidence_threshold,
            )
            if distributed is None or distributed.is_main
            else None
        )
        if distributed is not None:
            official = distributed.broadcast_object(official)
        computed.update(official)
    model.train()
    return computed


def cosine_schedule(step: int, *, warmup_steps: int, total_steps: int, min_ratio: float) -> float:
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def should_validate_epoch(epoch: int, total_epochs: int, eval_every: int) -> bool:
    """Return true on the configured cadence and unconditionally at the end."""

    completed_epochs = epoch + 1
    return completed_epochs % eval_every == 0 or completed_epochs == total_epochs


def validation_selection_score(metrics: Mapping[str, Any]) -> tuple[str, float]:
    """Select checkpoints with official COCO mAP50-95 when it is available."""

    key = "coco_map50_95" if metrics.get("official_coco") else "map50"
    if key not in metrics:
        raise ValueError(f"validation metrics are missing checkpoint selection key: {key}")
    return key, float(metrics[key])


def mosaic_source_images_per_sample(mosaic_probability: float, mosaic_tiles: int = 4) -> float:
    """Expected source-image count composited per probabilistic Mosaic sample.

    This is informational only (logging / dataset statistics): it no longer
    feeds the step schedule. Mosaic tiles are downscaled to fit the shared
    canvas, so crediting them 1:1 against un-packed, full-resolution steps
    silently traded step count for per-tile quality. ``packed_epochs`` is the
    sole, explicit lever on step count now (see ``detector_epoch_steps``).
    """

    if not 0.0 <= mosaic_probability <= 1.0:
        raise ValueError("mosaic probability must be in [0, 1]")
    if mosaic_tiles < 1:
        raise ValueError("mosaic_tiles must be positive")
    return 1.0 + (mosaic_tiles - 1) * mosaic_probability


def mosaic_is_active_epoch(
    epoch: int,
    *,
    total_epochs: int,
    close_mosaic_epochs: int,
    mosaic_probability: float,
) -> bool:
    if mosaic_probability <= 0.0:
        return False
    if not total_epochs or not close_mosaic_epochs:
        return True
    return epoch < max(total_epochs - close_mosaic_epochs, 0)


def detector_epoch_steps(
    full_loader_steps: int,
    epoch: int,
    *,
    total_epochs: int,
    mosaic_probability: float,
    close_mosaic_epochs: int,
    mosaic_packed_epoch: bool,
    packed_epochs: int = 1,
) -> int:
    """Return training microbatches for one epoch.

    ``packed_epochs`` is a step-count divisor independent of Mosaic: it is
    *not* scaled by how many source images a Mosaic canvas composites, since
    packed tiles are downscaled and are not a full-quality substitute for an
    un-packed step. ``packed_epochs=1`` reproduces the natural, un-packed
    step count; higher values divide it further.
    """

    active = mosaic_packed_epoch and mosaic_is_active_epoch(
        epoch,
        total_epochs=total_epochs,
        close_mosaic_epochs=close_mosaic_epochs,
        mosaic_probability=mosaic_probability,
    )
    if packed_epochs < 1:
        raise ValueError("packed_epochs must be positive")
    exposure = float(packed_epochs) if active else 1.0
    return resolve_packed_epoch_schedule(
        full_steps=full_loader_steps,
        exposure_factors=(exposure,),
        enabled=active,
    ).steps[0]


def detector_step_schedule(
    full_loader_steps: int,
    *,
    total_epochs: int,
    mosaic_probability: float,
    close_mosaic_epochs: int,
    mosaic_packed_epoch: bool,
    packed_epochs: int = 1,
) -> tuple[int, ...]:
    if packed_epochs < 1:
        raise ValueError("packed_epochs must be positive")
    exposures = tuple(
        (
            float(packed_epochs)
            if mosaic_packed_epoch
            and mosaic_is_active_epoch(
                epoch,
                total_epochs=total_epochs,
                close_mosaic_epochs=close_mosaic_epochs,
                mosaic_probability=mosaic_probability,
            )
            else 1.0
        )
        for epoch in range(total_epochs)
    )
    return resolve_packed_epoch_schedule(
        full_steps=full_loader_steps,
        exposure_factors=exposures,
        enabled=mosaic_packed_epoch,
    ).steps


def resolve_accumulation_steps(
    *,
    per_device_batch_size: int,
    world_size: int,
    nominal_batch_size: int,
    requested_steps: int = 0,
) -> int:
    """Resolve an optimizer-step cadence from the requested global batch budget."""

    if per_device_batch_size <= 0 or world_size <= 0:
        raise ValueError("batch size and world size must be positive")
    if nominal_batch_size < 0 or requested_steps < 0:
        raise ValueError("nominal batch size and accumulation steps cannot be negative")
    if requested_steps:
        return requested_steps
    if not nominal_batch_size:
        return 1
    global_batch_size = per_device_batch_size * world_size
    return max(round(nominal_batch_size / global_batch_size), 1)


def optimizer_step_schedule(
    microbatch_schedule: Sequence[int],
    accumulation_steps: int,
) -> tuple[int, ...]:
    """Convert per-epoch microbatch counts to optimizer-step counts."""

    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")
    if any(steps <= 0 for steps in microbatch_schedule):
        raise ValueError("every epoch must contain at least one microbatch")
    return tuple(math.ceil(steps / accumulation_steps) for steps in microbatch_schedule)


def accumulation_group(
    batch_index: int,
    *,
    epoch_batches: int,
    accumulation_steps: int,
) -> tuple[int, bool]:
    """Return the current group's divisor and whether this microbatch steps the optimizer."""

    if epoch_batches <= 0 or accumulation_steps <= 0:
        raise ValueError("epoch_batches and accumulation_steps must be positive")
    if not 0 <= batch_index < epoch_batches:
        raise ValueError("batch_index is outside the epoch")
    group_start = batch_index - batch_index % accumulation_steps
    group_size = min(accumulation_steps, epoch_batches - group_start)
    boundary = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == epoch_batches
    return group_size, boundary


def normalized_weight_decay(
    weight_decay: float,
    *,
    global_batch_size: int,
    accumulation_steps: int,
    nominal_batch_size: int,
) -> float:
    """Scale weight decay to the realized effective batch, matching common YOLO recipes."""

    if weight_decay < 0.0:
        raise ValueError("weight_decay cannot be negative")
    if global_batch_size <= 0 or accumulation_steps <= 0:
        raise ValueError("global batch size and accumulation steps must be positive")
    if nominal_batch_size < 0:
        raise ValueError("nominal_batch_size cannot be negative")
    if not nominal_batch_size:
        return weight_decay
    return weight_decay * global_batch_size * accumulation_steps / nominal_batch_size


def resolve_warmup_steps(
    *,
    requested_steps: int,
    warmup_epochs: float | None,
    steps_in_first_epoch: int,
    total_steps: int,
    unpacked_total_steps: int | None = None,
) -> int:
    """Resolve warmup in optimizer-step units and clamp it to the run horizon.

    ``requested_steps`` is a raw optimizer-step count tuned against an
    unpacked schedule. When ``warmup_epochs`` is not given and mosaic packing
    shrinks the realized optimizer-step budget below that unpacked horizon
    (``unpacked_total_steps``), the raw count is rescaled by the same ratio so
    warmup still covers the same fraction of the run instead of silently
    covering a larger fraction of the shortened packed schedule.
    """

    if requested_steps < 0 or steps_in_first_epoch <= 0 or total_steps <= 0:
        raise ValueError("invalid warmup or optimizer-step horizon")
    if unpacked_total_steps is not None and unpacked_total_steps <= 0:
        raise ValueError("unpacked_total_steps must be positive")
    if warmup_epochs is not None:
        if warmup_epochs < 0.0:
            raise ValueError("warmup_epochs cannot be negative")
        requested_steps = round(warmup_epochs * steps_in_first_epoch)
    elif unpacked_total_steps and unpacked_total_steps > total_steps:
        requested_steps = round(requested_steps * total_steps / unpacked_total_steps)
    return min(requested_steps, total_steps)


def format_loss_metrics_for_logging(averages: Mapping[str, float]) -> Dict[str, float]:
    """Expose a stationary monitoring loss without hiding the optimized objective.

    Progressive box/quality weights and the one-to-one ramp make the actual
    optimization objective non-stationary. ``loss`` is therefore the fixed-weight
    monitoring value, while ``optimization_loss`` remains the exact value used for
    backpropagation.
    """

    metrics = dict(averages)
    optimization_loss = metrics.pop("loss")
    metrics["optimization_loss"] = optimization_loss
    metrics["loss"] = metrics.pop("monitor_loss", optimization_loss)
    if "one_to_many_monitor_loss" in metrics:
        metrics["one_to_many_optimization_loss"] = metrics.pop("one_to_many_loss")
        metrics["one_to_many_loss"] = metrics.pop("one_to_many_monitor_loss")
    if "one_to_one_monitor_loss" in metrics:
        metrics["one_to_one_optimization_loss"] = metrics.pop("one_to_one_loss")
        metrics["one_to_one_loss"] = metrics.pop("one_to_one_monitor_loss")
    return metrics


def main() -> None:
    args = parse_args()
    provenance = resolve_initialization_provenance(args)
    distributed = DistributedContext.initialize(resolve_device(args.device))
    if args.compile_mode != "none" and distributed.enabled and args.multi_scale_min is not None:
        raise ValueError(
            "--compile-mode is not supported with DDP multi-scale training; "
            "use a fixed resolution, one GPU, or --compile-mode none"
        )
    logging.basicConfig(
        level=logging.INFO if distributed.is_main else logging.ERROR,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[TqdmLoggingHandler()],
        force=True,
    )
    torch.manual_seed(args.seed + distributed.rank)
    device = distributed.device
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    LOGGER.info(
        "Initialization: %s (implementation=%s dataset=%s external_checkpoint=%s)",
        provenance["initialization"],
        provenance["implementation"],
        provenance["dataset"],
        provenance["external_checkpoint"],
    )

    validation_dataset = None
    epoch_dataset = None
    dataset_sources = (
        args.video_annotations is not None,
        args.yolo_images is not None,
        args.annotations is not None,
        args.hf_detection_dataset is not None,
    )
    if args.video_annotations is None and args.video_images is not None:
        raise ValueError("--video-images requires --video-annotations")
    if args.video_annotations is None and any(
        value is not None
        for value in (
            args.validation_video_annotations,
            args.validation_video_images,
        )
    ):
        raise ValueError("video validation paths require a video training dataset")
    if sum(dataset_sources) > 1:
        raise ValueError(
            "video, YOLO, COCO, and Hugging Face dataset sources are mutually exclusive"
        )
    if args.video_annotations is not None:
        if args.video_images is None:
            raise ValueError("--video-images is required alongside --video-annotations")
        if not args.video_motion:
            raise ValueError("COCO-Video training requires --video-motion")
        unsupported_video_augmentations = {
            "mosaic": args.mosaic,
            "mixup": args.mixup,
            "copy-paste": args.copy_paste,
            "random-erasing": args.random_erasing,
        }
        enabled = [
            name for name, probability in unsupported_video_augmentations.items() if probability
        ]
        if enabled:
            raise ValueError(
                "video clips require synchronized augmentation; disable: " + ", ".join(enabled)
            )
        dataset = CocoVideoDetectionDataset(
            args.video_annotations,
            args.video_images,
            image_size=args.image_size,
            clip_frames=args.video_clip_frames,
            frame_stride=args.video_frame_stride,
            augmentation=args.augmentation,
            seed=args.seed,
        )
        epoch_dataset = dataset
        num_classes = dataset.num_classes
        validation_video_paths = (
            args.validation_video_annotations,
            args.validation_video_images,
        )
        if all(path is not None for path in validation_video_paths):
            validation_dataset = CocoVideoDetectionDataset(
                args.validation_video_annotations,
                args.validation_video_images,
                image_size=args.image_size,
                clip_frames=args.video_clip_frames,
                frame_stride=args.video_frame_stride,
            )
            if validation_dataset.num_classes != num_classes:
                raise ValueError("COCO-Video train/validation class counts differ")
        elif any(path is not None for path in validation_video_paths):
            raise ValueError("both validation video annotations and image dirs are required")
        LOGGER.info(
            "COCO-Video dataset: %d clips, %d frames/clip, %d classes",
            len(dataset),
            args.video_clip_frames,
            num_classes,
        )
    elif args.hf_detection_dataset is not None:
        if args.hf_detection_train_examples <= 0:
            raise ValueError("--hf-detection-train-examples must be positive")
        if args.hf_detection_validation_examples <= 0:
            raise ValueError("--hf-detection-validation-examples must be positive")
        if args.hf_detection_num_classes <= 0:
            raise ValueError("--hf-detection-num-classes must be positive")
        if args.workers:
            raise ValueError("HF detection streaming currently requires --workers 0")
        unsupported_streaming_augmentations = {
            "mosaic": args.mosaic,
            "mixup": args.mixup,
            "copy-paste": args.copy_paste,
            "random-erasing": args.random_erasing,
        }
        enabled = [
            name for name, probability in unsupported_streaming_augmentations.items() if probability
        ]
        if enabled:
            raise ValueError(
                "HF detection streaming supports synchronized per-image augmentation only; "
                "disable: " + ", ".join(enabled)
            )
        common_hf = {
            "dataset_id": args.hf_detection_dataset,
            "num_classes": args.hf_detection_num_classes,
            "image_size": args.image_size,
            "rank": distributed.rank,
            "world_size": distributed.world_size,
            "image_column": args.hf_detection_image_column,
            "annotations_column": args.hf_detection_annotations_column,
            "metadata_file_glob": args.hf_detection_metadata_file_glob,
            "category_id_offset": args.hf_detection_category_offset,
        }
        dataset = HuggingFaceDetectionDataset(
            split=args.hf_detection_train_split,
            num_examples=args.hf_detection_train_examples,
            augmentation=args.augmentation,
            seed=args.seed,
            shuffle_buffer=args.hf_detection_shuffle_buffer,
            **common_hf,
        )
        validation_dataset = HuggingFaceDetectionDataset(
            split=args.hf_detection_validation_split,
            num_examples=args.hf_detection_validation_examples,
            **common_hf,
        )
        epoch_dataset = dataset
        num_classes = args.hf_detection_num_classes
        LOGGER.info(
            "HF detection dataset %s: %d train, %d validation, %d classes (streaming)",
            args.hf_detection_dataset,
            args.hf_detection_train_examples,
            args.hf_detection_validation_examples,
            num_classes,
        )
    elif args.yolo_images is not None:
        if args.yolo_labels is None:
            raise ValueError("--yolo-labels is required alongside --yolo-images")
        train_full = YoloDetectionDataset(
            args.yolo_images,
            args.yolo_labels,
            image_size=args.image_size,
            augmentation=args.augmentation,
            augmentation_backend=args.augmentation_backend,
            image_backend=args.image_backend,
            seed=args.seed,
            mosaic_probability=args.mosaic,
            mixup_probability=args.mixup,
            copy_paste_probability=args.copy_paste,
            random_erasing_probability=args.random_erasing,
            total_epochs=args.epochs,
            close_mosaic_epochs=args.close_mosaic_epochs,
            mosaic_tiles=args.mosaic_tiles,
            mosaic_canvas_size=args.mosaic_canvas_size,
        )
        validation_paths = (
            args.validation_yolo_images,
            args.validation_yolo_labels,
        )
        if all(value is not None for value in validation_paths):
            dataset = train_full
            validation_dataset = YoloDetectionDataset(
                args.validation_yolo_images,
                args.validation_yolo_labels,
                image_size=args.image_size,
                image_backend=args.image_backend,
            )
        elif any(value is not None for value in validation_paths):
            raise ValueError("both validation YOLO image and label dirs are required")
        else:
            if not 0.0 < args.validation_fraction < 1.0:
                raise ValueError("--validation-fraction must be between 0 and 1")
            if len(train_full) < 2:
                raise ValueError("at least two YOLO images are required for an automatic split")
            validation_full = YoloDetectionDataset(
                args.yolo_images,
                args.yolo_labels,
                image_size=args.image_size,
                image_backend=args.image_backend,
            )
            generator = torch.Generator().manual_seed(args.validation_seed)
            indices = torch.randperm(len(train_full), generator=generator).tolist()
            validation_count = min(
                max(round(len(indices) * args.validation_fraction), 1),
                len(indices) - 1,
            )
            dataset = Subset(train_full, indices[validation_count:])
            validation_dataset = Subset(validation_full, indices[:validation_count])
        epoch_dataset = train_full
        num_classes = train_full.num_classes
        LOGGER.info(
            "YOLO dataset: %d train, %d validation, %d classes",
            len(dataset),
            len(validation_dataset),
            num_classes,
        )
    elif args.annotations is not None:
        if args.images is None:
            raise ValueError("--images is required alongside --annotations")
        dataset = CocoDetectionDataset(
            args.annotations,
            args.images,
            image_size=args.image_size,
            augmentation=args.augmentation,
            augmentation_backend=args.augmentation_backend,
            image_backend=args.image_backend,
            seed=args.seed,
            mosaic_probability=args.mosaic,
            mixup_probability=args.mixup,
            copy_paste_probability=args.copy_paste,
            random_erasing_probability=args.random_erasing,
            total_epochs=args.epochs,
            close_mosaic_epochs=args.close_mosaic_epochs,
            mosaic_tiles=args.mosaic_tiles,
            mosaic_canvas_size=args.mosaic_canvas_size,
        )
        epoch_dataset = dataset
        num_classes = dataset.num_classes
        validation_coco_paths = (args.validation_annotations, args.validation_images)
        if all(path is not None for path in validation_coco_paths):
            validation_dataset = CocoDetectionDataset(
                args.validation_annotations,
                args.validation_images,
                image_size=args.image_size,
                image_backend=args.image_backend,
                return_metadata=args.eval_backend != "internal",
            )
            if validation_dataset.num_classes != num_classes:
                raise ValueError("COCO train/validation class counts differ")
        elif any(path is not None for path in validation_coco_paths):
            raise ValueError("both validation COCO annotations and image dirs are required")
        LOGGER.info("COCO dataset: %d images, %d classes", len(dataset), num_classes)
    else:
        dataset = SyntheticShapesDataset(
            args.synthetic_samples,
            image_size=args.image_size,
            seed=args.seed,
            resample_each_epoch=not args.fixed_synthetic_data,
        )
        validation_dataset = SyntheticShapesDataset(
            args.validation_samples,
            image_size=args.image_size,
            seed=args.validation_seed,
        )
        num_classes = args.num_classes
        epoch_dataset = dataset
        LOGGER.info("Synthetic dataset: %d images, %d classes", len(dataset), num_classes)

    if distributed.is_main:
        args.output.mkdir(parents=True, exist_ok=True)
    distributed.barrier()

    if args.eval_every <= 0:
        raise ValueError("--eval-every must be positive")
    if args.eval_max_detections <= 0:
        raise ValueError("--eval-max-detections must be positive")
    if args.eval_batch_size < 0:
        raise ValueError("--eval-batch-size cannot be negative")
    coco_validation = (
        validation_dataset if isinstance(validation_dataset, CocoDetectionDataset) else None
    )
    official_coco_evaluation = coco_validation is not None and args.eval_backend != "internal"
    if args.eval_backend in {"pycocotools", "faster"} and coco_validation is None:
        raise ValueError("official COCO evaluation requires a COCO validation dataset")
    if official_coco_evaluation and args.eval_max_detections != 100:
        raise ValueError("official comparable COCO evaluation requires --eval-max-detections 100")
    if args.augmentation_backend == "albumentations" and not isinstance(
        epoch_dataset, (CocoDetectionDataset, YoloDetectionDataset)
    ):
        raise ValueError("Albumentations currently supports random-access COCO/YOLO datasets")
    resolved_eval_backend = args.eval_backend if official_coco_evaluation else "internal"
    LOGGER.info(
        "Detection data backends: images=%s augmentation=%s evaluation=%s",
        args.image_backend,
        args.augmentation_backend,
        resolved_eval_backend,
    )
    multi_scale_max = args.multi_scale_max or args.image_size
    multi_scale_step = args.multi_scale_step or args.patch_size
    if args.multi_scale_min:
        if not args.multi_scale_min <= multi_scale_max <= args.image_size:
            raise ValueError("multi-scale resolutions must satisfy min <= max <= image-size")
        if (
            args.multi_scale_min % args.patch_size
            or multi_scale_max % args.patch_size
            or multi_scale_step % args.patch_size
        ):
            raise ValueError("multi-scale resolutions must be divisible by patch-size")
    streaming_dataset = isinstance(dataset, IterableDataset)
    mosaic_packed_epoch = args.mosaic_packed_epoch and args.mosaic > 0.0
    if args.packed_epochs < 1:
        raise ValueError("--packed-epochs must be positive")
    mosaic_grid = math.isqrt(args.mosaic_tiles)
    if mosaic_grid < 2 or mosaic_grid * mosaic_grid != args.mosaic_tiles:
        raise ValueError("--mosaic-tiles must be a perfect square of at least 4")
    mosaic_canvas_size = args.mosaic_canvas_size or args.image_size
    if mosaic_canvas_size < args.image_size:
        raise ValueError("--mosaic-canvas-size must be at least --image-size")
    if mosaic_canvas_size % mosaic_grid:
        raise ValueError("--mosaic-canvas-size must be divisible by the Mosaic grid")
    if mosaic_packed_epoch:
        if streaming_dataset or not isinstance(
            dataset,
            (CocoDetectionDataset, YoloDetectionDataset, Subset),
        ):
            raise ValueError("--mosaic-packed-epoch requires a random-access COCO or YOLO dataset")
    shuffle_generator = torch.Generator()
    train_sampler = None if streaming_dataset else distributed.train_sampler(dataset, args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=not streaming_dataset and train_sampler is None,
        sampler=train_sampler,
        generator=(shuffle_generator if not streaming_dataset and train_sampler is None else None),
        num_workers=args.workers,
        # Recreate workers each epoch so dataset.set_epoch() reaches worker copies.
        persistent_workers=False,
        pin_memory=device.type == "cuda",
        collate_fn=collate_detection,
    )
    validation_loader = (
        DataLoader(
            validation_dataset,
            batch_size=args.eval_batch_size or args.batch_size,
            sampler=(
                None
                if isinstance(validation_dataset, IterableDataset)
                else distributed.eval_sampler(validation_dataset)
            ),
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
            pin_memory=device.type == "cuda",
            collate_fn=collate_detection,
        )
        if validation_dataset is not None
        else None
    )

    vision_precision = args.vision_precision
    if vision_precision == "auto":
        vision_precision = "fp32" if device.type == "mps" else "bf16"
    config = TRHashDetectorConfig(
        architecture_version=args.architecture_version,
        image_size=args.image_size,
        patch_size=args.patch_size,
        vision_hidden_size=args.vision_hidden_size,
        vision_layers=args.vision_layers,
        vision_heads=args.vision_heads,
        vision_num_experts=args.vision_num_experts,
        vision_top_k=args.vision_top_k,
        vision_shared_width=args.vision_shared_width,
        vision_expert_width=args.vision_expert_width,
        vision_stage_depths=tuple(args.vision_stage_depths),
        vision_window_size=args.vision_window_size,
        vision_precision=vision_precision,
        num_classes=num_classes,
        multi_scale=not args.single_scale,
        p2_head=args.p2_head,
        neck_mode=args.neck_mode,
        neck_normalized_fusion=args.neck_normalized_fusion,
        neck_repeats=args.neck_repeats,
        dynamic_assignment=not args.static_assignment,
        assignment_top_k=args.assignment_top_k,
        stal_enabled=not args.no_stal,
        progressive_loss_enabled=not args.no_progressive_loss,
        reg_max=args.reg_max,
        head_hidden_size=args.head_hidden_size,
        head_spatial_mixing=args.head_spatial_mixing,
        regression_logit_scale=args.regression_logit_scale,
        dfl_loss_weight=args.dfl_loss_weight,
        quality_focal_beta=args.quality_focal_beta,
        end_to_end=args.end_to_end,
        one_to_one_loss_weight=args.one_to_one_loss_weight,
        one_to_one_loss_start=args.one_to_one_loss_start,
        one_to_one_shared_gradient_scale=args.one_to_one_shared_gradient_scale,
        box_loss_weight=args.box_loss_weight,
        quality_loss_weight=args.quality_loss_weight,
        box_l1_weight=args.box_l1_weight,
        box_iou_weight=args.box_iou_weight,
        level_adapters_enabled=args.level_adapters,
        level_adapter_ratio=args.level_adapter_ratio,
        class_level_hash_gate_enabled=args.class_level_hash_gate,
        class_level_gate_temperature=args.class_level_gate_temperature,
        object_weighting_enabled=args.object_weighting,
        object_weighting_beta=args.object_weighting_beta,
        object_weighting_max=args.object_weighting_max,
        level_aux_loss_weight=args.level_aux_loss_weight,
        gate_calibration_loss_weight=args.gate_calibration_loss_weight,
        object_contrastive_loss_weight=args.object_contrastive_loss_weight,
        object_contrastive_temperature=args.object_contrastive_temperature,
        object_contrastive_dim=args.object_contrastive_dim,
        video_motion_enabled=args.video_motion,
        video_motion_hidden_size=args.video_motion_hidden_size,
        video_motion_scale_init=args.video_motion_scale_init,
    )
    model = TRHashObjectDetector(config).to(device)
    if config.object_weighting_enabled:
        if isinstance(dataset, HuggingFaceDetectionDataset):
            if distributed.is_main:
                object_count_cache = args.output / "object_bucket_counts.json"
                object_counts = load_object_bucket_count_cache(
                    object_count_cache,
                    dataset_id=dataset.dataset_id,
                    split=dataset.split,
                    num_examples=dataset.total_examples,
                    num_classes=dataset.num_classes,
                    category_id_offset=dataset.category_id_offset,
                    metadata_file_glob=dataset.metadata_file_glob,
                )
                if object_counts is None:
                    LOGGER.info("Scanning streamed annotations for object-weight buckets")
                    object_counts = dataset.object_bucket_counts()
                    save_object_bucket_count_cache(
                        object_count_cache,
                        object_counts,
                        dataset_id=dataset.dataset_id,
                        split=dataset.split,
                        num_examples=dataset.total_examples,
                        num_classes=dataset.num_classes,
                        category_id_offset=dataset.category_id_offset,
                        metadata_file_glob=dataset.metadata_file_glob,
                    )
                    LOGGER.info("Object-weight bucket cache saved: %s", object_count_cache)
                else:
                    LOGGER.info("Loaded object-weight bucket cache: %s", object_count_cache)
            else:
                object_counts = torch.zeros(num_classes, 3, 3, dtype=torch.float64)
            if distributed.enabled:
                object_counts = object_counts.to(device)
                torch.distributed.broadcast(object_counts, src=0)
                object_counts = object_counts.cpu()
            object_weights, object_counts = object_weight_table_from_counts(
                object_counts,
                beta=config.object_weighting_beta,
                max_weight=config.object_weighting_max,
            )
        else:
            object_weights, object_counts = build_object_weight_table(
                dataset,
                num_classes,
                beta=config.object_weighting_beta,
                max_weight=config.object_weighting_max,
            )
        model.set_object_weight_table(object_weights)
        observed_weights = object_weights[object_counts > 0]
        if observed_weights.numel():
            LOGGER.info(
                "Object weighting: %d observed class/size/density buckets, " "range %.3f..%.3f",
                int((object_counts > 0).sum()),
                float(observed_weights.min()),
                float(observed_weights.max()),
            )
    if args.resume is not None:
        saved_config = TRHashDetectorConfig.from_dict(
            json.loads((args.resume / "config.json").read_text())
        )
        config_mismatches = sorted(
            key
            for key in set(saved_config.to_dict()) | set(config.to_dict())
            if saved_config.to_dict().get(key) != config.to_dict().get(key)
        )
        if config_mismatches:
            raise ValueError(
                "exact resume requires unchanged model/loss config; mismatched: "
                + ", ".join(config_mismatches)
            )
        model.load_state_dict(load_file(str(args.resume / "model.safetensors")))
    elif args.detector_checkpoint is not None:
        if args.training_purpose == VISION_SUPERVISED_FINETUNING:
            source_config = TRHashDetectorConfig.from_dict(
                json.loads((args.detector_checkpoint / "config.json").read_text())
            )
            config_mismatches = sorted(
                key
                for key, value in source_config.to_dict().items()
                if config.to_dict().get(key) != value
            )
            if config_mismatches:
                raise ValueError(
                    "vision-supervised-finetuning requires the source architecture; "
                    "mismatched: " + ", ".join(config_mismatches)
                )
        load_pretrained_detector(
            model,
            args.detector_checkpoint,
            class_mapping=load_class_mapping(args.class_map),
        )
    elif args.backbone_checkpoint is not None:
        load_pretrained_tower(model, args.backbone_checkpoint)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    LOGGER.info("Model: %.2fM parameters", model.num_parameters() / 1e6)
    LOGGER.info(
        "Training purpose: %s full_parameter=%s trainable=%.2fM/%.2fM",
        args.training_purpose,
        args.training_purpose == VISION_SUPERVISED_FINETUNING,
        trainable_parameters / 1e6,
        model.num_parameters() / 1e6,
    )
    backend_summary = vision_backend_summary(
        model,
        device.type,
        require_triton=args.require_triton,
    )
    LOGGER.info(
        "TR-Hash vision backend: %s (requested=%s experts=%d top_k=%d "
        "shared_width=%d expert_width=%d active_width=%d stored_width=%d precision=%s)",
        backend_summary["selected_backend"],
        backend_summary["requested_backend"],
        backend_summary["experts"],
        backend_summary["top_k"],
        backend_summary["shared_width"],
        backend_summary["expert_width"],
        backend_summary["active_width"],
        backend_summary["stored_width"],
        backend_summary["precision"],
    )
    if args.backbone_lr_multiplier <= 0.0:
        raise ValueError("--backbone-lr-multiplier must be positive")
    if args.expert_lr_multiplier <= 0.0:
        raise ValueError("--expert-lr-multiplier must be positive")
    if args.one_to_one_lr_multiplier <= 0.0:
        raise ValueError("--one-to-one-lr-multiplier must be positive")

    global_batch_size = args.batch_size * distributed.world_size
    accumulation_steps = resolve_accumulation_steps(
        per_device_batch_size=args.batch_size,
        world_size=distributed.world_size,
        nominal_batch_size=args.nominal_batch_size,
        requested_steps=args.accumulation_steps,
    )
    effective_batch_size = global_batch_size * accumulation_steps
    effective_weight_decay = normalized_weight_decay(
        args.weight_decay,
        global_batch_size=global_batch_size,
        accumulation_steps=accumulation_steps,
        nominal_batch_size=args.nominal_batch_size,
    )

    def parameter_learning_rate(name: str) -> tuple[float, str]:
        is_backbone = name.startswith("tower.")
        is_expert = ".mlp.expert_" in name
        lr = args.lr
        if is_backbone:
            lr *= args.backbone_lr_multiplier
        if is_expert:
            lr *= args.expert_lr_multiplier
        if name.startswith("one_to_one_head."):
            lr *= args.one_to_one_lr_multiplier
        scope = "backbone" if is_backbone else "task"
        if name.startswith("one_to_one_head."):
            suffix = "_one_to_one"
        else:
            suffix = "_experts" if is_expert else ""
        return lr, f"{scope}{suffix}"

    parameter_groups = build_musgd_parameter_groups(
        model,
        learning_rate=parameter_learning_rate,
        momentum=args.momentum,
        weight_decay=effective_weight_decay,
    )
    optimizer = MuSGD(
        parameter_groups,
        muon_weight=args.musgd_muon_weight,
        sgd_weight=args.musgd_sgd_weight,
    )
    LOGGER.info(
        "Optimizer: MuSGD (task_lr=%.2e one_to_one_lr=%.2e backbone_lr=%.2e "
        "backbone_expert_lr=%.2e momentum=%.3f muon=%.2f sgd=%.2f "
        "global_batch=%d accumulation=%d effective_batch=%d weight_decay=%.3g)",
        args.lr,
        args.lr * args.one_to_one_lr_multiplier,
        args.lr * args.backbone_lr_multiplier,
        args.lr * args.backbone_lr_multiplier * args.expert_lr_multiplier,
        args.momentum,
        args.musgd_muon_weight,
        args.musgd_sgd_weight,
        global_batch_size,
        accumulation_steps,
        effective_batch_size,
        effective_weight_decay,
    )
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay else None
    if ema is not None:
        LOGGER.info("EMA enabled: decay=%.6f", args.ema_decay)
    full_loader_steps = len(loader)
    epoch_batch_schedule = detector_step_schedule(
        full_loader_steps,
        total_epochs=args.epochs,
        mosaic_probability=args.mosaic,
        close_mosaic_epochs=args.close_mosaic_epochs,
        mosaic_packed_epoch=mosaic_packed_epoch,
        packed_epochs=args.packed_epochs,
    )
    epoch_step_schedule = optimizer_step_schedule(epoch_batch_schedule, accumulation_steps)
    total_steps = sum(epoch_step_schedule)
    unpacked_total_steps = None
    if mosaic_packed_epoch:
        unpacked_epoch_batch_schedule = detector_step_schedule(
            full_loader_steps,
            total_epochs=args.epochs,
            mosaic_probability=args.mosaic,
            close_mosaic_epochs=args.close_mosaic_epochs,
            mosaic_packed_epoch=False,
        )
        unpacked_total_steps = sum(
            optimizer_step_schedule(unpacked_epoch_batch_schedule, accumulation_steps)
        )
    effective_warmup_steps = resolve_warmup_steps(
        requested_steps=args.warmup_steps,
        warmup_epochs=args.warmup_epochs,
        steps_in_first_epoch=epoch_step_schedule[0],
        total_steps=total_steps,
        unpacked_total_steps=unpacked_total_steps,
    )
    if mosaic_packed_epoch:
        tile_exposure = mosaic_source_images_per_sample(args.mosaic, args.mosaic_tiles)
        LOGGER.info(
            "Mosaic packed epochs: %d/%d batches while active "
            "(step-count divisor packed_epochs=%d, independent of mosaic_tiles=%d "
            "[~%.2f source images/canvas], canvas=%d -> crop=%d); "
            "%d planned optimizer steps; "
            "warmup %s -> %d optimizer steps",
            epoch_batch_schedule[0],
            full_loader_steps,
            args.packed_epochs,
            args.mosaic_tiles,
            tile_exposure,
            mosaic_canvas_size,
            args.image_size,
            total_steps,
            (
                f"{args.warmup_epochs:g} epochs"
                if args.warmup_epochs is not None
                else f"{args.warmup_steps} steps"
            ),
            effective_warmup_steps,
        )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_schedule(
            step,
            warmup_steps=effective_warmup_steps,
            total_steps=total_steps,
            min_ratio=args.min_lr_ratio,
        ),
    )
    compiled_model: torch.nn.Module = model
    if args.compile_mode != "none":
        if device.type != "cuda":
            raise ValueError("--compile-mode requires CUDA")
        compiled_model = torch.compile(model, mode=args.compile_mode)
        LOGGER.info("TorchInductor enabled: mode=%s", args.compile_mode)
    training_model = distributed.wrap(compiled_model)

    metrics_path = args.output / "metrics.jsonl"
    tensorboard = TensorBoardMetricWriter(
        args.output / "tensorboard",
        enabled=distributed.is_main,
    )
    if tensorboard.enabled:
        LOGGER.info("TensorBoard metrics: %s", tensorboard.log_dir)

    training_options: Dict[str, object] = {
        "optimizer": args.optimizer,
        "batch_size": args.batch_size,
        "global_batch_size": global_batch_size,
        "nominal_batch_size": args.nominal_batch_size,
        "accumulation_steps": accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "dataset_size": len(dataset),
        "dataset_source": args.hf_detection_dataset or "local",
        "dataset_global_size": getattr(dataset, "total_examples", len(dataset)),
        "seed": args.seed,
        "lr": args.lr,
        "backbone_lr_multiplier": args.backbone_lr_multiplier,
        "expert_lr_multiplier": args.expert_lr_multiplier,
        "one_to_one_lr_multiplier": args.one_to_one_lr_multiplier,
        "momentum": args.momentum,
        "musgd_muon_weight": args.musgd_muon_weight,
        "musgd_sgd_weight": args.musgd_sgd_weight,
        "weight_decay": args.weight_decay,
        "effective_weight_decay": effective_weight_decay,
        "warmup_steps": args.warmup_steps,
        "warmup_epochs": args.warmup_epochs,
        "effective_warmup_steps": effective_warmup_steps,
        "min_lr_ratio": args.min_lr_ratio,
        "augmentation": args.augmentation,
        "augmentation_backend": args.augmentation_backend,
        "image_backend": args.image_backend,
        "eval_backend": resolved_eval_backend,
        "eval_max_detections": args.eval_max_detections,
        "device_type": device.type,
        "use_amp": use_amp,
        "compile_mode": args.compile_mode,
        "provenance": provenance,
    }
    training_options.update(
        {
            "ema_decay": args.ema_decay,
            "mosaic": args.mosaic,
            "mosaic_tiles": args.mosaic_tiles,
            "mosaic_canvas_size": mosaic_canvas_size,
            "mosaic_packed_epoch": mosaic_packed_epoch,
            "packed_epochs": args.packed_epochs,
            "planned_optimizer_steps": total_steps,
            "mixup": args.mixup,
            "copy_paste": args.copy_paste,
            "random_erasing": args.random_erasing,
            "close_mosaic_epochs": args.close_mosaic_epochs,
            "multi_scale_min": args.multi_scale_min,
            "multi_scale_max": multi_scale_max,
            "multi_scale_step": multi_scale_step,
            "video_clip_frames": args.video_clip_frames,
            "video_frame_stride": args.video_frame_stride,
        }
    )
    if distributed.enabled:
        training_options["world_size"] = distributed.world_size
    start_epoch = 0
    start_batch = 0
    step = 0
    best_map50 = -1.0
    best_end_to_end_map50 = -1.0
    running_losses: Dict[str, float] = {}
    running_loss_steps = 0
    if args.resume is not None:
        resume_state = load_training_state(
            args.resume,
            optimizer,
            scheduler,
            total_epochs=args.epochs,
            steps_per_epoch=len(loader),
            training_options=training_options,
            rank=distributed.rank,
            world_size=distributed.world_size,
            device=device,
        )
        start_epoch = int(resume_state["epoch"])
        start_batch = int(resume_state["batch_in_epoch"])
        step = int(resume_state["step"])
        best_map50 = float(resume_state["best_map50"])
        best_end_to_end_map50 = float(resume_state.get("best_end_to_end_map50", -1.0))
        running_losses = {
            str(name): float(value)
            for name, value in resume_state.get("running_losses", {}).items()
        }
        running_loss_steps = int(resume_state.get("running_loss_steps", 0))
        if not 0 <= start_epoch <= args.epochs:
            raise ValueError(f"invalid resumed epoch cursor: {start_epoch}")
        resumed_epoch_batches = (
            epoch_batch_schedule[start_epoch] if start_epoch < args.epochs else 0
        )
        if not 0 <= start_batch <= resumed_epoch_batches:
            raise ValueError(f"invalid resumed batch cursor: {start_batch}")
        if (
            start_batch
            and start_batch != resumed_epoch_batches
            and start_batch % accumulation_steps
        ):
            raise ValueError("resumed batch cursor is inside a gradient-accumulation group")
        if start_epoch == args.epochs and start_batch:
            raise ValueError("a completed run cannot have a non-zero batch cursor")
        LOGGER.info(
            "Resumed exactly from %s: epoch=%d batch=%d step=%d",
            args.resume,
            start_epoch,
            start_batch,
            step,
        )
        if ema is not None:
            ema_path = args.resume / "ema.safetensors"
            if not ema_path.is_file():
                raise ValueError("EMA-enabled resume requires ema.safetensors")
            ema.load_state_dict(load_file(str(ema_path)), updates=step)

    def write_checkpoint(
        *,
        epoch: int,
        batch_in_epoch: int,
        name: str | None = None,
        validation_metrics: Dict[str, Any] | None = None,
    ) -> None:
        distributed_rng_states = distributed.gather_rng_states()
        if distributed.is_main:
            save_checkpoint(
                args.output,
                model,
                config,
                step,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                batch_in_epoch=batch_in_epoch,
                best_map50=best_map50,
                best_end_to_end_map50=best_end_to_end_map50,
                running_losses=running_losses,
                running_loss_steps=running_loss_steps,
                total_epochs=args.epochs,
                steps_per_epoch=len(loader),
                training_options=training_options,
                provenance=provenance,
                name=name,
                validation_metrics=validation_metrics,
                distributed_rng_states=distributed_rng_states,
                ema_model=ema.module if ema is not None else None,
            )
        distributed.barrier()

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs):
        epoch_batches = epoch_batch_schedule[epoch]
        if epoch_dataset is not None and hasattr(epoch_dataset, "set_epoch"):
            epoch_dataset.set_epoch(epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        else:
            shuffle_generator.manual_seed(args.seed + epoch)
        loader_iterator = iter(loader)
        batches_to_skip = start_batch if epoch == start_epoch else 0
        for _ in range(batches_to_skip):
            next(loader_iterator)
        loader_iterator = islice(loader_iterator, epoch_batches - batches_to_skip)
        progress = tqdm(
            loader_iterator,
            desc=f"detector train {epoch + 1}/{args.epochs}",
            unit="batch",
            total=epoch_batches,
            initial=batches_to_skip,
            dynamic_ncols=True,
            leave=False,
            disable=not distributed.is_main,
        )
        processed_batches = batches_to_skip
        for batch_index, (pixel_values, targets) in enumerate(progress, start=batches_to_skip):
            processed_batches = batch_index + 1
            group_size, optimizer_boundary = accumulation_group(
                batch_index,
                epoch_batches=epoch_batches,
                accumulation_steps=accumulation_steps,
            )
            pixel_values = pixel_values.to(device, non_blocking=device.type == "cuda")
            if args.multi_scale_min:
                choices = range(
                    args.multi_scale_min,
                    multi_scale_max + 1,
                    multi_scale_step,
                )
                resize_rng = random.Random(args.seed + epoch * len(loader) + batch_index)
                runtime_size = resize_rng.choice(tuple(choices))
                if tuple(pixel_values.shape[-2:]) != (runtime_size, runtime_size):
                    pixel_values = resize_detector_inputs(
                        pixel_values,
                        (runtime_size, runtime_size),
                    )
            targets = [target.to(device, non_blocking=device.type == "cuda") for target in targets]

            autocast = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
            sync_context = (
                training_model.no_sync()
                if distributed.enabled and not optimizer_boundary
                else nullcontext()
            )
            with sync_context:
                with autocast:
                    raw = training_model(
                        pixel_values,
                        return_branches=config.end_to_end,
                        return_auxiliary=bool(
                            config.gate_calibration_loss_weight
                            or config.object_contrastive_loss_weight
                        ),
                    )
                    losses = model.compute_loss(
                        raw,
                        targets,
                        training_progress=step / max(total_steps - 1, 1),
                    )
                (losses["loss"] / group_size).backward()
            for name, value in losses.items():
                running_losses[name] = running_losses.get(name, 0.0) + float(value.detach())
            running_loss_steps += 1
            if not optimizer_boundary:
                continue

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                foreach=False if device.type == "mps" else None,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)

            step += 1
            if step % args.log_steps == 0:
                averages = {
                    name: value / running_loss_steps for name, value in running_losses.items()
                }
                averages = distributed.mean_scalars(averages)
                if distributed.is_main:
                    logged_averages = format_loss_metrics_for_logging(averages)
                    current_lrs = named_learning_rates(optimizer)
                    task_lr = current_lrs.get("task", current_lrs.get("task_experts", 0.0))
                    backbone_lr = current_lrs.get(
                        "backbone", current_lrs.get("backbone_experts", 0.0)
                    )
                    expert_lr = current_lrs.get("backbone_experts", backbone_lr)
                    with metrics_path.open("a") as handle:
                        handle.write(
                            json.dumps(
                                {
                                    "step": step,
                                    "epoch": epoch,
                                    "lr": task_lr,
                                    "backbone_lr": backbone_lr,
                                    "expert_lr": expert_lr,
                                    **logged_averages,
                                }
                            )
                            + "\n"
                        )
                    tensorboard.add_scalars(
                        "train",
                        {
                            **logged_averages,
                            "lr": task_lr,
                            "backbone_lr": backbone_lr,
                            "expert_lr": expert_lr,
                            "epoch": epoch,
                        },
                        step,
                    )
                    progress.set_postfix(
                        loss=f"{logged_averages['loss']:.4f}",
                        opt=f"{logged_averages['optimization_loss']:.4f}",
                        lr=f"{task_lr:.2e}",
                        backbone_lr=f"{backbone_lr:.2e}",
                        expert_lr=f"{expert_lr:.2e}",
                    )
                running_losses.clear()
                running_loss_steps = 0
            if args.save_steps and step % args.save_steps == 0:
                write_checkpoint(epoch=epoch, batch_in_epoch=batch_index + 1)

        verify_loader_cardinality(
            loader,
            processed_batches,
            distributed,
            phase=f"training epoch {epoch + 1}",
            expected_batches=epoch_batches,
        )

        should_validate = validation_loader is not None and should_validate_epoch(
            epoch,
            args.epochs,
            args.eval_every,
        )
        if should_validate:
            validation_metrics = evaluate_detector(
                ema.module if ema is not None else model,
                validation_loader,
                device,
                confidence_threshold=args.eval_confidence,
                use_amp=use_amp,
                show_progress=distributed.is_main,
                max_detections=args.eval_max_detections,
                distributed=distributed,
                coco_annotations_path=(
                    coco_validation.annotations_path if official_coco_evaluation else None
                ),
                coco_category_ids=(
                    coco_validation.class_to_category if official_coco_evaluation else None
                ),
                coco_backend=args.eval_backend,
            )
            end_to_end_metrics = (
                evaluate_detector(
                    ema.module if ema is not None else model,
                    validation_loader,
                    device,
                    confidence_threshold=args.eval_confidence,
                    use_amp=use_amp,
                    show_progress=distributed.is_main,
                    max_detections=args.eval_max_detections,
                    distributed=distributed,
                    end_to_end=True,
                    coco_annotations_path=(
                        coco_validation.annotations_path if official_coco_evaluation else None
                    ),
                    coco_category_ids=(
                        coco_validation.class_to_category if official_coco_evaluation else None
                    ),
                    coco_backend=args.eval_backend,
                )
                if config.end_to_end
                else None
            )
            LOGGER.info(
                "validation epoch=%d mAP50=%.4f mAP50-95=%.4f "
                "APs=%.4f APm=%.4f APl=%.4f precision=%.4f recall=%.4f "
                "f1=%.4f best_f1=%.4f best_conf=%.3f",
                epoch,
                validation_metrics["map50"],
                validation_metrics["map50_95"],
                validation_metrics["ap_small"],
                validation_metrics["ap_medium"],
                validation_metrics["ap_large"],
                validation_metrics["precision"],
                validation_metrics["recall"],
                validation_metrics["f1"],
                validation_metrics["best_f1"],
                validation_metrics["best_confidence"],
            )
            if validation_metrics.get("official_coco"):
                LOGGER.info(
                    "official COCO evaluator=%s AR100=%.4f",
                    validation_metrics["evaluator_backend"],
                    validation_metrics["ar_100"],
                )
            if end_to_end_metrics is not None:
                LOGGER.info(
                    "validation NMS-free epoch=%d mAP50=%.4f mAP50-95=%.4f "
                    "APs=%.4f APm=%.4f APl=%.4f precision=%.4f recall=%.4f "
                    "f1=%.4f best_f1=%.4f best_conf=%.3f",
                    epoch,
                    end_to_end_metrics["map50"],
                    end_to_end_metrics["map50_95"],
                    end_to_end_metrics["ap_small"],
                    end_to_end_metrics["ap_medium"],
                    end_to_end_metrics["ap_large"],
                    end_to_end_metrics["precision"],
                    end_to_end_metrics["recall"],
                    end_to_end_metrics["f1"],
                    end_to_end_metrics["best_f1"],
                    end_to_end_metrics["best_confidence"],
                )
            selection_metric, selection_score = validation_selection_score(validation_metrics)
            validation_metrics["checkpoint_selection_metric"] = selection_metric
            validation_metrics["checkpoint_selection_score"] = selection_score
            if end_to_end_metrics is not None:
                end_to_end_selection_metric, end_to_end_selection_score = (
                    validation_selection_score(end_to_end_metrics)
                )
                end_to_end_metrics["checkpoint_selection_metric"] = end_to_end_selection_metric
                end_to_end_metrics["checkpoint_selection_score"] = end_to_end_selection_score
            if distributed.is_main:
                with metrics_path.open("a") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "step": step,
                                "epoch": epoch,
                                "validation": validation_metrics,
                                **(
                                    {"validation_nms_free": end_to_end_metrics}
                                    if end_to_end_metrics is not None
                                    else {}
                                ),
                            }
                        )
                        + "\n"
                    )
                tensorboard.add_scalars("validation/o2m", validation_metrics, step)
                if end_to_end_metrics is not None:
                    tensorboard.add_scalars(
                        "validation/nms_free",
                        end_to_end_metrics,
                        step,
                    )
                tensorboard.flush()
            if selection_score > best_map50:
                best_map50 = selection_score
                write_checkpoint(
                    epoch=epoch + 1,
                    batch_in_epoch=0,
                    name="best",
                    validation_metrics=validation_metrics,
                )
            if (
                end_to_end_metrics is not None
                and end_to_end_selection_score > best_end_to_end_map50
            ):
                best_end_to_end_map50 = end_to_end_selection_score
                write_checkpoint(
                    epoch=epoch + 1,
                    batch_in_epoch=0,
                    name="best_nms_free",
                    validation_metrics=end_to_end_metrics,
                )
        start_batch = 0

    write_checkpoint(epoch=args.epochs, batch_in_epoch=0)
    LOGGER.info("Training complete: %d steps over %d epochs", step, args.epochs)
    tensorboard.close()
    distributed.close()


if __name__ == "__main__":
    main()
