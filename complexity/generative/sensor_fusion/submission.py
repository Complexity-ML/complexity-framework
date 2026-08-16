"""Generate and validate a CUHK-X Small Model Track submission CSV."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..detection.distributed import DistributedContext
from .config import TRHashSensorFusionConfig
from .data import CUHKXSmallTrackTestDataset, collate_cuhkx_test
from .model import TRHashSensorFusionClassifier
from .preprocessing import CUHKXPreprocessingConfig
from .quality_gate import validate_cross_subject_quality

LOGGER = logging.getLogger("tr_hash_sensor_fusion_submission")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--validation-checkpoints",
        type=Path,
        nargs="+",
        required=True,
        help="Compatible cross-subject fold checkpoints required by the quality gate",
    )
    parser.add_argument("--minimum-folds", type=int, default=3)
    parser.add_argument("--minimum-fold-top1", type=float, default=0.86)
    parser.add_argument("--logits-output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8, help="Per process/GPU")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--require-fused-cuda", action="store_true")
    return parser.parse_args()


def resolve_device(override: str | None) -> torch.device:
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _autocast(device: torch.device, precision: str):
    if precision == "bf16" and device.type in {"cuda", "cpu"}:
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def read_test_paths(test_csv: str | Path) -> list[str]:
    with Path(test_csv).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["path", "prediction"]:
            raise ValueError("CUHK-X test.csv must have exactly path,prediction columns")
        rows = list(reader)
    if any(row.get("prediction", "").strip() for row in rows):
        raise ValueError("test.csv contains predictions; refusing possible label leakage")
    paths = [row["path"] for row in rows]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("CUHK-X test paths must be non-empty and unique")
    return paths


def write_submission(
    output: str | Path,
    paths: Sequence[str],
    predictions: Sequence[int],
    *,
    num_classes: int = 40,
) -> None:
    if len(paths) != len(predictions):
        raise ValueError("submission paths and predictions have different lengths")
    if len(paths) != len(set(paths)):
        raise ValueError("submission paths contain duplicates")
    predictions = [int(value) for value in predictions]
    invalid = [value for value in predictions if not 0 <= value < num_classes]
    if invalid:
        raise ValueError(f"submission contains invalid class id: {invalid[0]}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("path", "prediction"))
        writer.writerows(zip(paths, predictions))
    temporary.replace(output)


def validate_submission(
    submission: str | Path,
    test_csv: str | Path,
    *,
    num_classes: int = 40,
) -> dict[str, Any]:
    expected = read_test_paths(test_csv)
    with Path(submission).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["path", "prediction"]:
            raise ValueError("submission must have exactly path,prediction columns")
        rows = list(reader)
    paths = [row["path"] for row in rows]
    if paths != expected:
        raise ValueError("submission paths or ordering differ from official test.csv")
    try:
        predictions = [int(row["prediction"]) for row in rows]
    except (TypeError, ValueError) as error:
        raise ValueError("every submission prediction must be an integer") from error
    if any(not 0 <= value < num_classes for value in predictions):
        raise ValueError(f"submission predictions must be in [0, {num_classes - 1}]")
    counts = np.bincount(predictions, minlength=num_classes)
    return {
        "rows": len(rows),
        "num_classes": num_classes,
        "predicted_classes": int(np.count_nonzero(counts)),
        "min_class_count": int(counts.min()),
        "max_class_count": int(counts.max()),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    context = DistributedContext.initialize(resolve_device(args.device))
    device = context.device
    try:
        quality = validate_cross_subject_quality(
            args.checkpoint,
            args.validation_checkpoints,
            minimum_folds=args.minimum_folds,
            minimum_fold_top1=args.minimum_fold_top1,
        )
        if context.is_main:
            LOGGER.info("Cross-subject quality gate: %s", quality)
        metadata = json.loads((args.checkpoint / "config.json").read_text())
        model_config = TRHashSensorFusionConfig.from_dict(metadata["model"])
        preprocessing = CUHKXPreprocessingConfig.from_dict(metadata["preprocessing"])
        model = TRHashSensorFusionClassifier(model_config).to(device).eval()
        model.load_state_dict(
            load_file(str(args.checkpoint / "model.safetensors"), device="cpu")
        )
        dataset = CUHKXSmallTrackTestDataset(
            args.data_root,
            args.test_csv,
            preprocessing=preprocessing,
        )
        loader_options = {
            "batch_size": args.batch_size,
            "sampler": context.eval_sampler(dataset),
            "shuffle": False,
            "drop_last": False,
            "num_workers": args.workers,
            "pin_memory": device.type == "cuda",
            "collate_fn": collate_cuhkx_test,
        }
        if args.workers > 0:
            loader_options["persistent_workers"] = True
            loader_options["prefetch_factor"] = 2
        loader = DataLoader(dataset, **loader_options)
        if context.is_main:
            LOGGER.info(
                "CUHK-X test: %d clips, checkpoint=%s, device=%s, world=%d",
                len(dataset),
                args.checkpoint,
                device,
                context.world_size,
            )

        local_results = []
        backend_checked = False
        progress = tqdm(
            loader,
            desc="CUHK-X submission",
            unit="batch",
            disable=not context.is_main,
            dynamic_ncols=True,
        )
        for batch in progress:
            inputs = _move(batch["inputs"], device)
            masks = _move(batch["modality_mask"], device)
            with _autocast(device, model_config.precision):
                logits = model(inputs, modality_mask=masks)["logits"]
            if not backend_checked:
                backends = {
                    block.mlp.last_backend.selected.value
                    for block in model.blocks
                    if block.mlp.last_backend is not None
                }
                if args.require_fused_cuda and backends != {"fused_cuda"}:
                    raise RuntimeError(
                        f"expected fused_cuda for every TR-HASH block, got {sorted(backends)}"
                    )
                if context.is_main:
                    LOGGER.info("TR-HASH backends: %s", ", ".join(sorted(backends)))
                backend_checked = True
            for index, path, values in zip(
                batch["indices"].tolist(),
                batch["paths"],
                logits.float().cpu().tolist(),
            ):
                local_results.append((int(index), path, values))
        gathered = context.all_gather_objects(local_results)
        if context.is_main:
            results = sorted(
                (item for rank_results in gathered for item in rank_results),
                key=lambda item: item[0],
            )
            expected_indices = list(range(len(dataset)))
            if [item[0] for item in results] != expected_indices:
                raise RuntimeError("distributed inference produced missing or duplicate test clips")
            paths = [item[1] for item in results]
            official_paths = read_test_paths(args.test_csv)
            if paths != official_paths:
                raise RuntimeError("inference result ordering differs from official test.csv")
            logits = np.asarray([item[2] for item in results], dtype=np.float32)
            predictions = logits.argmax(axis=1).tolist()
            write_submission(
                args.output,
                paths,
                predictions,
                num_classes=model_config.num_classes,
            )
            report = validate_submission(
                args.output,
                args.test_csv,
                num_classes=model_config.num_classes,
            )
            logits_output = args.logits_output or args.output.with_suffix(".logits.npz")
            logits_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                logits_output,
                paths=np.asarray(paths),
                logits=logits,
                checkpoint=np.asarray(str(args.checkpoint)),
            )
            LOGGER.info("Submission saved: %s", args.output)
            LOGGER.info("Logits saved: %s", logits_output)
            LOGGER.info("Validation: %s", report)
        context.barrier()
    finally:
        context.close()


if __name__ == "__main__":
    main()
