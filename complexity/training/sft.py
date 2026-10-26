"""Local full-parameter SFT, artifact preflight and bounded CPU diagnostics.

This prepares the explicit ``no_refinement_ablation`` experiment. It does not
change the release stage policy, download artifacts, or select a device automatically.
The reader retains ec721a2's next-fit packing and already shifted SFT labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import time
from contextlib import nullcontext, redirect_stdout
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from complexity.config import ModelConfig
from complexity.core.losses import causal_lm_loss_from_hidden
from complexity.inference.chat_template import validate_chat_template
from complexity.models import ComplexityModel

from .adapter_optimizer import build_finetuning_optimizer
from .finetuning import PRETRAINING_STAGE, SUPERVISED_FINETUNING_STAGE
from .sequence_packing import pack_example_lengths, resolve_epoch_schedule
from .sft_shard import FINAL_ASSISTANT_SUPERVISION, SHARD_FORMAT_V2, validate_shard_supervision

EXPERIMENT = "no_refinement_ablation"


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class SFTTokenShardDataset(Dataset):
    """Read complete final-assistant examples; preserve every supervised EOS.

    Inputs are little-endian uint32; labels are little-endian int32 and already
    predict the next token. There is no runtime relabeling or truncation.
    Packing follows source order, inserts one ignored EOS between examples,
    and pads the remaining window with EOS inputs / ignored labels.
    """

    packing_contract = "supervised-finetuning"

    def __init__(self, root: str | Path, *, sequence_length: int, vocab_size: int):
        root = Path(root)
        self.root = root / "train" if (root / "train" / "sft.idx.json").is_file() else root
        dataset_root = self.root.parent
        if isinstance(sequence_length, bool) or sequence_length < 1 or vocab_size < 1:
            raise ValueError("sequence_length and vocab_size must be positive")
        self.sequence_length, self.vocab_size = sequence_length, vocab_size
        self.metadata = _read_json(self.root / "sft.idx.json")
        if self.metadata.get("format") != SHARD_FORMAT_V2:
            raise ValueError("This SFT workflow requires token-shard v2")
        validate_shard_supervision(self.metadata)
        if self.metadata.get("assistant_supervision") != FINAL_ASSISTANT_SUPERVISION:
            raise ValueError("This experiment requires final_assistant_only labels")
        self.chat_template = validate_chat_template(_read_json(dataset_root / "chat_template.json"))
        if self.metadata.get("chat_template_id") != self.chat_template["id"]:
            raise ValueError("Shard and chat template IDs differ")
        self.eos_token_id = self.metadata["eos_token_id"]
        if isinstance(self.eos_token_id, bool) or not 0 <= self.eos_token_id < vocab_size:
            raise ValueError("Shard EOS token ID is outside the model vocabulary")
        if self.metadata.get("chat_template_eos_token") != self.chat_template["eos_token"]:
            raise ValueError("Shard and chat template EOS spellings differ")
        if (
            self.metadata.get("input_dtype") != "uint32-le"
            or self.metadata.get("label_dtype") != "int32-le"
        ):
            raise ValueError("Shard must declare uint32-le inputs and int32-le labels")
        if self.metadata.get("truncation_policy") != "fail_closed_no_truncation":
            raise ValueError("Shard must declare the no-truncation contract")
        for filename in ("input_ids.bin", "labels.bin", "examples.jsonl"):
            expected = self.metadata.get("files", {}).get(filename)
            if not isinstance(expected, str) or _digest(self.root / filename) != expected:
                raise ValueError(f"Shard integrity failed: {filename}")
        manifest_path = dataset_root / "manifest.json"
        if manifest_path.is_file():
            manifest = _read_json(manifest_path)
            partition = manifest.get("partitions", {}).get(self.root.name)
            if partition != self.metadata:
                raise ValueError("Dataset manifest partition and shard index differ")
            tokenizer_path = dataset_root / "tokenizer" / "tokenizer.json"
            if _digest(tokenizer_path) != manifest.get("tokenizer_sha256"):
                raise ValueError("Dataset tokenizer hash differs from its manifest")
        declared_vocab = self.metadata.get("vocab_size")
        if declared_vocab is not None and declared_vocab != vocab_size:
            raise ValueError("Shard tokenizer vocabulary differs from the model")
        for filename in ("input_ids.bin", "labels.bin"):
            path = self.root / filename
            if path.stat().st_size == 0 or path.stat().st_size % 4:
                raise ValueError(f"Invalid 32-bit token array: {path}")
        self.input_ids = np.memmap(self.root / "input_ids.bin", mode="r", dtype="<u4")
        self.labels = np.memmap(self.root / "labels.bin", mode="r", dtype="<i4")
        if (
            len(self.input_ids) != len(self.labels)
            or len(self.input_ids) != self.metadata["num_tokens"]
        ):
            raise ValueError("Token arrays and manifest counts differ")
        self.examples = [
            json.loads(line)
            for line in (self.root / "examples.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.examples or len(self.examples) != self.metadata["examples"]:
            raise ValueError("Example index and manifest counts differ")
        cursor, supervised = 0, 0
        lengths = []
        for example in self.examples:
            start, length = example["offset"], example["num_tokens"]
            if start != cursor or not 1 <= length <= sequence_length:
                raise ValueError("Examples must cover the shard once, in order, without truncation")
            end = start + length
            if end > len(self.input_ids):
                raise ValueError("Example exceeds its token array")
            inputs, labels = self.input_ids[start:end], self.labels[start:end]
            active = np.flatnonzero(labels != -100)
            if (
                not active.size
                or active[-1] != length - 1
                or np.any(np.diff(active) != 1)
                or labels[-1] != self.eos_token_id
            ):
                raise ValueError(
                    "Each example must supervise one complete final assistant suffix with EOS"
                )
            if (
                inputs.max() >= vocab_size
                or np.any(labels[active] < 0)
                or labels[active].max() >= vocab_size
            ):
                raise ValueError("Input or label token is outside the model vocabulary")
            visible = labels[:-1] != -100
            if np.any(labels[:-1][visible] != inputs[1:][visible]):
                raise ValueError("Shard labels must already be aligned next-token targets")
            declared = example.get("supervised_tokens", int(active.size))
            if declared != active.size:
                raise ValueError("Example supervised-token count differs from labels")
            lengths.append(length)
            cursor, supervised = end, supervised + int(active.size)
        if cursor != len(self.input_ids):
            raise ValueError("Example index leaves unreferenced tokens")
        if supervised != self.metadata.get("supervised_tokens"):
            raise ValueError("Manifest supervised-token count differs from labels")
        self.supervised_tokens = supervised
        self.packing_plan = pack_example_lengths(
            lengths, sequence_length=sequence_length, separator_tokens=1
        )

    def __len__(self) -> int:
        return self.packing_plan.packed_items

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        input_parts, label_parts = [], []
        for position, index in enumerate(self.packing_plan.packs[item]):
            if position:
                input_parts.append(np.array([self.eos_token_id], dtype=np.int64))
                label_parts.append(np.array([-100], dtype=np.int64))
            example = self.examples[index]
            start, end = example["offset"], example["offset"] + example["num_tokens"]
            input_parts.append(np.array(self.input_ids[start:end], dtype=np.int64))
            label_parts.append(np.array(self.labels[start:end], dtype=np.int64))
        inputs, labels = np.concatenate(input_parts), np.concatenate(label_parts)
        padding = self.sequence_length - len(inputs)
        return {
            "input_ids": torch.from_numpy(
                np.pad(inputs, (0, padding), constant_values=self.eos_token_id)
            ),
            "labels": torch.from_numpy(np.pad(labels, (0, padding), constant_values=-100)),
        }


def _normalized_model_config(checkpoint: Path) -> tuple[dict, ModelConfig]:
    raw = _read_json(checkpoint / "config.json")
    if raw.get("mlp_type") not in {"tr_hash_engine", "tr_hash_moe"} or raw.get(
        "routing_strategy"
    ) != "token_id_composed_modulo":
        raise ValueError("SFT checkpoints must declare the composed modulo architecture")
    patched = dict(raw)
    if "num_experts_per_tok" in patched:
        if "top_k" in patched and patched["top_k"] != patched["num_experts_per_tok"]:
            raise ValueError("Checkpoint top_k and num_experts_per_tok disagree")
        patched["top_k"] = patched["num_experts_per_tok"]
    if "norm_eps" not in patched and "rms_norm_eps" in patched:
        patched["norm_eps"] = patched["rms_norm_eps"]
    known = {field.name for field in fields(ModelConfig)}
    config = ModelConfig.from_dict(
        {key: value for key, value in patched.items() if key in known}, strict=True
    )
    return raw, config


def load_sft_model(checkpoint: str | Path) -> ComplexityModel:
    """Load canonical SafeTensors on CPU with validated composed routing anchors."""
    checkpoint = Path(checkpoint)
    raw, config = _normalized_model_config(checkpoint)
    config.use_custom_kernels = False
    model = ComplexityModel(config)
    model._load_pretrained_weights(checkpoint / "model.safetensors")
    return model.to(device="cpu", dtype=torch.float32).requires_grad_(True)


def build_sft_plan(
    checkpoint: str | Path,
    data: str | Path,
    *,
    experiment: str,
    batch_size: int,
    world_size: int,
    expert_lr_scale: float,
    sequence_length: int = 2048,
    epochs: int = 3,
    seed: int = 42,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    warmup_ratio: float = 0.03,
    min_lr_ratio: float = 0.1,
) -> dict[str, Any]:
    """Inspect local artifacts without constructing a model or an optimizer."""
    from safetensors import safe_open

    if experiment != EXPERIMENT:
        raise ValueError(f"Direct pretraining -> SFT requires explicit experiment={EXPERIMENT!r}")
    for name, value in (
        ("batch_size", batch_size),
        ("world_size", world_size),
        ("sequence_length", sequence_length),
        ("epochs", epochs),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    for name, value in (("learning_rate", learning_rate), ("expert_lr_scale", expert_lr_scale)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if len(betas) != 2 or any(not math.isfinite(beta) or not 0 <= beta < 1 for beta in betas):
        raise ValueError("AdamW betas must be finite and in [0, 1)")
    if not 0 <= warmup_ratio < 1 or not 0 <= min_lr_ratio <= 1:
        raise ValueError("Invalid warmup/minimum LR ratio")
    checkpoint = Path(checkpoint)
    raw, config = _normalized_model_config(checkpoint)
    if sequence_length > config.max_position_embeddings:
        raise ValueError("SFT sequence length exceeds the checkpoint's context")
    weights = checkpoint / "model.safetensors"
    with safe_open(weights, framework="pt", device="cpu") as handle:
        shapes = {key: tuple(handle.get_slice(key).get_shape()) for key in handle.keys()}
        tensor_elements = sum(math.prod(shape) for shape in shapes.values())
    with torch.device("meta"):
        skeleton = ComplexityModel(config)
    expected = {name: tuple(value.shape) for name, value in skeleton.state_dict().items()}
    if shapes != expected:
        missing = sorted(expected.keys() - shapes.keys())
        unexpected = sorted(shapes.keys() - expected.keys())
        mismatched = sorted(
            name for name in shapes.keys() & expected.keys() if shapes[name] != expected[name]
        )
        raise ValueError(
            f"Checkpoint tensor layout mismatch: missing={missing}, unexpected={unexpected}, shape={mismatched}"
        )
    del skeleton
    dataset = SFTTokenShardDataset(
        data, sequence_length=sequence_length, vocab_size=config.vocab_size
    )
    schedule = resolve_epoch_schedule(
        items=len(dataset), world_size=world_size, batch_size_per_rank=batch_size, epochs=epochs
    )
    files = [checkpoint / "config.json", weights, dataset.root.parent / "chat_template.json"]
    files.extend(
        dataset.root / filename
        for filename in ("input_ids.bin", "labels.bin", "examples.jsonl", "sft.idx.json")
    )
    for directory in (dataset.root.parent, checkpoint):
        for relative in (
            "manifest.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ):
            candidate = directory / relative
            if candidate.is_file():
                files.append(candidate)
    data_tokenizer = dataset.root.parent / "tokenizer" / "tokenizer.json"
    base_tokenizer = checkpoint / "tokenizer.json"
    if (
        data_tokenizer.is_file()
        and base_tokenizer.is_file()
        and _digest(data_tokenizer) != _digest(base_tokenizer)
    ):
        raise ValueError("Base and SFT tokenizer encodings differ")
    return {
        "schema_version": 1,
        "experiment": experiment,
        "source_stage": PRETRAINING_STAGE,
        "target_stage": SUPERVISED_FINETUNING_STAGE,
        "skipped_stage": "refinement",
        "release_eligible": False,
        "mode": "full_parameter",
        "checkpoint": str(checkpoint.resolve()),
        "data": str(dataset.root.parent.resolve()),
        "model_config": config.to_dict(),
        "historical_mlp_type": raw.get("mlp_type"),
        "tensor_elements_in_checkpoint": tensor_elements,
        "sequence_length": sequence_length,
        "epochs": epochs,
        "batch_size": batch_size,
        "world_size": world_size,
        "effective_batch_size": batch_size * world_size,
        "eos_token_id": dataset.eos_token_id,
        "seed": seed,
        "examples": len(dataset.examples),
        "packed_items": len(dataset),
        "source_tokens_per_epoch": dataset.packing_plan.source_tokens,
        "supervised_tokens_per_epoch": dataset.supervised_tokens,
        "steps_per_epoch": schedule.steps_per_epoch,
        "optimizer_updates": schedule.total_steps,
        "learning_rate": learning_rate,
        "expert_lr_scale": expert_lr_scale,
        "weight_decay": weight_decay,
        "betas": list(betas),
        "warmup_updates": max(1, int(schedule.total_steps * warmup_ratio)),
        "min_lr_ratio": min_lr_ratio,
        "label_alignment": "already_next_token",
        "packing": "ec721a2_next_fit_one_ignored_eos",
        "files": {
            str(path.resolve()): {"bytes": path.stat().st_size, "sha256": _digest(path)}
            for path in files
        },
        "training_started": False,
        "launch_supported": True,
        "supported_devices": ["cpu", "mps", "cuda"],
        "limitations": [
            "CPU tests qualify continuation only; MPS/CUDA hardware runs are not qualified",
            "Portable PyTorch CE differs from the historical BF16 CUDA/Liger runtime",
            "Published settings/provenance must be compared with the pinned release manifest",
        ],
    }


def sft_loss(model, batch: dict[str, torch.Tensor], *, chunk_tokens: int = 256) -> torch.Tensor:
    """Final-assistant causal CE using every pre-shifted target, including EOS."""
    base = model.module if hasattr(model, "module") else model
    device = next(base.parameters()).device
    labels = batch["labels"].to(device)
    if not torch.any(labels != -100):
        raise ValueError("SFT batch has no supervised labels")
    hidden = model(batch["input_ids"].to(device), return_logits=False)["last_hidden_state"]
    weight = base.embed_tokens.weight if base.lm_head is None else base.lm_head.weight
    loss, _ = causal_lm_loss_from_hidden(
        hidden,
        weight,
        labels,
        shift=False,
        label_smoothing=0.0,
        z_loss_coef=0.0,
        chunk_tokens=chunk_tokens,
        sync_metrics=False,
    )
    return loss


def _autocast(model, precision):
    device = next(model.parameters()).device
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if precision == "bf16" and device.type == "cuda"
        else nullcontext()
    )


def _optimizer_schedule(model, plan):
    optimizer = build_finetuning_optimizer(
        model,
        learning_rate=plan["learning_rate"],
        expert_lr_scale=plan["expert_lr_scale"],
        weight_decay=plan["weight_decay"],
        betas=tuple(plan["betas"]),
    )
    warmup, horizon, minimum = (
        plan["warmup_updates"],
        plan["optimizer_updates"],
        plan["min_lr_ratio"],
    )

    def factor(step):
        step += 1
        if step <= warmup:
            return step / warmup
        progress = min(1.0, max(0.0, (step - warmup) / max(1, horizon - warmup)))
        return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * progress))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def reference_epoch_batches(plan: dict, epoch: int):
    """Simulate reference DDP ranks locally, including their repeated padding."""
    indices = list(range(plan["packed_items"]))
    for previous_epoch in range(epoch + 1):
        random.Random(plan["seed"] + previous_epoch).shuffle(indices)
    ranks = []
    target = plan["steps_per_epoch"] * plan["batch_size"]
    for rank in range(plan["world_size"]):
        items = indices[rank :: plan["world_size"]]
        if not items:
            items = [indices[rank % len(indices)]]
        source = list(items)
        while len(items) < target:
            items.extend(source[: target - len(items)])
        ranks.append(items)
    for update in range(plan["steps_per_epoch"]):
        start = update * plan["batch_size"]
        yield [items[start : start + plan["batch_size"]] for items in ranks]


def _stack_items(data, indices):
    samples = [data[item] for item in indices]
    return {
        name: torch.stack([sample[name] for sample in samples]) for name in ("input_ids", "labels")
    }


def accumulated_sft_update(
    model,
    data,
    rank_batches,
    optimizer,
    scheduler,
    *,
    micro_batch_size: int,
    grad_clip: float = 1.0,
    precision: str = "fp32",
    chunk_tokens: int = 256,
):
    """Preserve DDP's equal-rank means under arbitrary local microbatching."""
    optimizer.zero_grad(set_to_none=True)
    model.train()
    rank_losses = []
    for rank_items in rank_batches:
        total_targets = sum(int((data[item]["labels"] != -100).sum()) for item in rank_items)
        rank_loss = 0.0
        for start in range(0, len(rank_items), micro_batch_size):
            batch = _stack_items(data, rank_items[start : start + micro_batch_size])
            targets = int((batch["labels"] != -100).sum())
            with _autocast(model, precision):
                loss = sft_loss(model, batch, chunk_tokens=chunk_tokens)
            if not torch.isfinite(loss):
                raise RuntimeError("SFT update produced a nonfinite loss")
            fraction = targets / total_targets
            (loss * fraction / len(rank_batches)).backward()
            rank_loss += float(loss.detach()) * fraction
        rank_losses.append(rank_loss)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "loss": sum(rank_losses) / len(rank_losses),
        "gradient_norm": float(norm),
        "lr": scheduler.get_last_lr()[0],
    }


@torch.no_grad()
def evaluate_sft(
    model, data, *, batch_size: int = 1, max_batches: int | None = None, precision: str = "fp32"
) -> dict:
    """Evaluate every held-out target, weighting batch means by target count."""
    model.eval()
    total, targets, examples, batches = 0.0, 0, 0, 0
    for batch in DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0):
        count = int((batch["labels"] != -100).sum())
        with _autocast(model, precision):
            loss = sft_loss(model, batch)
        if not torch.isfinite(loss):
            raise RuntimeError("SFT evaluation produced a nonfinite loss")
        total += float(loss) * count
        targets += count
        examples += len(batch["input_ids"])
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    return {
        "loss": total / targets,
        "supervised_tokens": targets,
        "packed_items": examples,
        "complete": examples == len(data),
        "aggregation": "supervised_token_weighted",
    }


def cpu_diagnostic(plan: dict) -> dict:
    """Execute one local forward/backward/update without writing checkpoints."""
    _verify_plan(plan)
    model = load_sft_model(plan["checkpoint"])
    data = SFTTokenShardDataset(
        plan["data"], sequence_length=plan["sequence_length"], vocab_size=model.config.vocab_size
    )
    optimizer, scheduler = _optimizer_schedule(model, plan)
    before = model.embed_tokens.weight.detach().clone()
    rank_batches = next(reference_epoch_batches(plan, 0))
    metrics = accumulated_sft_update(
        model, data, rank_batches, optimizer, scheduler, micro_batch_size=1
    )
    changed = bool(torch.any(before != model.embed_tokens.weight.detach()))
    if not changed:
        raise RuntimeError("CPU diagnostic did not update model weights")
    return {
        "experiment": EXPERIMENT,
        "release_eligible": False,
        "device": "cpu",
        "optimizer_updates": 1,
        **metrics,
        "parameters_updated": changed,
        "supervised_eos_targets": sum(
            int((data[index]["labels"] == data.eos_token_id).sum())
            for rank in rank_batches
            for index in rank
        ),
        "full_training_started": False,
        "checkpoint_written": False,
    }


def _verify_plan(plan):
    if plan.get("experiment") != EXPERIMENT or plan.get("release_eligible") is not False:
        raise ValueError("A marked non-release SFT experiment plan is required")
    for filename, entry in plan["files"].items():
        if _digest(Path(filename)) != entry["sha256"]:
            raise ValueError(f"Artifact changed after preflight: {filename}")


def _prepare_resume_metrics(output: Path, step: int) -> None:
    """Discard records newer than the committed cursor before replaying them."""
    path = output / "metrics.jsonl"
    if not path.is_file():
        return
    records = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line:
            continue
        record = json.loads(line)
        if type(record.get("step")) is not int:
            raise ValueError("Resume metrics contain an invalid optimizer-update cursor")
        if record["step"] <= step:
            records.append(line)
    pending = output / ".metrics.writing"
    pending.write_text("".join(line + "\n" for line in records), encoding="utf-8")
    pending.replace(path)


def _evaluation_identity(data, max_batches):
    paths = [
        data.root / name
        for name in ("sft.idx.json", "input_ids.bin", "labels.bin", "examples.jsonl")
    ]
    paths.append(data.root.parent / "chat_template.json")
    for relative in ("manifest.json", "tokenizer/tokenizer.json"):
        path = data.root.parent / relative
        if path.is_file():
            paths.append(path)
    return {
        "partition": str(data.root.resolve()),
        "max_batches": max_batches,
        "files": {str(path.resolve()): _digest(path) for path in paths},
    }


def _run_identity(
    plan,
    *,
    device,
    micro_batch_size,
    gradient_checkpointing,
    evaluation,
    eval_max_batches,
    precision,
):
    return {
        "plan": plan,
        "device": str(device),
        "precision": precision,
        "master_parameters": "fp32",
        "micro_batch_size": micro_batch_size,
        "gradient_checkpointing": gradient_checkpointing,
        "loss_aggregation": "equal_reference_rank_means",
        "evaluation_aggregation": "supervised_token_weighted",
        "evaluation": _evaluation_identity(evaluation, eval_max_batches),
    }


def _rng_state(device):
    numpy = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": (numpy[0], torch.from_numpy(numpy[1].copy()), numpy[2], numpy[3], numpy[4]),
        "mps": torch.mps.get_rng_state() if device.type == "mps" else None,
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def _restore_rng(state, device):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], numpy[1].numpy(), *numpy[2:]))
    if device.type == "mps":
        torch.mps.set_rng_state(state["mps"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def _save_run_checkpoint(
    output, *, model, optimizer, scheduler, step, identity, device, history, total_limit=3
):
    from complexity.utils.local_checkpoint import rotate_local_checkpoints

    def cpu_tree(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: cpu_tree(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cpu_tree(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cpu_tree(item) for item in value)
        return value

    state = {
        "model": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "optimizer": cpu_tree(optimizer.state_dict()),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "identity": identity,
        "rng": _rng_state(device),
        "history": history,
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"step_{step:06d}"
    pending = output / f".step_{step:06d}.writing"
    if pending.exists():
        shutil.rmtree(pending)
    pending.mkdir()
    torch.save(state, pending / "checkpoint.pt")
    marker = {
        "step": step,
        "experiment": identity["plan"].get("experiment", "staged_lora_sft"),
        "release_eligible": False,
        "checkpoint_sha256": _digest(pending / "checkpoint.pt"),
    }
    (pending / "complete.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    if path.exists():
        shutil.rmtree(path)
    pending.rename(path)
    latest = output / ".latest.writing"
    latest.write_text(path.name + "\n", encoding="utf-8")
    latest.replace(output / "latest")
    rotate_local_checkpoints(output, total_limit=total_limit)
    return path


def _export_epoch(model, data, output, epoch, step, identity):
    from complexity.export.assets import write_chat_template

    target = output / f"epoch_{epoch:02d}"
    pending = output / f".epoch_{epoch:02d}.writing"
    if pending.exists():
        shutil.rmtree(pending)
    with redirect_stdout(sys.stderr):
        model.save_pretrained(pending)
    write_chat_template({"chat_template": data.chat_template}, pending)
    source_tokenizer = data.root.parent / "tokenizer"
    if source_tokenizer.is_dir():
        from complexity.export.assets import copy_tokenizer_files

        copy_tokenizer_files(source_tokenizer, pending)
    (pending / "experiment.json").write_text(
        json.dumps(
            {
                "experiment": EXPERIMENT,
                "release_eligible": False,
                "step": step,
                "epoch": epoch,
                "run_identity_sha256": hashlib.sha256(
                    json.dumps(identity, sort_keys=True).encode()
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    if target.exists():

        def contents(directory):
            return {
                str(path.relative_to(directory)): _digest(path)
                for path in directory.rglob("*")
                if path.is_file()
            }

        matches = contents(target) == contents(pending)
        shutil.rmtree(pending)
        if not matches:
            raise ValueError(f"Epoch export differs from recovered update: {target}")
        return
    pending.rename(target)


def run_sft(
    plan: dict,
    output: str | Path,
    *,
    device: str,
    micro_batch_size: int = 1,
    steps: int | None = None,
    resume: str | Path | None = None,
    eval_data: str | Path | None = None,
    eval_max_batches: int | None = None,
    save_every: int = 100,
    gradient_checkpointing: bool = True,
    precision: str = "fp32",
) -> dict:
    """Explicit single-device experiment; a step limit preserves full LR horizon.

    Local accumulation simulates each reference rank's target-weighted CE and
    averages those rank means. Fresh initialization carries weights only.
    Resume restores AdamW, continuous cosine, exact update cursor and RNG.
    """
    _verify_plan(plan)
    device = torch.device(device)
    if device.type not in {"cpu", "mps", "cuda"}:
        raise ValueError("Local SFT requires explicit CPU, MPS or CUDA device")
    if precision not in {"fp32", "bf16"} or (precision == "bf16" and device.type != "cuda"):
        raise ValueError("BF16 autocast is supported only on CUDA; CPU/MPS require fp32")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing a silent device fallback")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This CUDA device does not support BF16")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; refusing a silent device fallback")
    if (
        micro_batch_size < 1
        or save_every < 1
        or (eval_max_batches is not None and eval_max_batches < 1)
    ):
        raise ValueError("Microbatch, save interval and evaluation limit must be positive")
    if steps is not None and not 1 <= steps <= plan["optimizer_updates"]:
        raise ValueError("steps must be within the complete experiment horizon")
    if (steps is None or steps == plan["optimizer_updates"]) and eval_max_batches is not None:
        raise ValueError("Full experiment requires complete held-out evaluation")
    output = Path(output)
    if resume is None and output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh output directory or explicitly resume this experiment")
    train = SFTTokenShardDataset(
        plan["data"],
        sequence_length=plan["sequence_length"],
        vocab_size=plan["model_config"]["vocab_size"],
    )
    evaluation = SFTTokenShardDataset(
        eval_data or Path(plan["data"]) / "eval",
        sequence_length=plan["sequence_length"],
        vocab_size=train.vocab_size,
    )
    identity = _run_identity(
        plan,
        device=device,
        micro_batch_size=micro_batch_size,
        gradient_checkpointing=gradient_checkpointing,
        evaluation=evaluation,
        eval_max_batches=eval_max_batches,
        precision=precision,
    )
    state = None
    if resume is not None:
        from complexity.utils.local_checkpoint import resolve_checkpoint_path

        checkpoint = resolve_checkpoint_path(resume)
        marker = _read_json(checkpoint / "complete.json")
        if marker.get("checkpoint_sha256") != _digest(checkpoint / "checkpoint.pt"):
            raise ValueError("Resume checkpoint integrity failed")
        state = torch.load(checkpoint / "checkpoint.pt", map_location="cpu", weights_only=True)
        if state["identity"] != identity:
            raise ValueError("Resume settings/artifact identity differ from the checkpoint")
        if (
            marker.get("step") != state["step"]
            or not 0 <= state["step"] <= plan["optimizer_updates"]
        ):
            raise ValueError("Resume checkpoint contains an invalid optimizer-update cursor")
        if steps is not None and steps < state["step"]:
            raise ValueError("Absolute step limit precedes the restored checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(plan["seed"])
    random.seed(plan["seed"])
    np.random.seed(plan["seed"])
    model = load_sft_model(plan["checkpoint"]).to(device)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    optimizer, scheduler = _optimizer_schedule(model, plan)
    step, history = 0, []
    if state is not None:
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step, history = state["step"], state["history"]
        _restore_rng(state["rng"], device)
        _prepare_resume_metrics(output, step)
    else:
        history.append(
            {
                "step": 0,
                "epoch": 0,
                "evaluation": evaluate_sft(
                    model,
                    evaluation,
                    batch_size=micro_batch_size,
                    max_batches=eval_max_batches,
                    precision=precision,
                ),
            }
        )
        _save_run_checkpoint(
            output,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=0,
            identity=identity,
            device=device,
            history=history,
        )
    (output / "run.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    horizon = plan["optimizer_updates"] if steps is None else steps
    started = time.monotonic()
    metrics_path = output / "metrics.jsonl"
    with metrics_path.open("a", encoding="utf-8") as metrics:
        try:
            while step < horizon:
                epoch, cursor = divmod(step, plan["steps_per_epoch"])
                for position, ranks in enumerate(reference_epoch_batches(plan, epoch)):
                    if position < cursor:
                        continue
                    update_started = time.monotonic()
                    values = accumulated_sft_update(
                        model,
                        train,
                        ranks,
                        optimizer,
                        scheduler,
                        micro_batch_size=micro_batch_size,
                        precision=precision,
                    )
                    if device.type == "mps":
                        torch.mps.synchronize()
                    step += 1
                    record = {
                        "step": step,
                        "epoch": epoch + 1,
                        **values,
                        "seconds": time.monotonic() - update_started,
                    }
                    metrics.write(json.dumps(record) + "\n")
                    metrics.flush()
                    if step % plan["steps_per_epoch"] == 0:
                        held_out = evaluate_sft(
                            model,
                            evaluation,
                            batch_size=micro_batch_size,
                            max_batches=eval_max_batches,
                            precision=precision,
                        )
                        history.append({"step": step, "epoch": epoch + 1, "evaluation": held_out})
                        _export_epoch(model, train, output, epoch + 1, step, identity)
                    if (
                        step % save_every == 0
                        or step >= horizon
                        or step % plan["steps_per_epoch"] == 0
                    ):
                        _save_run_checkpoint(
                            output,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            step=step,
                            identity=identity,
                            device=device,
                            history=history,
                        )
                    if step >= horizon:
                        break
        except KeyboardInterrupt:
            # Discard an unfinished microbatch update. Its data cursor is the
            # last committed optimizer update; the RNG is restored on resume.
            optimizer.zero_grad(set_to_none=True)
            raise
    result = {
        "experiment": EXPERIMENT,
        "release_eligible": False,
        "device": str(device),
        "precision": precision,
        "step": step,
        "planned_updates": plan["optimizer_updates"],
        "complete": step == plan["optimizer_updates"],
        "elapsed_seconds": time.monotonic() - started,
        "evaluations": history,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def load_sft_config(path: str | Path) -> tuple[dict, dict]:
    """Validate an authored experiment JSON; paths are relative to that file.

    Required keys are checkpoint, data, experiment, batch_size and world_size.
    Optional optimization keys match ``build_sft_plan``. Execution is an
    explicit separate object, so reading this configuration starts no run.
    """
    path = Path(path).resolve()
    config = _read_json(path)
    if config.pop("format", None) != "complexity-sft-experiment-v1":
        raise ValueError("SFT config requires format='complexity-sft-experiment-v1'")
    execution = config.pop("execution", {})
    if not isinstance(execution, dict):
        raise ValueError("SFT execution must be an object")
    allowed = {
        "checkpoint",
        "data",
        "experiment",
        "batch_size",
        "world_size",
        "expert_lr_scale",
        "sequence_length",
        "epochs",
        "seed",
        "learning_rate",
        "weight_decay",
        "betas",
        "warmup_ratio",
        "min_lr_ratio",
    }
    unknown = config.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown SFT configuration keys: {sorted(unknown)}")
    required = {"checkpoint", "data", "experiment", "batch_size", "world_size"}
    if missing := required - config.keys():
        raise ValueError(f"Missing SFT configuration keys: {sorted(missing)}")
    execution_keys = {
        "output",
        "device",
        "precision",
        "micro_batch_size",
        "save_every",
        "gradient_checkpointing",
        "eval_data",
        "eval_max_batches",
    }
    if unknown := execution.keys() - execution_keys:
        raise ValueError(f"Unknown SFT execution keys: {sorted(unknown)}")
    for mapping, names in ((config, ("checkpoint", "data")), (execution, ("output", "eval_data"))):
        for name in names:
            if name in mapping:
                if not isinstance(mapping[name], str) or not mapping[name]:
                    raise ValueError(f"SFT {name} must be a nonempty local path")
                mapping[name] = str((path.parent / mapping[name]).resolve())
    for name in ("micro_batch_size", "save_every", "eval_max_batches"):
        if name in execution and (type(execution[name]) is not int or execution[name] < 1):
            raise ValueError(f"SFT {name} must be a positive integer")
    if (
        "gradient_checkpointing" in execution
        and type(execution["gradient_checkpointing"]) is not bool
    ):
        raise ValueError("SFT gradient_checkpointing must be a boolean")
    if "device" in execution and not isinstance(execution["device"], str):
        raise ValueError("SFT device must be a string")
    if "precision" in execution and execution["precision"] not in {"fp32", "bf16"}:
        raise ValueError("SFT precision must be fp32 or bf16")
    config.setdefault("expert_lr_scale", 1.0)
    return build_sft_plan(**config), execution


def execute_sft_config(
    path: str | Path,
    *,
    run: bool = False,
    steps: int | None = None,
    resume: str | Path | None = None,
) -> dict:
    """Shared front door for CLI and Python; execution is always opt-in."""
    plan, execution = load_sft_config(path)
    if not run:
        if steps is not None or resume is not None:
            raise ValueError("--steps and --resume require explicit --run")
        return {**plan, "execution": execution}
    if not {"output", "device"} <= execution.keys():
        raise ValueError("SFT execution requires output and explicit device in the configuration")
    output = execution.pop("output")
    return run_sft(plan, output, steps=steps, resume=resume, **execution)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--experiment", choices=[EXPERIMENT], required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--expert-lr-scale", type=float, required=True)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--cpu-diagnostic",
        action="store_true",
        help="Explicitly perform one local CPU update, without saving",
    )
    parser.add_argument(
        "--run", action="store_true", help="Explicitly run the local full-parameter experiment"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument(
        "--steps", type=int, help="Bounded optimizer-update benchmark; keep the full schedule"
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--eval-max-batches", type=int, help="Partial evaluation for a bounded benchmark only"
    )
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    try:
        plan = build_sft_plan(
            args.checkpoint,
            args.data,
            experiment=args.experiment,
            batch_size=args.batch_size,
            world_size=args.world_size,
            expert_lr_scale=args.expert_lr_scale,
            sequence_length=args.sequence_length,
            epochs=args.epochs,
        )
        if args.cpu_diagnostic and args.run:
            raise ValueError("Choose CPU diagnostic or local run")
        if args.run:
            if args.output is None or args.device is None:
                raise ValueError("Local run requires --output and explicit --device")
            result = run_sft(
                plan,
                args.output,
                device=args.device,
                micro_batch_size=args.micro_batch_size,
                steps=args.steps,
                resume=args.resume,
                eval_max_batches=args.eval_max_batches,
                save_every=args.save_every,
                gradient_checkpointing=args.gradient_checkpointing,
                precision=args.precision,
            )
        else:
            result = cpu_diagnostic(plan) if args.cpu_diagnostic else plan
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        parser.exit(2, f"SFT preflight failed: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
