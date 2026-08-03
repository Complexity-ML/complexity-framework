#!/usr/bin/env python3
"""SFT runner for local o200k Complexity checkpoints.

Supports inspectable JSONL records and pre-tokenized SFT shards containing
causally aligned ``input_ids.bin`` / ``labels.bin`` pairs. The binary format
uses ``-100`` labels for prompt tokens so only assistant responses contribute
to the supervised loss.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from complexity.config import ModelConfig
from complexity.core.losses import causal_lm_loss_from_hidden
from complexity.inference.chat_template import (
    default_chat_template,
    load_chat_template,
    render_inference_prompt,
    render_messages_before_assistant,
)
from complexity.models import ComplexityModel
from complexity.tokenizer import Tokenizer
from complexity.training.o200k_pretrain import init_distributed
from complexity.utils import autocast, autocast_dtype, empty_cache, synchronize
from complexity.utils.device import backend_metadata, configure_torch_acceleration
from complexity.utils.local_checkpoint import save_local_checkpoint

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
for noisy_logger in ("httpx", "httpcore", "huggingface_hub", "datasets"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


TOY_RECORDS = [
    {
        "messages": [
            {"role": "user", "content": "Explique Token-Routed MLP en une phrase."},
            {
                "role": "assistant",
                "content": "Token-Routed MLP envoie chaque token vers des experts fixes tout en gardant un chemin partagé dense.",
            },
        ]
    },
    {
        "instruction": "Donne une réponse courte.",
        "input": "Pourquoi masquer le prompt en SFT ?",
        "output": "Pour apprendre seulement la réponse assistant, pas recopier l'instruction.",
    },
    {
        "prompt": "User:\nQuel est le but du SFT ?\n\nAssistant:\n",
        "completion": "Adapter le modèle à un style de réponse utile sans refaire tout le pré-entraînement.",
    },
]


def load_checkpoint_state(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[Path, dict[str, Any]]:
    ckpt = Path(path)
    if ckpt.is_file():
        return ckpt.parent, torch.load(ckpt, map_location=map_location)
    ckpt_file = ckpt / "checkpoint.pt"
    if ckpt_file.exists():
        return ckpt, torch.load(ckpt_file, map_location=map_location)
    latest = ckpt / "latest"
    if latest.exists():
        target = latest.read_text(encoding="utf-8").strip()
        if target:
            return load_checkpoint_state(ckpt / target, map_location=map_location)
    config_file = ckpt / "config.json"
    weights_file = ckpt / "model.safetensors"
    if config_file.exists() and weights_file.exists():
        config = json.loads(config_file.read_text(encoding="utf-8"))
        device = str(map_location)
        return ckpt, {
            "config": config,
            "model": load_safetensors(str(weights_file), device=device),
            "export_format": "huggingface_safetensors",
        }
    raise FileNotFoundError(
        f"No checkpoint.pt or config.json + model.safetensors found under {ckpt}"
    )


def checkpoint_config(state: dict[str, Any]) -> ModelConfig:
    if "config" not in state:
        raise KeyError("Checkpoint does not contain a 'config' entry")
    return ModelConfig.from_dict(state["config"])


def format_record(
    record: dict[str, Any],
    chat_template: dict[str, Any] | None = None,
) -> tuple[str, str]:
    template = chat_template or default_chat_template()
    if "messages" in record:
        messages = record["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        assistant_idx = None
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "assistant":
                assistant_idx = idx
                break
        if assistant_idx is None:
            raise ValueError("messages record has no assistant message")

        prompt = render_messages_before_assistant(
            messages[:assistant_idx],
            template,
        )
        completion = str(messages[assistant_idx].get("content", "")).strip()
        return prompt, completion

    if "instruction" in record or "output" in record:
        instruction = str(record.get("instruction", "")).strip()
        extra_input = str(record.get("input", "")).strip()
        output = str(record.get("output", record.get("response", ""))).strip()
        user = instruction if not extra_input else f"{instruction}\n\n{extra_input}"
        return render_inference_prompt(user, template), output

    if "prompt" in record and ("completion" in record or "response" in record):
        return str(record["prompt"]), str(record.get("completion", record.get("response", "")))

    raise ValueError("Supported JSONL formats: messages, instruction/output, or prompt/completion")


def encode_sft_example(
    tokenizer: Tokenizer,
    record: dict[str, Any],
    seq_len: int,
    min_completion_tokens: int,
    chat_template: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    prompt, completion = format_record(record, chat_template)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        completion_ids = completion_ids + [eos_id]

    if not completion_ids:
        completion_ids = [eos_id if eos_id is not None else 0]

    max_prompt = max(1, seq_len + 1 - max(min_completion_tokens, 1))
    if len(prompt_ids) > max_prompt:
        prompt_ids = prompt_ids[-max_prompt:]

    max_completion = max(1, seq_len + 1 - len(prompt_ids))
    completion_ids = completion_ids[:max_completion]
    full = prompt_ids + completion_ids
    if len(full) < 2:
        full = full + [eos_id if eos_id is not None else 0]
    full = full[: seq_len + 1]

    input_ids = full[:-1]
    labels = full[1:]
    prompt_targets = max(0, min(len(labels), len(prompt_ids) - 1))
    labels[:prompt_targets] = [-100] * prompt_targets

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_id if eos_id is not None else 0
    pad = seq_len - len(input_ids)
    if pad > 0:
        input_ids = input_ids + [pad_id] * pad
        labels = labels + [-100] * pad

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class SFTJsonlDataset(IterableDataset):
    def __init__(
        self,
        path: str | None,
        tokenizer_path: str,
        seq_len: int,
        seed: int,
        rank: int,
        world_size: int,
        min_completion_tokens: int = 32,
        chat_template: dict[str, Any] | None = None,
        repeat: bool = True,
        epochs: int | None = None,
    ):
        self.records = load_records(path)
        self.tokenizer_path = tokenizer_path
        self.seq_len = seq_len
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.min_completion_tokens = min_completion_tokens
        self.chat_template = chat_template or default_chat_template()
        self.repeat = repeat
        self.epochs = epochs

    def __iter__(self):
        tokenizer = Tokenizer.load(self.tokenizer_path)
        records = list(self.records)
        epoch = 0
        while True:
            rng = random.Random(self.seed + epoch)
            rng.shuffle(records)
            for idx, record in enumerate(records):
                if idx % self.world_size != self.rank:
                    continue
                yield encode_sft_example(
                    tokenizer,
                    record,
                    self.seq_len,
                    self.min_completion_tokens,
                    self.chat_template,
                )
            if not self.repeat:
                return
            epoch += 1
            if self.epochs is not None and epoch >= self.epochs:
                return


def load_model_state_compat(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> None:
    """Load current and historical checkpoints without changing their routes.

    Historical top-2 checkpoints persist the primary ``token_to_expert`` table
    but predate the derived ``topk_token_to_expert`` buffer.  The current model
    reconstructs that secondary table deterministically from the same config,
    so this one missing buffer is safe to tolerate.  Every parameter and every
    other buffer remains strict.
    """

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    tolerated_suffixes = (
        "topk_token_to_expert",
        "rotary_emb.inv_freq",
        "pair_hash_route_codes",
        "pair_hash_expert_pairs",
    )
    unexpected_missing = [
        key for key in missing if not key.endswith(tolerated_suffixes)
    ]
    if unexpected_missing or unexpected:
        raise RuntimeError(
            "Checkpoint mismatch: "
            f"missing={unexpected_missing}, unexpected={list(unexpected)}"
        )


class SFTBinDataset(IterableDataset):
    """Stream independent, indexed SFT examples from memory-mapped shards."""

    def __init__(
        self,
        root: str | Path,
        seq_len: int,
        seed: int,
        rank: int,
        world_size: int,
        repeat: bool = True,
        epochs: int | None = None,
    ):
        root = Path(root)
        dataset_root = root
        if (root / "train" / "sft.idx.json").exists():
            root = root / "train"
        elif (root / "sft.idx.json").exists():
            dataset_root = root.parent
        index_path = root / "sft.idx.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"SFT shard index not found: {index_path}. Pass the tokenized "
                "dataset root or its train partition."
            )
        self.root = root
        self.metadata = json.loads(index_path.read_text())
        if self.metadata.get("format") != "complexity-sft-token-shard-v1":
            raise ValueError(f"Unsupported SFT shard format: {self.metadata.get('format')}")
        self.chat_template = load_chat_template(dataset_root)
        metadata_template = self.metadata.get("chat_template_id")
        if metadata_template and metadata_template != self.chat_template["id"]:
            raise ValueError(
                "SFT shard/template mismatch: "
                f"index={metadata_template} template={self.chat_template['id']}"
            )
        self.input_ids = np.memmap(root / "input_ids.bin", mode="r", dtype=np.dtype("<u4"))
        self.labels = np.memmap(root / "labels.bin", mode="r", dtype=np.dtype("<i4"))
        if len(self.input_ids) != len(self.labels):
            raise ValueError("SFT input and label shards have different lengths")
        if len(self.input_ids) != int(self.metadata["num_tokens"]):
            raise ValueError("SFT shard length does not match sft.idx.json")
        with (root / "examples.jsonl").open(encoding="utf-8") as handle:
            self.examples = [json.loads(line) for line in handle if line.strip()]
        if len(self.examples) != int(self.metadata["examples"]):
            raise ValueError("SFT example index count does not match sft.idx.json")
        self.seq_len = seq_len
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.repeat = repeat
        self.epochs = epochs
        self.pad_id = int(self.metadata["eos_token_id"])

    def _tensor_example(self, example: dict[str, Any]) -> dict[str, torch.Tensor]:
        start = int(example["offset"])
        length = int(example["num_tokens"])
        end = start + length
        if start < 0 or length <= 0 or end > len(self.input_ids):
            raise ValueError(f"Invalid SFT example bounds: {example}")
        input_ids = np.asarray(self.input_ids[start:end], dtype=np.int64)
        labels = np.asarray(self.labels[start:end], dtype=np.int64)
        if length > self.seq_len:
            # Retain the final assistant response and its EOS target.
            input_ids = input_ids[-self.seq_len :]
            labels = labels[-self.seq_len :]
        elif length < self.seq_len:
            padding = self.seq_len - length
            input_ids = np.pad(input_ids, (0, padding), constant_values=self.pad_id)
            labels = np.pad(labels, (0, padding), constant_values=-100)
        if not np.any(labels != -100):
            raise ValueError(f"SFT example has no supervised assistant tokens: {example['example_id']}")
        return {
            "input_ids": torch.from_numpy(input_ids.copy()),
            "labels": torch.from_numpy(labels.copy()),
        }

    def __iter__(self):
        epoch = 0
        indices = list(range(len(self.examples)))
        while True:
            random.Random(self.seed + epoch).shuffle(indices)
            for position, example_index in enumerate(indices):
                if position % self.world_size != self.rank:
                    continue
                yield self._tensor_example(self.examples[example_index])
            if not self.repeat:
                break
            epoch += 1
            if self.epochs is not None and epoch >= self.epochs:
                break


def load_records(path: str | None) -> list[dict[str, Any]]:
    if path is None:
        return list(TOY_RECORDS)
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    if not records:
        raise ValueError(f"No SFT records found in {path}")
    return records


def build_optimizer(args, raw_model):
    decay, no_decay = [], []
    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "bias" in name or "norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )


def configure_trainable_parameters(raw_model, *, freeze_token_io: bool) -> dict[str, int | bool]:
    """Optionally freeze the large token embedding/output parameter tables.

    The o200k 100M profile stores most parameters in ``embed_tokens``. Because
    the output projection is tied to the same tensor, freezing it preserves
    both token input and token output geometry while gradients still flow
    through the fixed output projection into the transformer hidden states.
    Untied ``lm_head`` parameters are frozen as well.
    """

    if freeze_token_io:
        raw_model.embed_tokens.requires_grad_(False)
        if raw_model.lm_head is not None:
            raw_model.lm_head.requires_grad_(False)

    total = sum(param.numel() for param in raw_model.parameters())
    trainable = sum(
        param.numel() for param in raw_model.parameters() if param.requires_grad
    )
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "token_io_frozen": bool(freeze_token_io),
    }


def update_early_stopping(
    best_loss: float,
    evaluations_without_improvement: int,
    current_loss: float,
    *,
    min_delta: float,
) -> tuple[bool, float, int]:
    """Update deterministic validation tracking state."""

    improved = current_loss < best_loss - min_delta
    if improved:
        return True, current_loss, 0
    return False, best_loss, evaluations_without_improvement + 1


def label_stats(labels: torch.Tensor, vocab_size: int) -> dict[str, int]:
    valid = labels != -100
    if not valid.any():
        return {
            "supervised_tokens": 0,
            "min_label": -1,
            "max_label": -1,
            "bad_labels": 0,
        }
    valid_labels = labels[valid]
    bad = (valid_labels < 0) | (valid_labels >= vocab_size)
    return {
        "supervised_tokens": int(valid.sum().item()),
        "min_label": int(valid_labels.min().item()),
        "max_label": int(valid_labels.max().item()),
        "bad_labels": int(bad.sum().item()),
    }


def sft_loss_from_hidden(
    hidden_states: torch.Tensor,
    output_weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_tokens: int,
) -> torch.Tensor:
    """Chunked CE for SFT, computing logits/CE in fp32 for stability."""

    flat_hidden = hidden_states.reshape(-1, hidden_states.size(-1))
    flat_labels = labels.reshape(-1)
    valid = flat_labels != -100
    denom = valid.sum().clamp_min(1).to(dtype=torch.float32)
    total = flat_hidden.new_zeros((), dtype=torch.float32)
    chunk = max(1, int(chunk_tokens or flat_hidden.size(0)))
    for start in range(0, flat_hidden.size(0), chunk):
        end = min(start + chunk, flat_hidden.size(0))
        labels_chunk = flat_labels[start:end]
        valid_chunk = labels_chunk != -100
        if not valid_chunk.any():
            continue
        hidden_chunk = flat_hidden[start:end][valid_chunk].float()
        labels_chunk = labels_chunk[valid_chunk]
        logits = F.linear(hidden_chunk, output_weight.float())
        total = total + F.cross_entropy(
            logits,
            labels_chunk,
            reduction="sum",
        )
    return total / denom


@torch.no_grad()
def evaluate_sft(
    model,
    raw_model,
    loader,
    *,
    device: torch.device,
    amp_dtype,
    fp32_loss: bool,
    chunk_tokens: int,
    distributed: bool,
    max_batches: int,
) -> tuple[float, int]:
    model.eval()
    # Metal does not implement float64 tensors. Float32 is sufficient for
    # aggregating per-batch validation loss and supervised-token counts.
    loss_sum = torch.zeros((), dtype=torch.float32, device=device)
    token_count = torch.zeros((), dtype=torch.float32, device=device)
    for batch_index, batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        supervised = int((labels != -100).sum().item())
        if supervised == 0:
            continue
        with autocast(device, dtype=amp_dtype, enabled=amp_dtype is not None):
            outputs = model(input_ids, return_logits=False)
            if fp32_loss:
                loss = sft_loss_from_hidden(
                    outputs["last_hidden_state"],
                    raw_model.embed_tokens.weight,
                    labels,
                    chunk_tokens=chunk_tokens,
                )
            else:
                loss, _ = causal_lm_loss_from_hidden(
                    outputs["last_hidden_state"],
                    raw_model.embed_tokens.weight,
                    labels,
                    chunk_tokens=chunk_tokens,
                )
        loss_sum += loss.detach().float() * supervised
        token_count += supervised
    if distributed:
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
    model.train()
    if token_count.item() == 0:
        raise ValueError("SFT evaluation contains no supervised assistant tokens")
    return float((loss_sum / token_count).item()), int(token_count.item())


def save_checkpoint(
    args,
    raw_model,
    optimizer,
    scheduler,
    config,
    source_checkpoint: str,
    step: int,
    is_main: bool,
    distributed: bool,
    chat_template: dict[str, Any],
    *,
    force: bool = False,
    save_dir: str | Path | None = None,
    eval_loss: float | None = None,
):
    if distributed:
        dist.barrier()
    if not is_main or (args.save_steps <= 0 and not force):
        if distributed:
            dist.barrier()
        return
    checkpoint_state = {
        "step": step,
        "model": {k: v.detach().cpu() for k, v in raw_model.state_dict().items()},
        "config": config.to_dict(),
        "args": vars(args),
        "sft_source_checkpoint": source_checkpoint,
        "chat_template": chat_template,
        "backend": backend_metadata(kernel_policy=getattr(args, "use_custom_kernels", "auto")),
    }
    if eval_loss is not None:
        checkpoint_state["sft_eval_loss"] = float(eval_loss)
    if not args.save_model_only:
        checkpoint_state["optimizer"] = optimizer.state_dict()
        checkpoint_state["scheduler"] = scheduler.state_dict()
    ckpt_dir = save_local_checkpoint(
        save_dir or args.save_dir,
        step=step,
        total_limit=args.save_total_limit,
        state=checkpoint_state,
    )
    logger.info(f"Checkpoint saved: {ckpt_dir}")
    if distributed:
        dist.barrier()
    return ckpt_dir


def save_best_checkpoint(
    args,
    raw_model,
    optimizer,
    scheduler,
    config,
    source_checkpoint: str,
    step: int,
    eval_loss: float,
    eval_tokens: int,
    is_main: bool,
    distributed: bool,
    chat_template: dict[str, Any],
) -> None:
    """Save a validation-selected checkpoint independently of periodic saves."""

    best_root = Path(args.save_dir) / "best"
    checkpoint_dir = save_checkpoint(
        args,
        raw_model,
        optimizer,
        scheduler,
        config,
        source_checkpoint,
        step,
        is_main,
        distributed,
        chat_template,
        force=True,
        save_dir=best_root,
        eval_loss=eval_loss,
    )
    if is_main and checkpoint_dir is not None:
        metadata = {
            "step": step,
            "eval_loss": eval_loss,
            "eval_tokens": eval_tokens,
            "checkpoint": str(checkpoint_dir),
        }
        (Path(args.save_dir) / "best.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SFT a local o200k Complexity checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint dir or checkpoint.pt to fine-tune")
    parser.add_argument("--tokenizer", default="./tokenizer-o200k")
    dataset = parser.add_mutually_exclusive_group()
    dataset.add_argument("--jsonl", default=None, help="Inspectable SFT JSONL.")
    dataset.add_argument(
        "--sft-bin",
        default=None,
        help="Tokenized SFT root containing train/input_ids.bin and train/labels.bin.",
    )
    parser.add_argument(
        "--eval-jsonl",
        default=None,
        help="Finite held-out JSONL evaluated during JSONL-based SFT.",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Consume the complete training dataset this many times; 0 keeps the step-limited stream.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--min-completion-tokens", type=int, default=32)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--freeze-token-io",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze token embeddings and the tied/untied LM head during SFT.",
    )
    parser.add_argument(
        "--use-custom-kernels",
        choices=["auto", "true", "false"],
        default="auto",
        help="Custom Triton/CUDA kernels. auto enables NVIDIA CUDA, disables ROCm by default.",
    )
    parser.add_argument("--grad-ckpt", action="store_true")
    parser.add_argument("--loss-chunk-tokens", type=int, default=1024)
    parser.add_argument(
        "--sft-fp32-loss",
        action="store_true",
        default=True,
        help="Compute the tied o200k SFT loss in fp32 chunks for stability.",
    )
    parser.add_argument(
        "--no-sft-fp32-loss",
        dest="sft_fp32_loss",
        action="store_false",
        help="Use the generic causal_lm_loss_from_hidden path.",
    )
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Evaluate every N steps for SFT bin datasets; 0 disables evaluation.",
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=0,
        help="Maximum eval batches; 0 evaluates the complete held-out shard.",
    )
    parser.add_argument(
        "--eval-at-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Measure the held-out SFT loss before the first optimizer step.",
    )
    parser.add_argument(
        "--save-best",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save checkpoints only when held-out SFT loss improves.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many non-improving evaluations; 0 disables it.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum held-out loss decrease counted as an improvement.",
    )
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-dir", default="checkpoints/sft-100m-o200k-tr")
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument(
        "--save-model-only",
        action="store_true",
        help="Omit optimizer and scheduler state from checkpoints used for evaluation/inference.",
    )
    parser.add_argument("--run-name", default="sft-100m-o200k-tr")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--empty-cache-every", type=int, default=50)
    parser.add_argument("--cpu", action="store_true", help="Force CPU for smoke tests")
    return parser


def main():
    args = build_parser().parse_args()
    if args.cpu:
        device = torch.device("cpu")
        distributed = False
        rank = local_rank = 0
        world_size = 1
        torch.manual_seed(args.seed)
    else:
        device, distributed, rank, local_rank, world_size = init_distributed(args.seed)
    is_main = rank == 0
    kernel_policy = (
        True if args.use_custom_kernels == "true"
        else False if args.use_custom_kernels == "false"
        else "auto"
    )
    args.use_custom_kernels = kernel_policy
    configure_torch_acceleration(kernel_policy=kernel_policy, log=is_main)

    ckpt_dir, state = load_checkpoint_state(args.checkpoint, map_location="cpu")
    config = checkpoint_config(state)
    config.use_custom_kernels = kernel_policy
    raw_model = ComplexityModel(config).to(device)
    load_model_state_compat(raw_model, state["model"])
    if args.grad_ckpt:
        raw_model.gradient_checkpointing_enable()
    parameter_stats = configure_trainable_parameters(
        raw_model,
        freeze_token_io=args.freeze_token_io,
    )
    if parameter_stats["trainable"] == 0:
        raise ValueError("SFT configuration froze every model parameter")

    model = raw_model
    if distributed:
        model = DDP(
            raw_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    if args.sft_bin is not None:
        train_ds = SFTBinDataset(
            args.sft_bin,
            args.seq_len,
            args.seed,
            rank,
            world_size,
            repeat=True,
            epochs=args.epochs or None,
        )
        eval_ds = SFTBinDataset(
            Path(args.sft_bin) / "eval"
            if (Path(args.sft_bin) / "eval" / "sft.idx.json").exists()
            else Path(args.sft_bin),
            args.seq_len,
            args.seed,
            rank,
            world_size,
            repeat=False,
        ) if (
            (Path(args.sft_bin) / "eval" / "sft.idx.json").exists()
            or Path(args.sft_bin).name == "eval"
        ) else None
    else:
        train_ds = SFTJsonlDataset(
            args.jsonl,
            args.tokenizer,
            args.seq_len,
            args.seed,
            rank,
            world_size,
            args.min_completion_tokens,
            repeat=True,
            epochs=args.epochs or None,
        )
        eval_ds = (
            SFTJsonlDataset(
                args.eval_jsonl,
                args.tokenizer,
                args.seq_len,
                args.seed,
                rank,
                world_size,
                args.min_completion_tokens,
                repeat=False,
            )
            if args.eval_jsonl is not None
            else None
        )
    chat_template = train_ds.chat_template
    if eval_ds is not None and eval_ds.chat_template != chat_template:
        raise ValueError("Train and eval SFT shards use different chat templates")
    loader_kwargs = {"batch_size": args.batch_size, "pin_memory": False}
    if args.num_workers > 0:
        loader_kwargs.update(num_workers=args.num_workers, persistent_workers=True)
    train_loader = DataLoader(train_ds, **loader_kwargs)
    eval_loader = DataLoader(eval_ds, **loader_kwargs) if eval_ds is not None else None

    optimizer = build_optimizer(args, raw_model)
    warmup = max(1, int(args.steps * args.warmup_ratio))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, args.steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    amp_dtype = autocast_dtype(device) if args.bf16 else None

    run_dir = Path("runs") / args.run_name
    csv_file = None
    writer = None
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"SFT source: {ckpt_dir} (pretrain step={state.get('step', 'unknown')})")
        logger.info(f"Model: {parameter_stats['total'] / 1e6:.1f}M params")
        logger.info(
            "Parameters: "
            f"trainable={parameter_stats['trainable'] / 1e6:.1f}M "
            f"frozen={parameter_stats['frozen'] / 1e6:.1f}M "
            f"token_io_frozen={parameter_stats['token_io_frozen']}"
        )
        backend = backend_metadata(kernel_policy=kernel_policy)
        logger.info(
            "Backend: "
            f"{backend['backend']} device={backend['device_name']} "
            f"matmul={backend['matmul']} distributed={backend['distributed']} "
            f"sdpa={backend['sdpa']} flash={backend['flash_attention']} "
            f"custom_triton={backend['custom_triton']}"
        )
        logger.info(
            f"Config: vocab={config.vocab_size}, hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
            f"GQA={config.num_attention_heads}/{config.num_key_value_heads}, "
            f"TR experts={config.num_experts}, top_k={config.top_k}, use_mu={config.use_mu_guidance}"
        )
        if args.sft_bin is not None:
            logger.info(
                f"Dataset: SFT bin {train_ds.root} "
                f"({len(train_ds.examples):,} examples, "
                f"{train_ds.metadata['supervised_tokens']:,} supervised tokens)"
            )
            if train_ds.metadata["supervised_tokens"] < 3_000_000:
                logger.warning(
                    "Training shard contains fewer than 3,000,000 supervised "
                    "tokens; use held-out early stopping and treat the run as "
                    "a small-data adaptation."
                )
            if eval_ds is not None and len(eval_ds.examples) < 500:
                logger.warning(
                    "Held-out SFT evaluation contains fewer than 500 examples; "
                    "loss is useful for early stopping but not a stable general "
                    "capability estimate."
                )
        elif args.jsonl is None:
            logger.info("Dataset: built-in toy SFT records")
        else:
            logger.info(f"Dataset: {args.jsonl} ({len(train_ds.records)} records)")
            if eval_ds is not None:
                logger.info(
                    f"Evaluation: {args.eval_jsonl} "
                    f"({len(eval_ds.records)} held-out records)"
                )
        logger.info(f"Chat template: {chat_template['id']}")
        csv_file = (run_dir / "metrics.csv").open("w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow([
            "step", "train_loss", "train_ppl", "eval_loss", "eval_ppl", "lr", "tok_s",
            "supervised_tokens", "min_label", "max_label", "bad_labels",
        ])
        csv_file.flush()

    best_eval_loss = math.inf
    evaluations_without_improvement = 0
    if eval_loader is not None and args.eval_steps > 0 and args.eval_at_start:
        initial_eval_loss, initial_eval_tokens = evaluate_sft(
            model,
            raw_model,
            eval_loader,
            device=device,
            amp_dtype=amp_dtype,
            fp32_loss=args.sft_fp32_loss,
            chunk_tokens=args.loss_chunk_tokens,
            distributed=distributed,
            max_batches=args.eval_batches,
        )
        best_eval_loss = initial_eval_loss
        if is_main:
            logger.info(
                f"SFT eval step=0: loss={initial_eval_loss:.6f} "
                f"ppl={math.exp(min(initial_eval_loss, 20)):.2f} "
                f"tokens={initial_eval_tokens:,}"
            )
            writer.writerow([
                0,
                "",
                "",
                f"{initial_eval_loss:.6f}",
                f"{math.exp(min(initial_eval_loss, 20)):.2f}",
                f"{optimizer.param_groups[0]['lr']:.6e}",
                "",
                initial_eval_tokens,
                "",
                "",
                0,
            ])
            csv_file.flush()

    model.train()
    pbar = tqdm(total=args.steps, desc="SFT o200k TR", unit="step", dynamic_ncols=True) if is_main else None
    t_log = time.perf_counter()
    tokens_since_log = 0
    last_step = 0

    for step, batch in enumerate(train_loader, start=1):
        if step > args.steps:
            break
        last_step = step
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        stats = label_stats(labels, config.vocab_size)
        if stats["supervised_tokens"] == 0:
            continue
        if stats["bad_labels"] > 0:
            raise ValueError(
                "SFT batch has labels outside model vocab: "
                f"min={stats['min_label']} max={stats['max_label']} "
                f"vocab={config.vocab_size} bad={stats['bad_labels']}"
            )
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, dtype=amp_dtype, enabled=amp_dtype is not None):
            outputs = model(input_ids, return_logits=False)
            if args.sft_fp32_loss:
                loss = sft_loss_from_hidden(
                    outputs["last_hidden_state"],
                    raw_model.embed_tokens.weight,
                    labels,
                    chunk_tokens=args.loss_chunk_tokens,
                )
            else:
                loss, metrics = causal_lm_loss_from_hidden(
                    outputs["last_hidden_state"],
                    raw_model.embed_tokens.weight,
                    labels,
                    chunk_tokens=args.loss_chunk_tokens,
                )
        if args.sft_fp32_loss:
            metrics_ce = float(loss.detach().item())
        else:
            metrics_ce = float(metrics.ce)
        if not math.isfinite(metrics_ce):
            raise FloatingPointError(
                "Non-finite SFT loss before backward: "
                f"loss={metrics_ce} supervised_tokens={stats['supervised_tokens']} "
                f"min_label={stats['min_label']} max_label={stats['max_label']} "
                f"vocab={config.vocab_size}"
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        tokens_since_log += args.batch_size * args.seq_len * world_size
        if pbar is not None:
            pbar.update(1)

        should_eval = (
            eval_loader is not None
            and args.eval_steps > 0
            and (step % args.eval_steps == 0 or step == args.steps)
        )
        should_log = step == 1 or step % args.log_steps == 0 or should_eval
        eval_loss = None
        stop_training = False
        if should_eval:
            eval_loss, eval_tokens = evaluate_sft(
                model,
                raw_model,
                eval_loader,
                device=device,
                amp_dtype=amp_dtype,
                fp32_loss=args.sft_fp32_loss,
                chunk_tokens=args.loss_chunk_tokens,
                distributed=distributed,
                max_batches=args.eval_batches,
            )
            if is_main:
                logger.info(
                    f"SFT eval step={step}: loss={eval_loss:.6f} "
                    f"ppl={math.exp(min(eval_loss, 20)):.2f} tokens={eval_tokens:,}"
                )
            improved, best_eval_loss, evaluations_without_improvement = update_early_stopping(
                best_eval_loss,
                evaluations_without_improvement,
                eval_loss,
                min_delta=args.early_stopping_min_delta,
            )
            if improved and args.save_best:
                save_best_checkpoint(
                    args,
                    raw_model,
                    optimizer,
                    scheduler,
                    config,
                    str(ckpt_dir),
                    step,
                    eval_loss,
                    eval_tokens,
                    is_main,
                    distributed,
                    chat_template,
                )
            if (
                args.early_stopping_patience > 0
                and evaluations_without_improvement >= args.early_stopping_patience
            ):
                stop_training = True
        if should_log:
            synchronize(device)
            now = time.perf_counter()
            tok_s = tokens_since_log / max(1e-9, now - t_log)
            train_loss = metrics_ce
            if distributed:
                loss_tensor = torch.tensor(train_loss, device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                train_loss = float(loss_tensor.item())
            train_ppl = math.exp(min(train_loss, 20))
            lr_now = scheduler.get_last_lr()[0]
            if is_main:
                writer.writerow([
                    step,
                    f"{train_loss:.6f}",
                    f"{train_ppl:.2f}",
                    "" if eval_loss is None else f"{eval_loss:.6f}",
                    "" if eval_loss is None else f"{math.exp(min(eval_loss, 20)):.2f}",
                    f"{lr_now:.6e}",
                    f"{tok_s:.0f}",
                    stats["supervised_tokens"], stats["min_label"], stats["max_label"], stats["bad_labels"],
                ])
                csv_file.flush()
                pbar.set_postfix(loss=f"{train_loss:.4f}", tok_s=f"{tok_s:.0f}")
            t_log = now
            tokens_since_log = 0

        if args.empty_cache_every > 0 and step % args.empty_cache_every == 0:
            empty_cache(device)
        if args.save_steps > 0 and step % args.save_steps == 0:
            save_checkpoint(
                args,
                raw_model,
                optimizer,
                scheduler,
                config,
                str(ckpt_dir),
                step,
                is_main,
                distributed,
                chat_template,
            )
        if stop_training:
            if is_main:
                logger.info(
                    f"Early stopping at step={step}: best_eval_loss={best_eval_loss:.6f}, "
                    f"evaluations_without_improvement={evaluations_without_improvement}"
                )
            break

    if args.save_steps > 0 and last_step > 0 and last_step % args.save_steps != 0:
        save_checkpoint(
            args,
            raw_model,
            optimizer,
            scheduler,
            config,
            str(ckpt_dir),
            last_step,
            is_main,
            distributed,
            chat_template,
        )
    if pbar is not None:
        pbar.close()
    if csv_file is not None:
        csv_file.close()
        logger.info(f"Metrics saved: {run_dir / 'metrics.csv'}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
