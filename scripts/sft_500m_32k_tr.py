#!/usr/bin/env python3
"""SFT runner for the 500M TR-HASH model with native 32k token shards.

Supports inspectable JSONL records and pre-tokenized SFT shards containing
causally aligned ``input_ids.bin`` / ``labels.bin`` pairs. The binary format
uses ``-100`` labels for prompt tokens so only assistant responses contribute
to the supervised loss.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import os
import random
import time
from collections import Counter
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
from complexity.core.losses import (
    causal_lm_loss_from_hidden,
    fused_linear_causal_lm_loss,
    log_liger_fused_linear_ce_status,
)
from complexity.inference.chat_template import (
    default_chat_template,
    load_chat_template,
    render_inference_prompt,
    render_messages_before_assistant,
)
from complexity.models import ComplexityModel
from complexity.tokenizer import Tokenizer
from complexity.training.finetuning import (
    REFINEMENT_STAGE,
    SUPERVISED_FINETUNING_STAGE,
    TEXT_MODEL_FAMILY,
    TEXT_SUPERVISED_FINETUNING,
    validate_full_parameter_finetuning,
    validate_training_stage_transition,
)
from complexity.training.lora import (
    LoRAConfig,
    adapter_state_dict,
    apply_lora,
    load_adapter_state_dict,
    merged_model_state_dict,
    unmerge_adapter_from_base,
)
from complexity.training.sequence_packing import (
    SequencePackingPlan,
    pack_example_lengths,
    resolve_epoch_schedule,
)
from complexity.training.sft_curriculum import (
    derive_task_loss_weights,
    load_curriculum,
    load_projected_metadata,
    loss_weight_key,
    select_stage_examples,
)
from complexity.training.sft_shard import validate_shard_supervision
from complexity.utils import autocast, autocast_dtype, empty_cache, setup_mps, synchronize
from complexity.utils.device import backend_metadata, configure_torch_acceleration
from complexity.utils.local_checkpoint import save_local_checkpoint
from complexity.utils.token_routed_conversion import (
    convert_token_routed_checkpoint,
)

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

REASONING_ENVELOPE_PLANS = {
    "reasoning_verification": (
        "I should solve the requested relation, then verify the result independently."
    ),
    "planning_comparison": (
        "I should test each option against every binding constraint before selecting a plan."
    ),
    "explanation_learning": (
        "I should identify the core idea, explain it simply, and check transfer."
    ),
    "critique_revision": (
        "I should locate the highest-impact defect, then revise without adding unsupported claims."
    ),
    "troubleshooting": (
        "I should isolate the likely cause, propose one bounded test, and state the next action."
    ),
}


def apply_reasoning_envelope(
    input_ids: np.ndarray,
    labels: np.ndarray,
    *,
    prefix_ids: list[int],
    suffix_ids: list[int],
    eos_token_id: int,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Wrap the final supervised assistant span in ``think``/``final`` tags."""

    active = np.flatnonzero(labels != -100)
    if not active.size:
        raise ValueError("reasoning envelope requires supervised assistant tokens")
    discontinuities = np.flatnonzero(np.diff(active) > 1)
    final_start = int(discontinuities[-1] + 1) if discontinuities.size else 0
    final_active = active[final_start:]
    prompt = np.asarray(input_ids[: int(final_active[0]) + 1], dtype=np.int64)
    response = [
        int(token)
        for token in labels[final_active]
        if int(token) != int(eos_token_id)
    ]
    fixed_tokens = len(prefix_ids) + len(suffix_ids) + 1
    response = response[: max(1, seq_len - fixed_tokens)]
    completion = [*prefix_ids, *response, *suffix_ids, int(eos_token_id)]
    prompt = prompt[-max(1, seq_len + 1 - len(completion)) :]
    full = np.asarray([*prompt.tolist(), *completion], dtype=np.int64)[: seq_len + 1]
    rebuilt_inputs = full[:-1]
    rebuilt_labels = full[1:].copy()
    rebuilt_labels[: max(0, len(prompt) - 1)] = -100
    if rebuilt_inputs.size < seq_len:
        padding = seq_len - rebuilt_inputs.size
        rebuilt_inputs = np.pad(
            rebuilt_inputs,
            (0, padding),
            constant_values=int(eos_token_id),
        )
        rebuilt_labels = np.pad(
            rebuilt_labels,
            (0, padding),
            constant_values=-100,
        )
    return rebuilt_inputs, rebuilt_labels


def load_checkpoint_state(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[Path, dict[str, Any]]:
    ckpt = Path(path)
    if ckpt.is_file():
        return ckpt.parent, torch.load(
            ckpt,
            map_location=map_location,
            mmap=True,
            weights_only=False,
        )
    ckpt_file = ckpt / "checkpoint.pt"
    if ckpt_file.exists():
        return ckpt, torch.load(
            ckpt_file,
            map_location=map_location,
            mmap=True,
            weights_only=False,
        )
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
        model_state = load_safetensors(str(weights_file), device=device)
        # Public 20B-token checkpoints predate the canonical TRHashEngineMLP
        # module name.  Convert the complete checkpoint here rather than
        # merely changing config.json: the converter also transplants each
        # layer's trained token->expert route table.  Regenerating those tables
        # would silently attach trained experts to different token identities.
        if config.get("mlp_type") in {
            "token_routed",
            "sort_split",
            "sort_split_moe",
        }:
            converted_model = convert_token_routed_checkpoint(model_state, config)
            config = converted_model.config.to_dict()
            model_state = {
                name: tensor.detach().cpu()
                for name, tensor in converted_model.state_dict().items()
            }
            del converted_model
        return ckpt, {
            "config": config,
            "model": model_state,
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


def pad_epoch_items(
    rank_items: list[Any],
    *,
    all_items: list[Any],
    rank: int,
    world_size: int,
    batch_size: int,
) -> list[Any]:
    """Pad one rank to complete batches without crossing epoch boundaries.

    ``DataLoader`` batches an ``IterableDataset`` as one continuous stream. If
    consecutive epochs are simply concatenated, the last partial batch of one
    epoch is merged with the first examples of the next one. That makes
    ``epochs * ceil(examples / batch_size)`` disagree with the number of
    batches actually produced and can skip the final evaluation boundary.

    Repeating at most one partial batch per rank keeps every epoch independent
    and gives every distributed rank the same deterministic batch count.
    """

    if batch_size < 1:
        raise ValueError("epoch batch size must be positive")
    if world_size < 1:
        raise ValueError("world size must be positive")
    if not all_items:
        raise ValueError("cannot pad an empty epoch")

    examples_per_rank = math.ceil(len(all_items) / world_size)
    target = math.ceil(examples_per_rank / batch_size) * batch_size
    padded = list(rank_items)
    if not padded:
        padded.append(all_items[rank % len(all_items)])
    source = list(padded)
    while len(padded) < target:
        padded.extend(source[: target - len(padded)])
    return padded


class SFTJsonlDataset(IterableDataset):
    packing_contract = "supervised-finetuning"

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
        epoch_batch_size: int | None = None,
        start_step: int = 0,
        pack_sequences: bool = True,
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
        self.epoch_batch_size = epoch_batch_size
        self.start_step = start_step
        self.pack_sequences = bool(pack_sequences)
        self.pad_id = 0
        self.encoded_records: list[dict[str, torch.Tensor]] | None = None
        self.packing_plan: SequencePackingPlan | None = None

    def _ensure_packing(self) -> None:
        if self.encoded_records is not None:
            return
        tokenizer = Tokenizer.load(self.tokenizer_path)
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        self.encoded_records = [
            encode_sft_example(
                tokenizer,
                record,
                self.seq_len,
                self.min_completion_tokens,
                self.chat_template,
            )
            for record in self.records
        ]
        lengths = [self._unpadded_length(row) for row in self.encoded_records]
        self.packing_plan = (
            pack_example_lengths(lengths, sequence_length=self.seq_len, separator_tokens=1)
            if self.pack_sequences
            else None
        )

    @staticmethod
    def _unpadded_length(example: dict[str, torch.Tensor]) -> int:
        supervised = torch.nonzero(example["labels"] != -100, as_tuple=False)
        if not len(supervised):
            raise ValueError("SFT JSONL example has no supervised assistant tokens")
        return int(supervised[-1].item()) + 1

    @property
    def training_items(self) -> int:
        self._ensure_packing()
        return self.packing_plan.packed_items if self.packing_plan is not None else len(self.records)

    def _tensor_item(self, item: int) -> dict[str, torch.Tensor]:
        self._ensure_packing()
        assert self.encoded_records is not None
        if self.packing_plan is None:
            return self.encoded_records[item]
        input_parts: list[torch.Tensor] = []
        label_parts: list[torch.Tensor] = []
        for position, record_index in enumerate(self.packing_plan.packs[item]):
            if position:
                input_parts.append(torch.tensor([self.pad_id], dtype=torch.long))
                label_parts.append(torch.tensor([-100], dtype=torch.long))
            row = self.encoded_records[record_index]
            length = self._unpadded_length(row)
            input_parts.append(row["input_ids"][:length])
            label_parts.append(row["labels"][:length])
        input_ids = torch.cat(input_parts)
        labels = torch.cat(label_parts)
        padding = self.seq_len - len(input_ids)
        if padding < 0:
            raise RuntimeError("packed SFT JSONL sequence exceeds seq_len")
        if padding:
            input_ids = torch.nn.functional.pad(input_ids, (0, padding), value=self.pad_id)
            labels = torch.nn.functional.pad(labels, (0, padding), value=-100)
        return {"input_ids": input_ids, "labels": labels}

    def __iter__(self):
        item_count = self.training_items
        items = list(range(item_count))
        steps_per_epoch = (
            math.ceil(math.ceil(item_count / self.world_size) / self.epoch_batch_size)
            if self.epoch_batch_size is not None
            else 0
        )
        start_epoch, start_batch = (
            divmod(self.start_step, steps_per_epoch) if steps_per_epoch else (0, 0)
        )
        if self.epochs is not None and start_epoch >= self.epochs:
            return
        epoch = start_epoch
        while True:
            rng = random.Random(self.seed + epoch)
            rng.shuffle(items)
            rank_items = items[self.rank :: self.world_size]
            if self.epoch_batch_size is not None:
                rank_items = pad_epoch_items(
                    rank_items,
                    all_items=items,
                    rank=self.rank,
                    world_size=self.world_size,
                    batch_size=self.epoch_batch_size,
                )
                if epoch == start_epoch and start_batch:
                    rank_items = rank_items[start_batch * self.epoch_batch_size :]
            for item in rank_items:
                yield self._tensor_item(item)
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
    unexpected_missing = [key for key in missing if not key.endswith(tolerated_suffixes)]
    if unexpected_missing or unexpected:
        raise RuntimeError(
            "Checkpoint mismatch: " f"missing={unexpected_missing}, unexpected={list(unexpected)}"
        )


class SFTBinDataset(IterableDataset):
    """Stream independent, indexed SFT examples from memory-mapped shards."""

    packing_contract = "supervised-finetuning"

    def __init__(
        self,
        root: str | Path,
        seq_len: int,
        seed: int,
        rank: int,
        world_size: int,
        repeat: bool = True,
        epochs: int | None = None,
        epoch_batch_size: int | None = None,
        start_step: int = 0,
        curriculum_config: str | Path | None = None,
        curriculum_stage: str | None = None,
        reasoning_envelope: bool = False,
        reasoning_tokenizer: Tokenizer | None = None,
        pack_sequences: bool = True,
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
        supported_formats = {
            "complexity-sft-token-shard-v1",
            "complexity-sft-token-shard-v2",
        }
        if self.metadata.get("format") not in supported_formats:
            raise ValueError(f"Unsupported SFT shard format: {self.metadata.get('format')}")
        validate_shard_supervision(self.metadata)
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
        loss_metadata_path = root / "loss_metadata.jsonl"
        if loss_metadata_path.is_file():
            with loss_metadata_path.open(encoding="utf-8") as handle:
                loss_metadata = {
                    str(row["example_id"]): row
                    for row in (json.loads(line) for line in handle if line.strip())
                }
            example_ids = {str(example["example_id"]) for example in self.examples}
            if set(loss_metadata) != example_ids:
                raise ValueError(
                    "SFT loss metadata must exactly match the example index: "
                    f"missing={len(example_ids - set(loss_metadata))} "
                    f"unused={len(set(loss_metadata) - example_ids)}"
                )
            self.examples = [
                {**loss_metadata[str(example["example_id"])], **example}
                for example in self.examples
            ]
        if (curriculum_config is None) != (curriculum_stage is None):
            raise ValueError("curriculum_config and curriculum_stage must be provided together")
        self.curriculum_stage = None
        self.loss_stage = None
        self.loss_task_targets: dict[str, float] = {}
        self.loss_groups = ()
        if curriculum_config is not None and curriculum_stage is not None:
            curriculum = load_curriculum(curriculum_config)
            stage = curriculum.stage(curriculum_stage)
            projected_metadata = load_projected_metadata(dataset_root / "projected.parquet")
            self.examples = select_stage_examples(
                self.examples,
                curriculum,
                curriculum_stage,
                projected_metadata,
            )
            self.curriculum_stage = curriculum_stage
            self.loss_stage = stage
            self.loss_task_targets = dict(stage.loss_task_targets)
            self.loss_groups = stage.loss_groups
        self.seq_len = seq_len
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.repeat = repeat
        self.epochs = epochs
        self.epoch_batch_size = epoch_batch_size
        self.start_step = start_step
        self.pad_id = int(self.metadata["eos_token_id"])
        self.reasoning_prefix_ids: dict[str, list[int]] = {}
        self.reasoning_suffix_ids: list[int] = []
        if reasoning_envelope:
            if reasoning_tokenizer is None:
                raise ValueError("reasoning envelope requires a tokenizer")
            self.reasoning_prefix_ids = {
                task: reasoning_tokenizer.encode(
                    f"<think>\n{plan}\n</think>\n<final>\n",
                    add_special_tokens=False,
                )
                for task, plan in REASONING_ENVELOPE_PLANS.items()
            }
            self.reasoning_suffix_ids = reasoning_tokenizer.encode(
                "\n</final>",
                add_special_tokens=False,
            )
        self.pack_sequences = bool(pack_sequences)
        if self.pack_sequences and reasoning_envelope:
            raise ValueError(
                "sequence packing is not yet compatible with --reasoning-envelope"
            )
        self.task_loss_weights: dict[str, float] = {}
        self.loss_target_audit: dict[str, Any] | None = None
        if self.loss_task_targets or self.loss_groups:
            self._configure_task_loss_weights()
        self.packing_plan: SequencePackingPlan | None = (
            self._build_packing_plan() if self.pack_sequences else None
        )
        self.packed_examples = (
            self.packing_plan.packs if self.packing_plan is not None else None
        )

    @property
    def training_items(self) -> int:
        return (
            len(self.packed_examples)
            if self.packed_examples is not None
            else len(self.examples)
        )

    def _build_packing_plan(self) -> SequencePackingPlan:
        """Pack complete examples into fixed windows without dropping rows.

        Packing is computed once from source order, then packs are shuffled per
        epoch. A separator token is inserted between examples so the next
        prompt is preceded by the checkpoint's EOS token. No example is split.
        """

        return pack_example_lengths(
            (
                min(self.seq_len, int(example["num_tokens"]))
                for example in self.examples
            ),
            sequence_length=self.seq_len,
            separator_tokens=1,
        )

    def _visible_supervised_tokens(self, example: dict[str, Any]) -> int:
        """Count labels that training will actually see for one example."""

        start = int(example["offset"])
        length = int(example["num_tokens"])
        end = start + length
        if start < 0 or length <= 0 or end > len(self.labels):
            raise ValueError(f"Invalid SFT example bounds: {example}")
        labels = np.asarray(self.labels[start:end], dtype=np.int64)
        task = str(example.get("task", ""))
        if task in self.reasoning_prefix_ids:
            input_ids = np.asarray(self.input_ids[start:end], dtype=np.int64)
            _, labels = apply_reasoning_envelope(
                input_ids,
                labels,
                prefix_ids=self.reasoning_prefix_ids[task],
                suffix_ids=self.reasoning_suffix_ids,
                eos_token_id=self.pad_id,
                seq_len=self.seq_len,
            )
        if len(labels) > self.seq_len:
            labels = labels[-self.seq_len :]
        return int(np.count_nonzero(labels != -100))

    def _configure_task_loss_weights(self) -> None:
        token_counts: Counter[str] = Counter()
        for example in self.examples:
            token_counts[loss_weight_key(self.loss_stage, example)] += (
                self._visible_supervised_tokens(example)
            )
        if self.loss_stage is None:
            raise ValueError("loss balancing requires a configured curriculum stage")
        mix = derive_task_loss_weights(self.loss_stage, token_counts)
        if mix["missing_targets"] or mix["unused_targets"]:
            raise ValueError(
                "loss targets must exactly match selected shard tasks: "
                f"missing={mix['missing_targets']} unused={mix['unused_targets']}"
            )
        if not mix["weights_within_cap"]:
            raise ValueError(
                "task loss weights exceed max_task_loss_weight: "
                f"{mix['overweight_tasks']}"
            )
        if any(count <= 0 for count in token_counts.values()):
            raise ValueError("every loss-weighted task must have supervised tokens")
        total = sum(token_counts.values())
        self.task_loss_weights = mix["task_loss_weights"]
        self.loss_target_audit = {
            "examples": len(self.examples),
            "visible_supervised_tokens": total,
            "task_visible_supervised_tokens": dict(sorted(token_counts.items())),
            **mix,
        }

    def _unpadded_example(
        self, example: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, float]:
        start = int(example["offset"])
        length = int(example["num_tokens"])
        end = start + length
        if start < 0 or length <= 0 or end > len(self.input_ids):
            raise ValueError(f"Invalid SFT example bounds: {example}")
        input_ids = np.asarray(self.input_ids[start:end], dtype=np.int64)
        labels = np.asarray(self.labels[start:end], dtype=np.int64)
        task = str(example.get("task", ""))
        loss_key = loss_weight_key(self.loss_stage, example) if self.loss_stage else task
        if task in self.reasoning_prefix_ids:
            input_ids, labels = apply_reasoning_envelope(
                input_ids,
                labels,
                prefix_ids=self.reasoning_prefix_ids[task],
                suffix_ids=self.reasoning_suffix_ids,
                eos_token_id=self.pad_id,
                seq_len=self.seq_len,
            )
            length = len(input_ids)
        if length > self.seq_len:
            # Retain the final assistant response and its EOS target.
            input_ids = input_ids[-self.seq_len :]
            labels = labels[-self.seq_len :]
        if not np.any(labels != -100):
            raise ValueError(
                f"SFT example has no supervised assistant tokens: {example['example_id']}"
            )
        return (
            input_ids,
            labels,
            float(self.task_loss_weights.get(loss_key, 1.0)),
        )

    def _tensor_example(self, example: dict[str, Any]) -> dict[str, torch.Tensor]:
        input_ids, labels, loss_weight = self._unpadded_example(example)
        length = len(input_ids)
        if length < self.seq_len:
            padding = self.seq_len - length
            input_ids = np.pad(input_ids, (0, padding), constant_values=self.pad_id)
            labels = np.pad(labels, (0, padding), constant_values=-100)
        return {
            "input_ids": torch.from_numpy(input_ids.copy()),
            "labels": torch.from_numpy(labels.copy()),
            "loss_weight": torch.tensor(loss_weight, dtype=torch.float32),
        }

    def _tensor_pack(self, pack: tuple[int, ...]) -> dict[str, torch.Tensor]:
        input_parts: list[np.ndarray] = []
        label_parts: list[np.ndarray] = []
        weight_parts: list[np.ndarray] = []
        for position, example_index in enumerate(pack):
            if position:
                input_parts.append(np.asarray([self.pad_id], dtype=np.int64))
                label_parts.append(np.asarray([-100], dtype=np.int64))
                weight_parts.append(np.asarray([0.0], dtype=np.float32))
            input_ids, labels, loss_weight = self._unpadded_example(
                self.examples[example_index]
            )
            input_parts.append(input_ids)
            label_parts.append(labels)
            weight_parts.append(
                np.full(len(labels), loss_weight, dtype=np.float32)
            )
        input_ids = np.concatenate(input_parts)
        labels = np.concatenate(label_parts)
        token_loss_weights = np.concatenate(weight_parts)
        if len(input_ids) > self.seq_len:
            raise RuntimeError("packed SFT sequence exceeds seq_len")
        if len(input_ids) < self.seq_len:
            padding = self.seq_len - len(input_ids)
            input_ids = np.pad(input_ids, (0, padding), constant_values=self.pad_id)
            labels = np.pad(labels, (0, padding), constant_values=-100)
            token_loss_weights = np.pad(
                token_loss_weights, (0, padding), constant_values=0.0
            )
        return {
            "input_ids": torch.from_numpy(input_ids.copy()),
            "labels": torch.from_numpy(labels.copy()),
            "loss_weight": torch.from_numpy(token_loss_weights.copy()),
        }

    def __iter__(self):
        steps_per_epoch = (
            math.ceil(math.ceil(self.training_items / self.world_size) / self.epoch_batch_size)
            if self.epoch_batch_size is not None
            else 0
        )
        start_epoch, start_batch = (
            divmod(self.start_step, steps_per_epoch) if steps_per_epoch else (0, 0)
        )
        if self.epochs is not None and start_epoch >= self.epochs:
            return
        epoch = start_epoch
        item_count = self.training_items
        indices = list(range(item_count))
        while True:
            random.Random(self.seed + epoch).shuffle(indices)
            rank_indices = indices[self.rank :: self.world_size]
            if self.epoch_batch_size is not None:
                rank_indices = pad_epoch_items(
                    rank_indices,
                    all_items=indices,
                    rank=self.rank,
                    world_size=self.world_size,
                    batch_size=self.epoch_batch_size,
                )
                if epoch == start_epoch and start_batch:
                    rank_indices = rank_indices[start_batch * self.epoch_batch_size :]
            for example_index in rank_indices:
                if self.packed_examples is None:
                    yield self._tensor_example(self.examples[example_index])
                else:
                    yield self._tensor_pack(self.packed_examples[example_index])
            if not self.repeat:
                break
            epoch += 1
            if self.epochs is not None and epoch >= self.epochs:
                break


def resolve_sft_bin_evaluation_partitions(
    root: str | Path,
) -> tuple[Path | None, Path | None]:
    """Return matched-distribution and separately-authored eval partitions.

    New corpus packages expose ``diagnostic`` for source-separated examples
    drawn from the same card distribution as training and ``eval`` for the
    smaller, separately-authored natural set.  Legacy packages with only an
    ``eval`` partition retain their historical single-evaluation behavior.
    """

    root = Path(root)
    if (root / "sft.idx.json").exists():
        return (root, None) if root.name in {"diagnostic", "eval"} else (None, None)

    diagnostic = root / "diagnostic"
    natural = root / "eval"
    diagnostic_exists = (diagnostic / "sft.idx.json").exists()
    natural_exists = (natural / "sft.idx.json").exists()
    if diagnostic_exists:
        return diagnostic, natural if natural_exists else None
    if natural_exists:
        return natural, None
    return None, None


def validate_evaluation_sample_fraction(
    train_examples: int,
    eval_examples: int,
    *,
    minimum_fraction: float,
    partition_name: str,
) -> None:
    """Reject statistical evaluation sets too small to represent the train set."""

    if not 0.0 <= minimum_fraction <= 1.0:
        raise ValueError("minimum evaluation fraction must be between 0 and 1")
    if minimum_fraction == 0.0:
        return
    fraction = eval_examples / max(1, train_examples)
    if fraction < minimum_fraction:
        raise ValueError(
            f"{partition_name} evaluation is only {fraction:.2%} of training "
            f"({eval_examples:,}/{train_examples:,}); required minimum is "
            f"{minimum_fraction:.2%}. Use a qualitative regression panel instead."
        )


def validate_sft_release_manifest(
    root: str | Path,
    *,
    reasoning_envelope: bool,
) -> dict[str, Any]:
    """Require an audited release and reject runtime double wrapping."""

    root = Path(root)
    dataset_root = root.parent if (root / "sft.idx.json").exists() else root
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"release-ready SFT requires a manifest: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("quality_status") != "passed":
        raise ValueError(
            "SFT release quality_status must be passed, got "
            f"{manifest.get('quality_status')!r}"
        )
    if manifest.get("release_quality", {}).get("ready") is not True:
        raise ValueError("SFT release_quality.ready must be true")
    envelope_version = manifest.get("release_quality", {}).get(
        "reasoning_envelope_version"
    )
    if envelope_version and reasoning_envelope:
        raise ValueError(
            "SFT targets already contain reasoning envelopes; disable the "
            "runtime --reasoning-envelope injector"
        )
    return manifest


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
    groups = {
        "base_decay": [],
        "base_no_decay": [],
        "expert_decay": [],
        "expert_no_decay": [],
    }
    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue
        expert = (
            ("parametrizations.expert_" in name and ".lora_" in name)
            or ".engine.expert_" in name
        )
        no_decay = param.ndim < 2 or "bias" in name or "norm" in name
        prefix = "expert" if expert else "base"
        groups[f"{prefix}_{'no_decay' if no_decay else 'decay'}"].append(param)
    optimizer_groups = []
    for name, parameters in groups.items():
        if not parameters:
            continue
        expert = name.startswith("expert_")
        optimizer_groups.append(
            {
                "name": name,
                "params": parameters,
                "weight_decay": 0.0 if name.endswith("no_decay") else args.weight_decay,
                "lr": args.lr * (args.expert_lr_multiplier if expert else 1.0),
            }
        )
    return torch.optim.AdamW(
        optimizer_groups,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )


def configure_sft_parameters(args, raw_model) -> dict[str, int | bool | str]:
    """Select explicit full-parameter SFT or the default LoRA adaptation."""

    if args.source_stage is None:
        raise ValueError(
            f"SFT requires --source-stage {REFINEMENT_STAGE}; direct "
            "pretraining -> SFT is forbidden"
        )
    validate_training_stage_transition(
        TEXT_MODEL_FAMILY,
        args.source_stage,
        SUPERVISED_FINETUNING_STAGE,
    )

    if args.full_parameter:
        validate_full_parameter_finetuning(
            TEXT_SUPERVISED_FINETUNING,
            source_stage=args.source_stage,
        )
        raw_model.requires_grad_(True)
        total = sum(parameter.numel() for parameter in raw_model.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in raw_model.parameters()
            if parameter.requires_grad
        )
        return {
            "mode": "full-parameter",
            "modules": sum(1 for _ in raw_model.modules()),
            "linear_modules": sum(
                1 for module in raw_model.modules() if isinstance(module, torch.nn.Linear)
            ),
            "expert_tensors": sum(
                1 for name, _ in raw_model.named_parameters() if ".engine.expert_" in name
            ),
            "trainable": trainable,
            "total": total,
            "frozen": total - trainable,
            "token_io_frozen": False,
        }

    targets = tuple(name.strip() for name in args.lora_targets.split(",") if name.strip())
    stats: dict[str, int | bool | str] = apply_lora(
        raw_model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        targets=targets,
    )
    stats["mode"] = "lora"
    stats["token_io_frozen"] = True
    return stats


RESUME_ARGUMENTS = (
    "jsonl",
    "sft_bin",
    "curriculum_config",
    "curriculum_stage",
    "epochs",
    "batch_size",
    "seq_len",
    "pack_sequences",
    "save_every_epoch",
    "eval_every_epoch",
    "lr",
    "weight_decay",
    "beta1",
    "beta2",
    "warmup_ratio",
    "bf16",
    "use_custom_kernels",
    "grad_ckpt",
    "loss_chunk_tokens",
    "sft_fp32_loss",
    "full_parameter",
    "source_stage",
    "reset_lr_each_epoch",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "lora_targets",
    "expert_lr_multiplier",
    "reasoning_envelope",
    "seed",
)


def validate_resume_state(args, state: dict[str, Any], world_size: int) -> None:
    missing = [key for key in ("optimizer", "scheduler", "step") if key not in state]
    if missing:
        raise ValueError(
            "SFT resume checkpoint is model-only or incomplete; missing " + ", ".join(missing)
        )
    saved_world_size = int(state.get("world_size", world_size))
    if saved_world_size != world_size:
        raise ValueError(
            f"SFT exact resume requires the same world size: "
            f"checkpoint={saved_world_size}, current={world_size}"
        )
    saved_args = state.get("args", {})
    mismatches = []
    for name in RESUME_ARGUMENTS:
        if name in saved_args and saved_args[name] != getattr(args, name):
            mismatches.append(
                f"{name}: checkpoint={saved_args[name]!r}, current={getattr(args, name)!r}"
            )
    if mismatches:
        raise ValueError("SFT resume arguments changed: " + "; ".join(mismatches))
    if args.steps > 0 and int(state["step"]) >= args.steps:
        raise ValueError(
            f"resume step {state['step']} must be smaller than target steps {args.steps}"
        )


def capture_rng_state(device: torch.device) -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device).cpu()
    return state


def gather_rng_states(device: torch.device, distributed: bool) -> list[dict[str, Any]]:
    local_state = capture_rng_state(device)
    if not distributed:
        return [local_state]
    states: list[dict[str, Any] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(states, local_state)
    return [state for state in states if state is not None]


def restore_rng_state(
    states: list[dict[str, Any]],
    *,
    rank: int,
    device: torch.device,
) -> None:
    if rank >= len(states):
        raise ValueError(f"SFT checkpoint has {len(states)} RNG states but current rank is {rank}")
    state = states[rank]
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device)


def distributed_barrier(distributed: bool) -> None:
    if not distributed:
        return
    device_ids = [torch.cuda.current_device()] if torch.cuda.is_available() else None
    dist.barrier(device_ids=device_ids)


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


def validation_baseline(initial_eval_loss: float | None) -> float:
    """Make the stage-entry checkpoint the validation baseline when measured."""

    return math.inf if initial_eval_loss is None else float(initial_eval_loss)


def early_stopping_is_eligible(
    step: int,
    *,
    steps_per_epoch: int,
    minimum_epochs: int,
) -> bool:
    """Do not select or stop before every selected example was seen once."""

    if steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be positive")
    if minimum_epochs < 0:
        raise ValueError("minimum_epochs cannot be negative")
    return step >= steps_per_epoch * minimum_epochs


def lr_schedule_horizon(
    step_limit: int,
    steps_per_epoch: int,
    *,
    reset_each_epoch: bool = True,
) -> int:
    """Resolve a per-epoch or whole-run learning-rate schedule horizon."""

    if step_limit < 1:
        raise ValueError("step_limit must be positive")
    if steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be positive")
    return min(step_limit, steps_per_epoch) if reset_each_epoch else step_limit


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
    example_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Chunked CE for SFT, computing logits/CE in fp32 for stability."""

    flat_hidden = hidden_states.reshape(-1, hidden_states.size(-1))
    flat_labels = labels.reshape(-1)
    valid = flat_labels != -100
    denom = valid.sum().clamp_min(1).to(dtype=torch.float32)
    flat_weights = None
    if example_weights is not None:
        if example_weights.ndim == 1 and example_weights.shape[0] == labels.shape[0]:
            flat_weights = (
                example_weights.to(device=hidden_states.device, dtype=torch.float32)
                .view(-1, 1)
                .expand_as(labels)
                .reshape(-1)
            )
        elif example_weights.shape == labels.shape:
            flat_weights = example_weights.to(
                device=hidden_states.device, dtype=torch.float32
            ).reshape(-1)
        else:
            raise ValueError(
                "example_weights must contain one scalar per batch example "
                "or one scalar per token"
            )
        denom = flat_weights[valid].sum().clamp_min(1e-12)
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
        losses = F.cross_entropy(
            logits,
            labels_chunk,
            reduction="none",
        )
        if flat_weights is not None:
            losses = losses * flat_weights[start:end][valid_chunk]
        total = total + losses.sum()
    return total / denom


def compute_sft_loss(
    hidden_states: torch.Tensor,
    output_weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    fp32_loss: bool,
    liger_loss: bool,
    chunk_tokens: int,
    example_weights: torch.Tensor | None = None,
    sync_metrics: bool = True,
) -> tuple[torch.Tensor, float]:
    """Dispatch the SFT loss through one explicit, testable backend policy."""

    if liger_loss:
        if example_weights is not None:
            raise ValueError("Liger SFT loss does not support per-example loss weights")
        loss, metrics = fused_linear_causal_lm_loss(
            hidden_states,
            output_weight,
            labels,
            use_liger=True,
            sync_metrics=sync_metrics,
        )
        return loss, float(metrics.ce)
    if fp32_loss:
        loss = sft_loss_from_hidden(
            hidden_states,
            output_weight,
            labels,
            chunk_tokens=chunk_tokens,
            example_weights=example_weights,
        )
        return loss, float(loss.detach().item()) if sync_metrics else float("nan")
    loss, metrics = causal_lm_loss_from_hidden(
        hidden_states,
        output_weight,
        labels,
        chunk_tokens=chunk_tokens,
        sync_metrics=sync_metrics,
    )
    return loss, float(metrics.ce)


@torch.no_grad()
def evaluate_sft(
    model,
    raw_model,
    loader,
    *,
    device: torch.device,
    amp_dtype,
    fp32_loss: bool,
    liger_loss: bool,
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
        # The shard collator always emits ``loss_weight`` (unit weights for an
        # ordinary unweighted run, plus zeroes on labels already masked with
        # -100). Only the explicit FP32 weighted-loss backend consumes it.
        # Passing these neutral tensors to Liger would incorrectly classify a
        # normal packed batch as task-weighted SFT.
        example_weights = batch.get("loss_weight") if fp32_loss else None
        if example_weights is not None:
            example_weights = example_weights.to(device, non_blocking=True)
        supervised = int((labels != -100).sum().item())
        if supervised == 0:
            continue
        with autocast(device, dtype=amp_dtype, enabled=amp_dtype is not None):
            outputs = model(input_ids, return_logits=False)
            loss, _ = compute_sft_loss(
                outputs["last_hidden_state"],
                raw_model.embed_tokens.weight,
                labels,
                fp32_loss=fp32_loss,
                liger_loss=liger_loss,
                chunk_tokens=chunk_tokens,
                example_weights=example_weights,
                sync_metrics=False,
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
    best_eval_loss: float | None = None,
    evaluations_without_improvement: int = 0,
):
    if args.save_steps <= 0 and not force:
        distributed_barrier(distributed)
        return
    device = next(raw_model.parameters()).device
    distributed_rng_states = gather_rng_states(device, distributed)
    if not is_main:
        distributed_barrier(distributed)
        return
    adapters = adapter_state_dict(raw_model)
    checkpoint_state = {
        "step": step,
        "model": (
            merged_model_state_dict(raw_model)
            if adapters
            else {k: v.detach().cpu() for k, v in raw_model.state_dict().items()}
        ),
        "config": config.to_dict(),
        "args": vars(args),
        "sft_source_checkpoint": source_checkpoint,
        "chat_template": chat_template,
        "backend": backend_metadata(kernel_policy=getattr(args, "use_custom_kernels", "auto")),
        "world_size": dist.get_world_size() if distributed else 1,
        "distributed_rng_states": distributed_rng_states,
        "best_eval_loss": best_eval_loss,
        "evaluations_without_improvement": evaluations_without_improvement,
    }
    if adapters:
        checkpoint_state["lora_adapter"] = adapters
        checkpoint_state["lora_config"] = LoRAConfig(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            targets=tuple(args.lora_targets.split(",")),
        ).to_dict()
    if eval_loss is not None:
        checkpoint_state["sft_matched_eval_loss"] = float(eval_loss)
        # Kept for compatibility with earlier exported SFT checkpoints.
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
    distributed_barrier(distributed)
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
        best_eval_loss=eval_loss,
        evaluations_without_improvement=0,
    )
    if is_main and checkpoint_dir is not None:
        metadata = {
            "step": step,
            "selection_metric": "matched_eval_loss",
            "matched_eval_loss": eval_loss,
            "matched_eval_tokens": eval_tokens,
            "checkpoint": str(checkpoint_dir),
        }
        (Path(args.save_dir) / "best.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def positive_lora_rank(value: str) -> int:
    rank = int(value)
    if rank <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return rank


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SFT the 500M TR-HASH 32k checkpoint")
    parser.add_argument(
        "--checkpoint", required=True, help="Checkpoint dir or checkpoint.pt to fine-tune"
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Exactly resume an SFT checkpoint, including optimizer, scheduler, data cursor, and RNG.",
    )
    parser.add_argument("--tokenizer", default="./tokenizer-32k")
    dataset = parser.add_mutually_exclusive_group()
    dataset.add_argument("--jsonl", default=None, help="Inspectable SFT JSONL.")
    dataset.add_argument(
        "--sft-bin",
        default=None,
        help="Tokenized SFT root containing train/input_ids.bin and train/labels.bin.",
    )
    parser.add_argument(
        "--require-release-ready",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Require quality_status=passed and release_quality.ready=true in "
            "the tokenized SFT manifest before training."
        ),
    )
    parser.add_argument(
        "--eval-jsonl",
        default=None,
        help="Finite held-out JSONL evaluated during JSONL-based SFT.",
    )
    parser.add_argument(
        "--curriculum-config",
        default=None,
        help="Runtime-only curriculum YAML; does not create derivative shards.",
    )
    parser.add_argument(
        "--curriculum-stage",
        default=None,
        help="Stage name selected from --curriculum-config.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Optimizer-step limit; use 0 with --epochs to derive it automatically.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Consume the complete training dataset this many times; 0 keeps the step-limited stream.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument(
        "--pack-sequences",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pack complete SFT examples into seq_len windows with EOS separators. "
            "Preserves assistant labels and per-token curriculum weights."
        ),
    )
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument(
        "--full-parameter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Train every model parameter. This is an explicit alternative to "
            "the default LoRA mode; --lora-* options are ignored."
        ),
    )
    parser.add_argument(
        "--source-stage",
        choices=(REFINEMENT_STAGE, SUPERVISED_FINETUNING_STAGE),
        default=None,
        help=(
            "Lineage stage of --checkpoint. Initial SFT must use refinement; "
            "supervised-finetuning is accepted only for an additional SFT stage."
        ),
    )
    parser.add_argument(
        "--lora-rank",
        type=positive_lora_rank,
        default=16,
        help="LoRA rank used when --full-parameter is disabled.",
    )
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--expert-lr-multiplier",
        type=float,
        default=0.25,
        help="LR multiplier applied only to TR-Hash expert LoRA factors.",
    )
    parser.add_argument(
        "--lora-targets",
        default="q_proj,v_proj,o_proj,shared_gate,shared_up,shared_down",
        help=(
            "Comma-separated linear suffixes or TR-Hash expert tensors "
            "(expert_gate, expert_up, expert_down) adapted by LoRA."
        ),
    )
    parser.add_argument(
        "--reasoning-envelope",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Wrap reasoning-task targets in "
            "<think>...</think><final>...</final> at runtime."
        ),
    )
    parser.add_argument("--min-completion-tokens", type=int, default=32)
    parser.add_argument("--bf16", action="store_true")
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
        help="Compute the tied 32k SFT loss in fp32 chunks for stability.",
    )
    parser.add_argument(
        "--no-sft-fp32-loss",
        dest="sft_fp32_loss",
        action="store_false",
        help="Use the generic causal_lm_loss_from_hidden path.",
    )
    parser.add_argument(
        "--sft-liger-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use Liger fused linear cross-entropy for SFT without materializing "
            "the [batch, sequence, vocabulary] logits tensor. Production CUDA "
            "launchers should combine this with COMPLEXITY_REQUIRE_LIGER=1."
        ),
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
        "--min-eval-fraction",
        type=float,
        default=0.0,
        help=(
            "Reject enabled evaluation partitions smaller than this fraction "
            "of training; 0 preserves legacy behavior."
        ),
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
    parser.add_argument(
        "--early-stopping-min-epochs",
        type=int,
        default=1,
        help=(
            "Minimum complete passes over the selected examples before best "
            "checkpoint selection and early stopping become active."
        ),
    )
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument(
        "--save-every-epoch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Derive periodic checkpoint cadence from the realized epoch size.",
    )
    parser.add_argument(
        "--eval-every-epoch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Derive validation cadence from the realized epoch size.",
    )
    parser.add_argument(
        "--reset-lr-each-epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reset warmup/cosine scheduling at epoch boundaries. Disable for "
            "one continuous schedule over the complete multi-epoch run."
        ),
    )
    parser.add_argument(
        "--save-milestones",
        default="",
        help="Comma-separated exact optimizer steps to checkpoint in addition to --save-steps.",
    )
    parser.add_argument("--save-dir", default="checkpoints/sft-500m-32k-tr")
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument(
        "--save-model-only",
        action="store_true",
        help="Omit optimizer and scheduler state from checkpoints used for evaluation/inference.",
    )
    parser.add_argument("--run-name", default="sft-500m-32k-tr")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--empty-cache-every", type=int, default=50)
    parser.add_argument("--cpu", action="store_true", help="Force CPU for smoke tests")
    return parser


def init_distributed(seed: int):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA. Run single-process for CPU/MPS.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        torch.manual_seed(seed + rank)
        return torch.device("cuda", local_rank), distributed, rank, local_rank, world_size

    device = setup_mps(unlimited_watermark=True, cpu_fallback=True, seed=seed)
    return device, distributed, rank, local_rank, world_size


def main():
    args = build_parser().parse_args()
    if args.steps < 0:
        raise ValueError("--steps cannot be negative")
    if args.steps == 0 and args.epochs <= 0:
        raise ValueError("--steps 0 requires a positive --epochs value")
    if (args.curriculum_config is None) != (args.curriculum_stage is None):
        raise ValueError("--curriculum-config and --curriculum-stage must be provided together")
    if args.curriculum_stage is not None and args.sft_bin is None:
        raise ValueError("runtime curriculum selection requires --sft-bin")
    if args.early_stopping_min_epochs < 0:
        raise ValueError("--early-stopping-min-epochs cannot be negative")
    if not args.full_parameter and args.lora_rank <= 0:
        raise ValueError("--lora-rank must be positive in LoRA mode")
    if args.expert_lr_multiplier <= 0:
        raise ValueError("--expert-lr-multiplier must be positive")
    if not 0.0 <= args.min_eval_fraction <= 1.0:
        raise ValueError("--min-eval-fraction must be between 0 and 1")
    try:
        save_milestones = {
            int(value.strip())
            for value in args.save_milestones.split(",")
            if value.strip()
        }
    except ValueError as error:
        raise ValueError("--save-milestones must contain comma-separated integers") from error
    if any(step <= 0 for step in save_milestones):
        raise ValueError("--save-milestones values must be positive")
    if args.require_release_ready:
        if args.sft_bin is None:
            raise ValueError("--require-release-ready requires --sft-bin")
        validate_sft_release_manifest(
            args.sft_bin,
            reasoning_envelope=args.reasoning_envelope,
        )
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
        True
        if args.use_custom_kernels == "true"
        else False if args.use_custom_kernels == "false" else "auto"
    )
    args.use_custom_kernels = kernel_policy
    configure_torch_acceleration(kernel_policy=kernel_policy, log=is_main)
    if args.sft_liger_loss:
        if args.sft_fp32_loss:
            raise ValueError("--sft-liger-loss requires --no-sft-fp32-loss")
        if device.type != "cuda":
            raise ValueError("--sft-liger-loss requires CUDA")
        log_liger_fused_linear_ce_status(device.type)

    checkpoint_to_load = args.resume or args.checkpoint
    ckpt_dir, state = load_checkpoint_state(checkpoint_to_load, map_location="cpu")
    if args.resume is not None:
        validate_resume_state(args, state, world_size)
    loaded_checkpoint_step = int(state.get("step", 0))
    resume_step = loaded_checkpoint_step if args.resume is not None else 0
    source_checkpoint = str(state.get("sft_source_checkpoint", args.checkpoint))
    resume_optimizer_state = state.get("optimizer") if args.resume is not None else None
    resume_scheduler_state = state.get("scheduler") if args.resume is not None else None
    resume_rng_states = state.get("distributed_rng_states") if args.resume is not None else None
    resume_best_eval_loss = state.get("best_eval_loss") if args.resume is not None else None
    resume_evaluations_without_improvement = int(
        state.get("evaluations_without_improvement", 0) if args.resume is not None else 0
    )
    config = checkpoint_config(state)
    config.use_custom_kernels = kernel_policy
    raw_model = ComplexityModel(config).to(device)
    load_model_state_compat(raw_model, state["model"])
    if args.grad_ckpt:
        raw_model.gradient_checkpointing_enable()
    parameter_stats = configure_sft_parameters(args, raw_model)
    if args.resume is not None:
        saved_adapter = state.get("lora_adapter")
        if args.full_parameter and saved_adapter is not None:
            raise ValueError("full-parameter resume checkpoint unexpectedly contains LoRA state")
        if not args.full_parameter and saved_adapter is None:
            raise ValueError("LoRA resume checkpoint does not contain adapter state")
        if saved_adapter is not None:
            load_adapter_state_dict(raw_model, saved_adapter)
            # ``model`` is deliberately canonical/merged so it works in the
            # normal runtime. Undo that merge before continuing LoRA training.
            unmerge_adapter_from_base(raw_model)
    if parameter_stats["trainable"] == 0:
        raise ValueError("SFT configuration froze every model parameter")
    if args.full_parameter and parameter_stats["trainable"] != parameter_stats["total"]:
        raise ValueError(
            "full-parameter SFT audit failed: "
            f"trainable={parameter_stats['trainable']:,} total={parameter_stats['total']:,}"
        )

    model = raw_model
    if distributed:
        model = DDP(
            raw_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    evaluation_enabled = (
        args.eval_at_start or args.eval_steps > 0 or args.eval_every_epoch
    )
    reasoning_tokenizer = (
        Tokenizer.load(args.tokenizer) if args.reasoning_envelope else None
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
            epoch_batch_size=args.batch_size,
            start_step=resume_step,
            curriculum_config=args.curriculum_config,
            curriculum_stage=args.curriculum_stage,
            reasoning_envelope=args.reasoning_envelope,
            reasoning_tokenizer=reasoning_tokenizer,
            pack_sequences=args.pack_sequences,
        )
        matched_eval_path, natural_eval_path = (
            resolve_sft_bin_evaluation_partitions(args.sft_bin)
            if evaluation_enabled
            else (None, None)
        )
        matched_eval_ds = (
            SFTBinDataset(
                matched_eval_path,
                args.seq_len,
                args.seed,
                rank,
                world_size,
                repeat=False,
                reasoning_envelope=args.reasoning_envelope,
                reasoning_tokenizer=reasoning_tokenizer,
            )
            if matched_eval_path is not None
            else None
        )
        natural_eval_ds = (
            SFTBinDataset(
                natural_eval_path,
                args.seq_len,
                args.seed,
                rank,
                world_size,
                repeat=False,
                reasoning_envelope=args.reasoning_envelope,
                reasoning_tokenizer=reasoning_tokenizer,
            )
            if natural_eval_path is not None
            else None
        )
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
            epoch_batch_size=args.batch_size,
            start_step=resume_step,
            pack_sequences=args.pack_sequences,
        )
        matched_eval_ds = (
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
            if evaluation_enabled and args.eval_jsonl is not None
            else None
        )
        natural_eval_ds = None
    if args.sft_bin is not None and int(train_ds.metadata["vocab_size"]) != int(
        config.vocab_size
    ):
        raise ValueError(
            "SFT tokenizer/model vocabulary mismatch: "
            f"shard={train_ds.metadata['vocab_size']} model={config.vocab_size}"
        )
    if (
        getattr(train_ds, "loss_task_targets", None)
        or getattr(train_ds, "loss_groups", None)
    ) and not args.sft_fp32_loss:
        raise ValueError("task-weighted SFT requires --sft-fp32-loss")
    chat_template = train_ds.chat_template
    for evaluation_dataset in (matched_eval_ds, natural_eval_ds):
        if evaluation_dataset is not None and evaluation_dataset.chat_template != chat_template:
            raise ValueError("Train and eval SFT shards use different chat templates")
    if evaluation_enabled and args.min_eval_fraction > 0:
        train_examples = len(train_ds.examples) if args.sft_bin is not None else len(train_ds.records)
        for partition_name, evaluation_dataset in (
            ("matched", matched_eval_ds),
            ("natural", natural_eval_ds),
        ):
            if evaluation_dataset is None:
                continue
            eval_examples = (
                len(evaluation_dataset.examples)
                if args.sft_bin is not None
                else len(evaluation_dataset.records)
            )
            validate_evaluation_sample_fraction(
                train_examples,
                eval_examples,
                minimum_fraction=args.min_eval_fraction,
                partition_name=partition_name,
            )
    loader_kwargs = {"batch_size": args.batch_size, "pin_memory": False}
    if args.num_workers > 0:
        loader_kwargs.update(num_workers=args.num_workers, persistent_workers=True)
    train_loader = DataLoader(train_ds, **loader_kwargs)
    matched_eval_loader = (
        DataLoader(matched_eval_ds, **loader_kwargs) if matched_eval_ds is not None else None
    )
    natural_eval_loader = (
        DataLoader(natural_eval_ds, **loader_kwargs) if natural_eval_ds is not None else None
    )

    schedule = resolve_epoch_schedule(
        items=train_ds.training_items,
        world_size=world_size,
        batch_size_per_rank=args.batch_size,
        epochs=max(1, args.epochs),
    )
    steps_per_epoch = schedule.steps_per_epoch
    if args.steps == 0:
        args.steps = schedule.total_steps
    if args.eval_every_epoch:
        args.eval_steps = steps_per_epoch
    if args.save_every_epoch:
        args.save_steps = steps_per_epoch
    if args.resume is not None and int(loaded_checkpoint_step) >= args.steps:
        raise ValueError(
            f"resume step {loaded_checkpoint_step} must be smaller than "
            f"target steps {args.steps}"
        )
    minimum_selection_step = steps_per_epoch * args.early_stopping_min_epochs
    if args.early_stopping_patience > 0 and args.steps < minimum_selection_step:
        raise ValueError(
            "early stopping cannot become active before this run ends: "
            f"steps={args.steps}, required={minimum_selection_step} "
            f"({args.early_stopping_min_epochs} complete epoch(s))"
        )

    optimizer = build_optimizer(args, raw_model)
    base_lrs = [group["lr"] for group in optimizer.param_groups]

    schedule_horizon = lr_schedule_horizon(
        args.steps,
        steps_per_epoch,
        reset_each_epoch=args.reset_lr_each_epoch,
    )

    def build_epoch_lr_lambda():
        warmup_steps = max(1, int(schedule_horizon * args.warmup_ratio))
        decay_denom = max(1, schedule_horizon - warmup_steps)

        def lr_lambda(step_in_epoch: int) -> float:
            step_in_epoch += 1
            if step_in_epoch <= warmup_steps:
                return step_in_epoch / warmup_steps
            progress = (step_in_epoch - warmup_steps) / decay_denom
            progress = min(1.0, max(0.0, progress))
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

        return lr_lambda

    def reset_epoch_scheduler() -> torch.optim.lr_scheduler.LambdaLR:
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            build_epoch_lr_lambda(),
            last_epoch=-1,
        )

    scheduler = reset_epoch_scheduler()
    if args.resume is not None:
        optimizer.load_state_dict(resume_optimizer_state)
        scheduler.load_state_dict(resume_scheduler_state)
        if resume_rng_states is None:
            raise ValueError("SFT resume checkpoint does not contain distributed RNG states")
        restore_rng_state(resume_rng_states, rank=rank, device=device)
    del state, resume_optimizer_state, resume_scheduler_state, resume_rng_states
    gc.collect()
    amp_dtype = autocast_dtype(device) if args.bf16 else None

    run_dir = Path("runs") / args.run_name
    csv_file = None
    writer = None
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"SFT source: {ckpt_dir} "
            f"(checkpoint step={loaded_checkpoint_step}, resume step={resume_step})"
        )
        if args.full_parameter:
            logger.info(
                "SFT mode: full-parameter "
                f"trainable={parameter_stats['trainable']:,}/{parameter_stats['total']:,}"
            )
        else:
            logger.info(
                f"SFT mode: LoRA rank={args.lora_rank} alpha={args.lora_alpha:g} "
                f"dropout={args.lora_dropout:g} modules={parameter_stats['modules']} "
                f"trainable={parameter_stats['trainable']:,}"
            )
        if args.resume is not None:
            logger.info(
                "Resumed exactly: optimizer, scheduler, data cursor, and "
                f"rank RNG state at step {resume_step}"
            )
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
            f"TR experts={config.num_experts}, top_k={config.top_k}"
        )
        if args.sft_bin is not None:
            logger.info(
                f"Dataset: SFT bin {train_ds.root} "
                f"({len(train_ds.examples):,} examples, "
                f"stage={train_ds.curriculum_stage or 'full-shard'})"
            )
            if train_ds.loss_target_audit is not None:
                logger.info(
                    "Full-shard task-weighted loss: "
                    + json.dumps(train_ds.loss_target_audit, sort_keys=True)
                )
            if train_ds.packing_plan is not None:
                plan = train_ds.packing_plan
                logger.info(
                    "Sequence packing: "
                    f"{len(train_ds.examples):,} examples -> "
                    f"{train_ds.training_items:,} packed sequences "
                    f"({plan.payload_utilization:.1%} payload utilization, "
                    f"{plan.compression_ratio:.2f}x fewer sequences)"
                )
            if evaluation_enabled:
                logger.info(
                    "Coverage: "
                    f"{steps_per_epoch:,} steps/epoch; best-checkpoint selection "
                    f"and early stopping start at step {minimum_selection_step:,}"
                )
            else:
                logger.info(
                    f"Coverage: {steps_per_epoch:,} steps/epoch; validation disabled"
                )
            if train_ds.metadata["supervised_tokens"] < 3_000_000:
                logger.warning(
                    "Training shard contains fewer than 3,000,000 supervised "
                    "tokens; use held-out early stopping and treat the run as "
                    "a small-data adaptation."
                )
            if matched_eval_ds is not None:
                logger.info(
                    "Matched evaluation: "
                    f"{matched_eval_ds.root} ({len(matched_eval_ds.examples):,} examples)"
                )
            if natural_eval_ds is not None:
                logger.info(
                    "Natural-gold evaluation: "
                    f"{natural_eval_ds.root} ({len(natural_eval_ds.examples):,} examples)"
                )
            if natural_eval_ds is not None and len(natural_eval_ds.examples) < 500:
                logger.warning(
                    "Natural-gold SFT evaluation contains fewer than 500 examples; "
                    "report it separately as a transfer diagnostic, not as the "
                    "checkpoint-selection metric."
                )
        elif args.jsonl is None:
            logger.info("Dataset: built-in toy SFT records")
        else:
            logger.info(f"Dataset: {args.jsonl} ({len(train_ds.records)} records)")
            if matched_eval_ds is not None:
                logger.info(
                    f"Evaluation: {args.eval_jsonl} "
                    f"({len(matched_eval_ds.records)} held-out records)"
                )
        logger.info(f"Chat template: {chat_template['id']}")
        if args.reasoning_envelope:
            logger.info(
                "Reasoning envelope: <think>/<final> enabled for "
                + ", ".join(sorted(REASONING_ENVELOPE_PLANS))
            )
        metrics_path = run_dir / "metrics.csv"
        append_metrics = args.resume is not None and metrics_path.exists()
        csv_file = metrics_path.open("a" if append_metrics else "w", newline="")
        writer = csv.writer(csv_file)
        if not append_metrics:
            writer.writerow(
                [
                    "step",
                    "train_loss",
                    "train_ppl",
                    "matched_eval_loss",
                    "matched_eval_ppl",
                    "lr",
                    "expert_lr",
                    "tok_s",
                    "supervised_tokens",
                    "min_label",
                    "max_label",
                    "bad_labels",
                    "matched_eval_tokens",
                    "natural_eval_loss",
                    "natural_eval_ppl",
                    "natural_eval_tokens",
                ]
            )
        csv_file.flush()

    best_eval_loss = validation_baseline(resume_best_eval_loss)
    evaluations_without_improvement = resume_evaluations_without_improvement
    if (
        resume_step == 0
        and matched_eval_loader is not None
        and args.eval_steps > 0
        and args.eval_at_start
    ):
        initial_eval_loss, initial_eval_tokens = evaluate_sft(
            model,
            raw_model,
            matched_eval_loader,
            device=device,
            amp_dtype=amp_dtype,
            fp32_loss=args.sft_fp32_loss,
            liger_loss=args.sft_liger_loss,
            chunk_tokens=args.loss_chunk_tokens,
            distributed=distributed,
            max_batches=args.eval_batches,
        )
        initial_natural_eval_loss = None
        initial_natural_eval_tokens = 0
        if natural_eval_loader is not None:
            initial_natural_eval_loss, initial_natural_eval_tokens = evaluate_sft(
                model,
                raw_model,
                natural_eval_loader,
                device=device,
                amp_dtype=amp_dtype,
                fp32_loss=args.sft_fp32_loss,
                liger_loss=args.sft_liger_loss,
                chunk_tokens=args.loss_chunk_tokens,
                distributed=distributed,
                max_batches=args.eval_batches,
            )
        # The source checkpoint is a real candidate for this stage.  Recording
        # its held-out loss prevents the first completed epoch from becoming
        # "best" merely because no trained checkpoint has been considered yet.
        best_eval_loss = validation_baseline(initial_eval_loss)
        if is_main:
            logger.info(
                f"SFT matched eval step=0: loss={initial_eval_loss:.6f} "
                f"ppl={math.exp(min(initial_eval_loss, 20)):.2f} "
                f"tokens={initial_eval_tokens:,}"
            )
            if initial_natural_eval_loss is not None:
                logger.info(
                    "SFT natural-gold eval step=0: "
                    f"loss={initial_natural_eval_loss:.6f} "
                    f"ppl={math.exp(min(initial_natural_eval_loss, 20)):.2f} "
                    f"tokens={initial_natural_eval_tokens:,}"
                )
            writer.writerow(
                [
                    0,
                    "",
                    "",
                    f"{initial_eval_loss:.6f}",
                    f"{math.exp(min(initial_eval_loss, 20)):.2f}",
                    f"{optimizer.param_groups[0]['lr']:.6e}",
                    "",
                    "",
                    "",
                    "",
                    "",
                    0,
                    initial_eval_tokens,
                    (
                        ""
                        if initial_natural_eval_loss is None
                        else f"{initial_natural_eval_loss:.6f}"
                    ),
                    (
                        ""
                        if initial_natural_eval_loss is None
                        else f"{math.exp(min(initial_natural_eval_loss, 20)):.2f}"
                    ),
                    initial_natural_eval_tokens,
                ]
            )
            csv_file.flush()

    model.train()
    pbar = (
        tqdm(
            total=args.steps,
            initial=resume_step,
            desc="SFT 500M 32k TR",
            unit="step",
            dynamic_ncols=True,
        )
        if is_main
        else None
    )
    t_log = time.perf_counter()
    tokens_since_log = 0
    last_step = resume_step

    current_epoch = max(0, (resume_step - 1) // steps_per_epoch)

    for local_step, batch in enumerate(train_loader, start=1):
        step = resume_step + local_step
        if step > args.steps:
            break
        epoch_idx = (step - 1) // steps_per_epoch
        if epoch_idx > current_epoch:
            current_epoch = epoch_idx
            if args.reset_lr_each_epoch:
                for group, base_lr in zip(optimizer.param_groups, base_lrs):
                    group["lr"] = base_lr
                scheduler = reset_epoch_scheduler()
            if is_main and args.reset_lr_each_epoch:
                logger.info(
                    f"Resetting LR scheduler for epoch {current_epoch + 1} "
                    f"({epoch_idx + 1}/{max(args.epochs or 1, epoch_idx + 1)})"
                )
        last_step = step
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        # Task weighting is supported only by the explicit FP32 SFT loss. The
        # Liger path is unweighted and must not receive the collator's neutral
        # unit/ignored-position weights.
        example_weights = (
            batch.get("loss_weight") if args.sft_fp32_loss else None
        )
        if example_weights is not None:
            example_weights = example_weights.to(device, non_blocking=True)
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
            loss, metrics_ce = compute_sft_loss(
                outputs["last_hidden_state"],
                raw_model.embed_tokens.weight,
                labels,
                fp32_loss=args.sft_fp32_loss,
                liger_loss=args.sft_liger_loss,
                chunk_tokens=args.loss_chunk_tokens,
                example_weights=example_weights,
                sync_metrics=True,
            )
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
            matched_eval_loader is not None
            and args.eval_steps > 0
            and (step % args.eval_steps == 0 or step == args.steps)
        )
        should_log = local_step == 1 or step % args.log_steps == 0 or should_eval
        eval_loss = None
        eval_tokens = 0
        natural_eval_loss = None
        natural_eval_tokens = 0
        stop_training = False
        if should_eval:
            eval_loss, eval_tokens = evaluate_sft(
                model,
                raw_model,
                matched_eval_loader,
                device=device,
                amp_dtype=amp_dtype,
                fp32_loss=args.sft_fp32_loss,
                liger_loss=args.sft_liger_loss,
                chunk_tokens=args.loss_chunk_tokens,
                distributed=distributed,
                max_batches=args.eval_batches,
            )
            if is_main:
                logger.info(
                    f"SFT matched eval step={step}: loss={eval_loss:.6f} "
                    f"ppl={math.exp(min(eval_loss, 20)):.2f} tokens={eval_tokens:,}"
                )
            if natural_eval_loader is not None:
                natural_eval_loss, natural_eval_tokens = evaluate_sft(
                    model,
                    raw_model,
                    natural_eval_loader,
                    device=device,
                    amp_dtype=amp_dtype,
                    fp32_loss=args.sft_fp32_loss,
                    liger_loss=args.sft_liger_loss,
                    chunk_tokens=args.loss_chunk_tokens,
                    distributed=distributed,
                    max_batches=args.eval_batches,
                )
                if is_main:
                    logger.info(
                        f"SFT natural-gold eval step={step}: "
                        f"loss={natural_eval_loss:.6f} "
                        f"ppl={math.exp(min(natural_eval_loss, 20)):.2f} "
                        f"tokens={natural_eval_tokens:,}"
                    )
            selection_eligible = early_stopping_is_eligible(
                step,
                steps_per_epoch=steps_per_epoch,
                minimum_epochs=args.early_stopping_min_epochs,
            )
            if selection_eligible:
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
                        source_checkpoint,
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
            elif is_main:
                logger.info(
                    "Checkpoint selection deferred until the selected stage "
                    f"has completed {args.early_stopping_min_epochs} full "
                    f"epoch(s) at step {minimum_selection_step}."
                )
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
            expert_lr_now = next(
                (
                    lr
                    for group, lr in zip(
                        optimizer.param_groups,
                        scheduler.get_last_lr(),
                    )
                    if str(group.get("name", "")).startswith("expert_")
                ),
                None,
            )
            if is_main:
                writer.writerow(
                    [
                        step,
                        f"{train_loss:.6f}",
                        f"{train_ppl:.2f}",
                        "" if eval_loss is None else f"{eval_loss:.6f}",
                        "" if eval_loss is None else f"{math.exp(min(eval_loss, 20)):.2f}",
                        f"{lr_now:.6e}",
                        "" if expert_lr_now is None else f"{expert_lr_now:.6e}",
                        f"{tok_s:.0f}",
                        stats["supervised_tokens"],
                        stats["min_label"],
                        stats["max_label"],
                        stats["bad_labels"],
                        eval_tokens if eval_loss is not None else "",
                        "" if natural_eval_loss is None else f"{natural_eval_loss:.6f}",
                        (
                            ""
                            if natural_eval_loss is None
                            else f"{math.exp(min(natural_eval_loss, 20)):.2f}"
                        ),
                        natural_eval_tokens if natural_eval_loss is not None else "",
                    ]
                )
                csv_file.flush()
                pbar.set_postfix(loss=f"{train_loss:.4f}", tok_s=f"{tok_s:.0f}")
            t_log = now
            tokens_since_log = 0

        if args.empty_cache_every > 0 and step % args.empty_cache_every == 0:
            empty_cache(device)
        periodic_save = args.save_steps > 0 and step % args.save_steps == 0
        milestone_save = step in save_milestones
        if periodic_save or milestone_save:
            save_checkpoint(
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
                best_eval_loss=best_eval_loss,
                evaluations_without_improvement=evaluations_without_improvement,
                force=milestone_save,
            )
        if stop_training:
            if is_main:
                logger.info(
                    f"Early stopping at step={step}: best_eval_loss={best_eval_loss:.6f}, "
                    f"evaluations_without_improvement={evaluations_without_improvement}"
                )
            break

    # ``save_steps=0`` disables periodic checkpoints, not the final artifact.
    # A completed run always gets a final artifact when it did not land exactly
    # on a periodic or milestone checkpoint.
    if last_step > 0 and not (
        (args.save_steps > 0 and last_step % args.save_steps == 0)
        or last_step in save_milestones
    ):
        save_checkpoint(
            args,
            raw_model,
            optimizer,
            scheduler,
            config,
            source_checkpoint,
            last_step,
            is_main,
            distributed,
            chat_template,
            force=True,
            best_eval_loss=best_eval_loss,
            evaluations_without_improvement=evaluations_without_improvement,
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
