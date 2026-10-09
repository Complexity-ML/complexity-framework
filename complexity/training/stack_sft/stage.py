"""Portable local LoRA SFT stages for the file-backed curriculum."""

from __future__ import annotations

import json
import math
import random
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from complexity.export.assets import copy_tokenizer_files, write_chat_template
from complexity.inference.chat_template import default_chat_template, validate_chat_template
from complexity.training.finetuning import (
    SUPERVISED_FINETUNING_STAGE,
    validate_training_stage_transition,
)
from complexity.training.lora import apply_lora, merged_model_state_dict
from complexity.training.sequence_packing import pack_example_lengths
from complexity.training.sft import (
    _digest,
    _optimizer_schedule,
    _prepare_resume_metrics,
    _restore_rng,
    _save_run_checkpoint,
    accumulated_sft_update,
    load_sft_model,
    reference_epoch_batches,
)


def validate_stack_execution(config, stage):
    """Reject undeclared lineage or unavailable settings before dataset writes."""
    if config.source_stage is None:
        raise ValueError(
            "Stack SFT requires explicit source_stage='refinement' or 'supervised-finetuning'"
        )
    validate_training_stage_transition("text", config.source_stage, SUPERVISED_FINETUNING_STAGE)
    device = torch.device(config.device)
    if device.type not in {"cpu", "mps", "cuda"}:
        raise ValueError("Stack SFT requires an explicit CPU, MPS or CUDA device")
    if config.precision not in {"fp32", "bf16"} or (
        config.precision == "bf16" and device.type != "cuda"
    ):
        raise ValueError("Stack SFT BF16 requires CUDA; CPU/MPS require fp32")
    if stage.extra_args:
        raise ValueError("Legacy extra_args are unsupported; use typed stack configuration fields")
    for name in ("batch_size", "seq_len", "save_steps", "save_total_limit", "loss_chunk_tokens"):
        value = getattr(stage, name)
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"Stack SFT stage {name} must be a positive integer")
    if config.use_custom_kernels not in {"auto", "false", "off"}:
        raise ValueError("Portable Stack SFT does not force custom kernels")
    for name, value in (
        ("steps", stage.steps),
        ("batch_size", stage.batch_size or config.batch_size),
        ("seq_len", stage.seq_len or config.seq_len),
        ("micro_batch_size", config.micro_batch_size),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"Stack SFT {name} must be a positive integer")
    if not math.isfinite(stage.lr) or stage.lr <= 0:
        raise ValueError("Stack SFT learning rate must be finite and positive")
    for name, value in (
        ("save_steps", stage.save_steps or config.save_steps),
        ("save_total_limit", stage.save_total_limit or config.save_total_limit),
        ("loss_chunk_tokens", stage.loss_chunk_tokens or config.loss_chunk_tokens),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"Stack SFT {name} must be a positive integer")
    if type(config.seed) is not int or not 0 <= config.seed < 2**32:
        raise ValueError("Stack SFT seed must be an integer in [0, 2**32)")
    return device


class SFTPromptCompletionDataset(Dataset):
    """Pack literal prompts and completions; supervise completion and final EOS.

    JSONL builders already format the conversation. This reader preserves that
    text rather than rerendering it as another chat format. Examples exceeding
    the context fail closed instead of dropping the assistant's EOS.
    """

    packing_contract = "supervised-finetuning"

    def __init__(self, path, tokenizer, *, sequence_length, vocab_size, chat_template):
        self.chat_template = validate_chat_template(chat_template)
        self.eos_token_id = tokenizer.token_to_id(self.chat_template["eos_token"])
        if self.eos_token_id is None or tokenizer.get_vocab_size() != vocab_size:
            raise ValueError("Stack SFT tokenizer vocabulary/EOS differs from the model")
        self.sequence_length, self.examples = sequence_length, []
        with Path(path).open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row.get("prompt"), str) or not isinstance(
                    row.get("completion"), str
                ):
                    raise ValueError(f"SFT JSONL row {line_no} needs string prompt/completion")
                prompt = tokenizer.encode(row["prompt"], add_special_tokens=False).ids
                completion = tokenizer.encode(row["completion"], add_special_tokens=False).ids
                if not prompt or not completion:
                    raise ValueError(f"SFT JSONL row {line_no} has empty tokenized text")
                full = prompt + completion + [self.eos_token_id]
                inputs, labels = full[:-1], full[1:]
                labels[: len(prompt) - 1] = [-100] * (len(prompt) - 1)
                if len(inputs) > sequence_length:
                    raise ValueError(
                        f"SFT JSONL row {line_no} exceeds context; no truncation allowed"
                    )
                self.examples.append((inputs, labels))
        if not self.examples:
            raise ValueError("SFT JSONL contains no examples")
        self.packing_plan = pack_example_lengths(
            [len(item[0]) for item in self.examples],
            sequence_length=sequence_length,
            separator_tokens=1,
        )

    def __len__(self):
        return self.packing_plan.packed_items

    def __getitem__(self, item):
        inputs, labels = [], []
        for position, index in enumerate(self.packing_plan.packs[item]):
            if position:
                inputs.append(self.eos_token_id)
                labels.append(-100)
            source, target = self.examples[index]
            inputs.extend(source)
            labels.extend(target)
        padding = self.sequence_length - len(inputs)
        return {
            "input_ids": torch.tensor(inputs + [self.eos_token_id] * padding),
            "labels": torch.tensor(labels + [-100] * padding),
        }


def run_lora_stage(config, stage, checkpoint, dataset_path, output, *, resume=None):
    """Fresh adapters/optimizer per stage, exact local checkpoint continuation."""
    from safetensors.torch import save_file
    from tokenizers import Tokenizer

    device = validate_stack_execution(config, stage)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; refusing a silent fallback")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing a silent fallback")
    if config.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This CUDA device does not support BF16")
    checkpoint, output = Path(checkpoint), Path(output)
    stage_marker = checkpoint / "stage.json"
    if (
        stage_marker.is_file()
        and json.loads(stage_marker.read_text()).get("training_stage") != config.source_stage
    ):
        raise ValueError("Declared Stack SFT source stage differs from checkpoint metadata")
    if resume is None and output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "Stack stage output already exists; explicitly resume or use a fresh directory"
        )
    tokenizer_dir = Path(config.tokenizer)
    tokenizer_file = tokenizer_dir / "tokenizer.json" if tokenizer_dir.is_dir() else tokenizer_dir
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    template_path = checkpoint / "chat_template.json"
    template = (
        json.loads(template_path.read_text())
        if template_path.is_file()
        else default_chat_template()
    )
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    model = load_sft_model(checkpoint)
    sequence_length = stage.seq_len or config.seq_len
    if sequence_length > model.config.max_position_embeddings:
        raise ValueError("Stack SFT sequence length exceeds checkpoint context")
    data = SFTPromptCompletionDataset(
        dataset_path,
        tokenizer,
        sequence_length=sequence_length,
        vocab_size=model.config.vocab_size,
        chat_template=template,
    )
    stats = apply_lora(
        model,
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
        targets=config.lora_targets.split(","),
    )
    model.to(device)
    gradient_checkpointing = config.grad_ckpt if stage.grad_ckpt is None else stage.grad_ckpt
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    batch_size = stage.batch_size or config.batch_size
    plan = {
        "purpose": "staged_lora_sft",
        "source_stage": config.source_stage,
        "target_stage": SUPERVISED_FINETUNING_STAGE,
        "release_eligible": False,
        "stage": asdict(stage),
        "lora": {
            "rank": config.lora_rank,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "targets": config.lora_targets,
        },
        "seed": config.seed,
        "packed_items": len(data),
        "batch_size": batch_size,
        "world_size": 1,
        "steps_per_epoch": math.ceil(len(data) / batch_size),
        "optimizer_updates": stage.steps,
        "learning_rate": stage.lr,
        "expert_lr_scale": 1.0,
        "weight_decay": 0.1,
        "betas": [0.9, 0.95],
        "warmup_updates": max(1, int(stage.steps * 0.03)),
        "min_lr_ratio": 0.1,
        "files": {
            str(path.resolve()): _digest(path)
            for path in (
                checkpoint / "config.json",
                checkpoint / "model.safetensors",
                Path(dataset_path),
                tokenizer_file,
            )
        },
    }
    if template_path.is_file():
        plan["files"][str(template_path.resolve())] = _digest(template_path)
    identity = {
        "plan": plan,
        "device": str(device),
        "precision": config.precision,
        "master_parameters": "fp32",
        "micro_batch_size": config.micro_batch_size,
        "gradient_checkpointing": gradient_checkpointing,
        "loss_chunk_tokens": stage.loss_chunk_tokens or config.loss_chunk_tokens,
    }
    optimizer, scheduler = _optimizer_schedule(model, plan)
    step = 0
    if resume is not None:
        from complexity.utils.local_checkpoint import resolve_checkpoint_path

        resume = resolve_checkpoint_path(resume)
        marker = json.loads((resume / "complete.json").read_text())
        if _digest(resume / "checkpoint.pt") != marker["checkpoint_sha256"]:
            raise ValueError("Stack SFT resume integrity failed")
        state = torch.load(resume / "checkpoint.pt", map_location="cpu", weights_only=True)
        if state["identity"] != identity:
            raise ValueError("Stack SFT resume identity differs")
        if marker.get("step") != state["step"] or not 0 <= state["step"] <= stage.steps:
            raise ValueError("Stack SFT resume contains an invalid update cursor")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        _restore_rng(state["rng"], device)
        step = state["step"]
    output.mkdir(parents=True, exist_ok=True)
    if resume is not None:
        _prepare_resume_metrics(output, step)
    (output / "run.json").write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    save_every = stage.save_steps or config.save_steps
    save_limit = stage.save_total_limit or config.save_total_limit

    def save():
        return _save_run_checkpoint(
            output,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            identity=identity,
            device=device,
            history=[],
            total_limit=save_limit,
        )

    if resume is None:
        save()
    with (output / "metrics.jsonl").open("a", encoding="utf-8") as metrics:
        while step < stage.steps:
            epoch, cursor = divmod(step, plan["steps_per_epoch"])
            for position, ranks in enumerate(reference_epoch_batches(plan, epoch)):
                if position < cursor:
                    continue
                values = accumulated_sft_update(
                    model,
                    data,
                    ranks,
                    optimizer,
                    scheduler,
                    micro_batch_size=config.micro_batch_size,
                    precision=config.precision,
                    chunk_tokens=identity["loss_chunk_tokens"],
                )
                step += 1
                metrics.write(json.dumps({"step": step, **values}) + "\n")
                metrics.flush()
                if step % save_every == 0 or step == stage.steps:
                    save()
                if step == stage.steps:
                    break
    final = output / f"step_{step:06d}"
    weights = {name: value.contiguous() for name, value in merged_model_state_dict(model).items()}
    save_file(weights, str(final / ".model.safetensors.tmp"))
    (final / ".model.safetensors.tmp").replace(final / "model.safetensors")
    (final / "config.json").write_text(json.dumps(model.config.to_dict(), indent=2) + "\n")
    write_chat_template({"chat_template": data.chat_template}, final)
    copy_tokenizer_files(tokenizer_file.parent, final)
    if tokenizer_file.name != "tokenizer.json":
        shutil.copyfile(tokenizer_file, final / "tokenizer.json")
    (final / "stage.json").write_text(
        json.dumps(
            {
                "training_stage": SUPERVISED_FINETUNING_STAGE,
                "source_stage": config.source_stage,
                "release_eligible": False,
                "lora": stats,
            },
            indent=2,
        )
        + "\n"
    )
    return str(final)
