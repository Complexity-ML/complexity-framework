"""Small text-lineage adapter over the shared training pipeline.

Plan and recipe checks run during argument parsing, before the parent runner
initializes devices or reads tokenizers. No subprocess launcher is involved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from complexity.config import PRESET_CONFIGS

from .corpus_mixture import PretokenizedCorpusMixtureDataset
from .lineage import make_tr_hash_config, validate_lineage_training_args
from .runner import TrainRunner, resolve_warmup_steps


class _AuditedParser(argparse.ArgumentParser):
    def __init__(self, parent, audit, arguments=None):
        super().__init__(parents=[parent], add_help=False, description=parent.description)
        self.audit = audit
        self.arguments = arguments

    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(self.arguments if args is None else args, namespace)
        self.audit(parsed)
        return parsed


def resume_skip_rows(args) -> int:
    """Convert a committed update cursor to this rank's consumed rows."""
    if not args.resume:
        return 0
    if args.resume == "auto":
        from complexity.utils.checkpointing import peek_latest_checkpoint_step

        step = peek_latest_checkpoint_step(args.checkpoint_dir)
    else:
        state_path = Path(args.resume) / "training_state.json"
        step = json.loads(state_path.read_text())["step"] if state_path.is_file() else None
    if step is None:
        return 0
    if type(step) is not int or step < 0:
        raise ValueError("lineage resume update cursor must be a nonnegative integer")
    return step * args.batch_size * args.gradient_accumulation


class TRHashTextLineageRunner(TrainRunner):
    """Reuse the native pipeline with audited pretraining/refinement inputs."""

    def __init__(self, preset="complexity-small", **kwargs):
        defaults = {
            "make_config": lambda: make_tr_hash_config(preset),
            "run_name": f"tr-hash-{preset}-text-lineage",
            "checkpoint_dir": f"artifacts/tr_hash_{preset}_text_lineage",
            "default_lr": 3e-4,
            "default_batch_size": 4,
            "default_seq_len": 1024,
            "default_gradient_accumulation": 1,
            "default_save_steps": 0,
            "default_distributed_mode": "ddp",
            "optimizer_type": "adamw",
            "label_smoothing": 0.0,
        }
        defaults.update(kwargs)
        super().__init__(**defaults)

    def add_args(self, parser):
        parser.add_argument("--model-preset", choices=sorted(PRESET_CONFIGS), required=True)
        parser.add_argument("--stage", choices=("pretraining", "refinement"), required=True)
        parser.add_argument("--tokenized-data", required=True)
        parser.add_argument("--tokenized-plan", required=True)
        parser.add_argument("--pretrain-plan", default=None)
        parser.add_argument("--tokenized-cache-dir", default="artifacts/tr_hash_token_cache")
        parser.add_argument("--tokenized-cache-gb", type=float, default=16.0)
        parser.add_argument("--tokenized-revision", required=True)
        parser.add_argument("--tokenized-prefetch-shards", type=int, default=0)
        parser.add_argument("--tokenized-hf-token-env", default="HF_TOKEN")

    def _build_parser(self):
        parent = super()._build_parser()
        parent.set_defaults(target_tokens=None)

        def audit(args):
            report = validate_lineage_training_args(
                args,
                world_size=getattr(self, "_audit_world_size", int(os.environ.get("WORLD_SIZE", "1"))),
            )
            args.max_steps, args.target_tokens = report["max_steps"], report["scheduled_tokens"]
            args.lineage_audit = report
            self.make_config = lambda: make_tr_hash_config(args.model_preset)

        return _AuditedParser(parent, audit, getattr(self, "_configured_arguments", None))

    def build_dataset(self, tokenizer, args, rank: int, world_size: int):
        del tokenizer
        report = validate_lineage_training_args(args, world_size=world_size)
        if str(args.tokenized_data).startswith("hf://") and args.num_workers != 0:
            raise ValueError("remote pretokenized mixtures require num_workers=0")
        dataset = PretokenizedCorpusMixtureDataset(
            args.tokenized_data,
            rank=rank,
            world_size=world_size,
            cache_dir=args.tokenized_cache_dir,
            cache_max_bytes=int(args.tokenized_cache_gb * 1024**3),
            revision=args.tokenized_revision,
            token=os.environ.get(args.tokenized_hf_token_env),
            prefetch_shards=args.tokenized_prefetch_shards,
            replay_plan=args.tokenized_plan,
            resume_skip_rows=resume_skip_rows(args),
        )
        if dataset.seq_len != args.seq_len or dataset.trained_tokens < report["scheduled_tokens"]:
            raise ValueError("loaded corpus does not cover the audited training schedule")
        return dataset

    def build_compute_loss(self, trainer, model):
        """Preserve pre-shifted targets and support the canonical tied head."""
        from complexity.core.losses import causal_lm_loss_from_hidden

        def compute_loss(current, batch):
            outputs = current(batch["input_ids"].to(trainer.device), return_logits=False)
            base = current
            while hasattr(base, "module"):
                base = base.module
            weight = base.embed_tokens.weight if base.lm_head is None else base.lm_head.weight
            loss, _ = causal_lm_loss_from_hidden(
                outputs["last_hidden_state"],
                weight,
                batch["labels"].to(trainer.device),
                shift=False,
                chunk_tokens=256,
                sync_metrics=False,
            )
            return loss

        return compute_loss


def _settings_arguments(parser, settings: dict) -> list[str]:
    """Use the runner's actual options while rejecting JSON coercion and typos."""
    actions = {}
    for action in parser._actions:
        if action.dest not in {"help", "stage", "model_preset"} and action.option_strings:
            actions.setdefault(action.dest, []).append(action)
    unknown = settings.keys() - actions.keys()
    if unknown:
        raise ValueError(f"unknown lineage settings: {sorted(unknown)}")
    nullable = {
        "target_tokens", "max_steps", "warmup_tokens", "warmup_steps", "resume",
        "init_checkpoint", "pretrain_plan", "wandb", "top_k", "stack_edu_data",
    }
    arguments = []
    for name, value in settings.items():
        options = actions[name]
        action = options[0]
        if value is None:
            if name not in nullable:
                raise ValueError(f"{name} cannot be null")
            continue
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            if type(value) is not bool:
                raise ValueError(f"{name} must be a JSON boolean")
            selected = next((option for option in options if option.const == value), None)
            if selected is not None:
                arguments.append(selected.option_strings[0])
            elif value != action.default:
                raise ValueError(f"{name} does not support {value!r}")
            continue
        expected = action.type or str
        if expected is int and type(value) is not int:
            raise ValueError(f"{name} must be a JSON integer")
        if expected is float and (type(value) not in {int, float} or not math.isfinite(value)):
            raise ValueError(f"{name} must be a finite JSON number")
        if expected is str and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{name} must be a nonempty JSON string")
        if action.choices is not None and value not in action.choices:
            raise ValueError(f"{name} must be one of {list(action.choices)}")
        arguments.extend((action.option_strings[0], str(value)))
    return arguments


def execute_lineage_config(path: str | Path, *, run: bool = False) -> dict:
    """Preflight or explicitly execute one typed JSON plan with the native runner.

    A distributed run must already be launched by torchrun. Preflight can audit
    its declared world size without starting processes or reading credentials.
    Paths authored inside this config are resolved relative to the config file.
    """
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    allowed = {"format", "profile", "stage", "world_size", "settings"}
    if not isinstance(data, dict) or data.keys() - allowed:
        raise ValueError("lineage config must be a JSON object with format/profile/stage/world_size/settings")
    if data.get("format") != "complexity-text-lineage-v1":
        raise ValueError("lineage config format must be complexity-text-lineage-v1")
    profile, stage = data.get("profile"), data.get("stage")
    if profile not in PRESET_CONFIGS and profile != "tr-hash-200m":
        raise ValueError("profile must be a canonical model preset or tr-hash-200m")
    if stage not in {"pretraining", "refinement"}:
        raise ValueError("stage must be pretraining or refinement")
    world_size = data.get("world_size", 1)
    if type(world_size) is not int or world_size < 1:
        raise ValueError("world_size must be a positive JSON integer")
    settings = data.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("lineage settings must contain a JSON object")
    settings = dict(settings)
    for name in ("tokenizer", "tokenized_data", "tokenized_plan", "tokenized_revision", "checkpoint_dir"):
        if not isinstance(settings.get(name), str) or not settings[name].strip():
            raise ValueError(f"lineage settings require an explicit {name}")
    for name in (
        "tokenizer", "tokenized_data", "tokenized_plan", "pretrain_plan", "checkpoint_dir",
        "tokenized_cache_dir", "init_checkpoint", "resume",
    ):
        value = settings.get(name)
        if isinstance(value, str) and not value.startswith("hf://") and not (name == "resume" and value == "auto"):
            settings[name] = str((path.parent / value).resolve())
    settings.setdefault("num_workers", 0)
    if profile == "tr-hash-200m":
        from .text_pretraining import build_200m_runner

        runner = build_200m_runner()
        prefix = []
    else:
        runner = TRHashTextLineageRunner(preset=profile)
        prefix = ["--model-preset", profile, "--stage", stage]
    runner._audit_world_size = world_size
    runner._lineage_config_mode = True
    runner._lineage_stage = stage
    parser = runner._build_parser()
    parser.set_defaults(target_tokens=None)
    arguments = prefix + _settings_arguments(parser, settings)
    parsed = parser.parse_args(arguments)
    for name in ("batch_size", "seq_len", "gradient_accumulation", "log_steps", "token_packs"):
        if getattr(parsed, name) < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("num_workers", "save_steps", "tokenized_prefetch_shards"):
        if getattr(parsed, name) < 0:
            raise ValueError(f"{name} must be nonnegative")
    for name in ("warmup_tokens", "warmup_steps"):
        if getattr(parsed, name) is not None and getattr(parsed, name) < 0:
            raise ValueError(f"{name} must be nonnegative")
    if parsed.lr <= 0 or parsed.weight_decay < 0 or parsed.tokenized_cache_gb <= 0:
        raise ValueError("learning rate/cache must be positive and weight decay nonnegative")
    audited = SimpleNamespace(**vars(parsed))
    audited.stage, audited.model_preset = stage, profile
    # Validate stage, exact unique-core proof and dataset identity for both profiles.
    schedule = validate_lineage_training_args(audited, world_size=world_size)
    schedule = {
        **getattr(parsed, "lineage_audit", schedule),
        "refinement_fingerprint": schedule["refinement_fingerprint"],
    }
    warmup_updates = resolve_warmup_steps(
        max_steps=schedule["max_steps"],
        tokens_per_step=schedule["tokens_per_step"],
        warmup_steps=parsed.warmup_steps,
        warmup_tokens=parsed.warmup_tokens,
    )
    native_config = runner.make_config()
    if parsed.top_k is not None:
        native_config.top_k = parsed.top_k
        native_config = type(native_config).from_dict(native_config.to_dict(), strict=True)
    report = {
        "format": "complexity-text-lineage-preflight-v1",
        "config": str(path),
        "profile": profile,
        "stage": stage,
        "world_size": world_size,
        "schedule": schedule,
        "warmup_updates": warmup_updates,
        "settings": {key: value for key, value in vars(parsed).items() if key != "lineage_audit"},
        "model": native_config.to_dict(),
        "plan_sha256": hashlib.sha256(Path(parsed.tokenized_plan).read_bytes()).hexdigest(),
        "training_started": False,
        "data_verified": False,
        "hardware_qualified": False,
    }
    if world_size > 1:
        report["launch_command"] = [
            sys.executable, "-m", "torch.distributed.run", "--standalone",
            "--nproc_per_node", str(world_size), "-m", "complexity", "train", "lineage", str(path), "--run",
        ]
    if run:
        actual_world = int(os.environ.get("WORLD_SIZE", "1"))
        if actual_world != world_size:
            raise ValueError(
                f"launch world size {actual_world} differs from config {world_size}; "
                "use the preflight launch_command under torchrun"
            )
        runner._configured_arguments = arguments
        runner.run()
        report["training_started"] = True
    return report


def main():
    TRHashTextLineageRunner().run()


if __name__ == "__main__":
    main()
