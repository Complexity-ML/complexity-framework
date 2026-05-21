"""Configuration objects for staged SFT curricula."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SourceConfig:
    """One weighted source in a stack SFT dataset mix."""

    name: str
    weight: float
    kind: str | None = None
    path: str | None = None
    format: str | None = None
    prompt_field: str = "prompt"
    completion_field: str = "completion"
    messages_field: str = "messages"
    instruction_field: str = "instruction"
    input_field: str = "input"
    output_field: str = "output"
    max_records: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceConfig":
        name = str(data.get("name") or data.get("kind") or data.get("path") or "")
        if not name:
            raise ValueError("source is missing name/kind/path")
        if "weight" not in data:
            raise ValueError(f"source {name!r} is missing weight")
        return cls(
            name=name,
            weight=float(data["weight"]),
            kind=data.get("kind"),
            path=data.get("path"),
            format=data.get("format"),
            prompt_field=str(data.get("prompt_field", "prompt")),
            completion_field=str(data.get("completion_field", "completion")),
            messages_field=str(data.get("messages_field", "messages")),
            instruction_field=str(data.get("instruction_field", "instruction")),
            input_field=str(data.get("input_field", "input")),
            output_field=str(data.get("output_field", "output")),
            max_records=data.get("max_records"),
        )


@dataclass
class DatasetMix:
    """Weighted dataset mix for one SFT stage."""

    sources: list[SourceConfig]
    records: int = 80_000
    seed: int = 42
    out: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetMix":
        raw_sources = data.get("sources")
        if not isinstance(raw_sources, list):
            raise ValueError(
                "dataset mix sources must be a list of file-backed source configs; "
                "hardcoded synthetic source mappings are not supported"
            )
        sources = [SourceConfig.from_dict(dict(item)) for item in raw_sources]
        if not sources:
            raise ValueError("dataset mix must contain at least one source")
        return cls(
            sources=sources,
            records=int(data.get("records", 80_000)),
            seed=int(data.get("seed", 42)),
            out=data.get("out"),
        )


@dataclass
class StageConfig:
    name: str
    steps: int
    lr: float
    mix: DatasetMix
    checkpoint: str | None = None
    batch_size: int | None = None
    seq_len: int | None = None
    save_steps: int | None = None
    save_total_limit: int | None = None
    grad_ckpt: bool | None = None
    loss_chunk_tokens: int | None = None
    extra_args: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StageConfig":
        if "name" not in data:
            raise ValueError("stage is missing name")
        if "mix" not in data:
            raise ValueError(f"stage {data['name']!r} is missing mix")
        return cls(
            name=str(data["name"]),
            steps=int(data["steps"]),
            lr=float(data["lr"]),
            mix=DatasetMix.from_dict(dict(data["mix"])),
            checkpoint=data.get("checkpoint"),
            batch_size=data.get("batch_size"),
            seq_len=data.get("seq_len"),
            save_steps=data.get("save_steps"),
            save_total_limit=data.get("save_total_limit"),
            grad_ckpt=data.get("grad_ckpt"),
            loss_chunk_tokens=data.get("loss_chunk_tokens"),
            extra_args=[str(x) for x in data.get("extra_args", [])],
        )


@dataclass
class StackSFTConfig:
    name: str
    base_checkpoint: str
    tokenizer: str = "./tokenizer-o200k"
    output_dir: str = "checkpoints"
    data_dir: str = "data/sft/stack_sft"
    run_dir: str = "runs"
    batch_size: int = 128
    seq_len: int = 1024
    save_steps: int = 250
    save_total_limit: int = 2
    grad_ckpt: bool = True
    bf16: bool = True
    use_custom_kernels: str = "auto"
    loss_chunk_tokens: int = 512
    stages: list[StageConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StackSFTConfig":
        stages = [StageConfig.from_dict(stage) for stage in data.get("stages", [])]
        if not stages:
            raise ValueError("stack config must define at least one stage")
        return cls(
            name=str(data["name"]),
            base_checkpoint=str(data["base_checkpoint"]),
            tokenizer=str(data.get("tokenizer", "./tokenizer-o200k")),
            output_dir=str(data.get("output_dir", "checkpoints")),
            data_dir=str(data.get("data_dir", "data/sft/stack_sft")),
            run_dir=str(data.get("run_dir", "runs")),
            batch_size=int(data.get("batch_size", 128)),
            seq_len=int(data.get("seq_len", 1024)),
            save_steps=int(data.get("save_steps", 250)),
            save_total_limit=int(data.get("save_total_limit", 2)),
            grad_ckpt=bool(data.get("grad_ckpt", True)),
            bf16=bool(data.get("bf16", True)),
            use_custom_kernels=str(data.get("use_custom_kernels", "auto")),
            loss_chunk_tokens=int(data.get("loss_chunk_tokens", 512)),
            stages=stages,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StackSFTConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"invalid stack config: {path}")
        return cls.from_dict(data)
