"""Configuration objects for staged SFT curricula."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetMix:
    """Weighted dataset mix for one SFT stage."""

    sources: dict[str, float]
    records: int = 80_000
    seed: int = 42
    out: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetMix":
        known = {"records", "seed", "out", "sources"}
        sources = dict(data.get("sources") or {k: v for k, v in data.items() if k not in known})
        if not sources:
            raise ValueError("dataset mix must contain at least one source")
        return cls(
            sources={str(k): float(v) for k, v in sources.items()},
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
