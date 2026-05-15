"""Cluster-scale run planning for TP/PP/DP jobs.

This module is intentionally about *planning and validation*, not launching.
Large cluster launchers differ across Slurm, Kubernetes, Ray, TorchElastic, and
vendor stacks; what must be invariant is the arithmetic and the safety checks.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ParallelPlan:
    tp_size: int
    pp_size: int
    dp_size: int
    micro_batch_size: int
    gradient_accumulation: int

    @property
    def world_size(self) -> int:
        return self.tp_size * self.pp_size * self.dp_size

    @property
    def model_replica_gpus(self) -> int:
        return self.tp_size * self.pp_size

    @property
    def batch_per_dp_replica(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation

    @property
    def global_batch(self) -> int:
        return self.dp_size * self.batch_per_dp_replica


@dataclass(frozen=True)
class ClusterRunPlan:
    name: str
    target_tokens: int
    seq_len: int
    params: int
    parallel: ParallelPlan

    @property
    def chinchilla_tokens(self) -> int:
        return 20 * self.params

    @property
    def chinchilla_multiple(self) -> float:
        return self.target_tokens / self.chinchilla_tokens

    @property
    def tokens_per_step(self) -> int:
        return self.parallel.global_batch * self.seq_len

    @property
    def steps(self) -> int:
        return math.ceil(self.target_tokens / self.tokens_per_step)

    @property
    def actual_tokens(self) -> int:
        return self.steps * self.tokens_per_step

    @property
    def overshoot_tokens(self) -> int:
        return self.actual_tokens - self.target_tokens

    def validate(self) -> None:
        if self.params <= 0:
            raise ValueError("params must be positive")
        if self.target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        p = self.parallel
        for key, value in {
            "tp_size": p.tp_size,
            "pp_size": p.pp_size,
            "dp_size": p.dp_size,
            "micro_batch_size": p.micro_batch_size,
            "gradient_accumulation": p.gradient_accumulation,
        }.items():
            if value <= 0:
                raise ValueError(f"{key} must be positive")

    def summary_lines(self) -> list[str]:
        self.validate()
        p = self.parallel
        return [
            f"run_name              {self.name}",
            f"params                {self.params:,}",
            f"target_tokens         {self.target_tokens:,}",
            f"chinchilla_tokens     {self.chinchilla_tokens:,}",
            f"chinchilla_multiple   {self.chinchilla_multiple:.2f}x",
            f"world_size            {p.world_size:,}",
            f"model_replica_gpus    {p.model_replica_gpus:,} (TP={p.tp_size} * PP={p.pp_size})",
            f"dp_replicas           {p.dp_size:,}",
            f"micro_batch_size      {p.micro_batch_size:,}",
            f"gradient_accumulation {p.gradient_accumulation:,}",
            f"batch_per_dp_replica  {p.batch_per_dp_replica:,}",
            f"global_batch          {p.global_batch:,}",
            f"seq_len               {self.seq_len:,}",
            f"tokens_per_step       {self.tokens_per_step:,}",
            f"steps                 {self.steps:,}",
            f"actual_tokens         {self.actual_tokens:,}",
            f"overshoot_tokens      {self.overshoot_tokens:,}",
        ]


def parse_tokens(value: Any) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip().lower().replace("_", "")
    multipliers = {
        "k": 1_000,
        "m": 1_000_000,
        "b": 1_000_000_000,
        "t": 1_000_000_000_000,
    }
    if text[-1:] in multipliers:
        return int(float(text[:-1]) * multipliers[text[-1]])
    return int(float(text))


def load_cluster_run_plan(path: str | Path) -> ClusterRunPlan:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Cluster config must be a mapping: {config_path}")

    run = raw.get("run", {})
    parallel = raw.get("parallel", {})
    model = raw.get("model", {})
    if not isinstance(run, dict) or not isinstance(parallel, dict) or not isinstance(model, dict):
        raise ValueError("Cluster config sections 'model', 'run', and 'parallel' must be mappings")

    plan = ClusterRunPlan(
        name=str(run.get("run_name") or config_path.stem),
        target_tokens=parse_tokens(run["target_tokens"]),
        seq_len=int(run["seq_len"]),
        params=parse_tokens(model["params"]),
        parallel=ParallelPlan(
            tp_size=int(parallel["tp_size"]),
            pp_size=int(parallel["pp_size"]),
            dp_size=int(parallel["dp_size"]),
            micro_batch_size=int(parallel["micro_batch_size"]),
            gradient_accumulation=int(parallel["gradient_accumulation"]),
        ),
    )
    plan.validate()
    expected_world = raw.get("world_size")
    if expected_world is not None and int(expected_world) != plan.parallel.world_size:
        raise ValueError(
            f"world_size mismatch: config={expected_world}, "
            f"parallel product={plan.parallel.world_size}"
        )
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan and validate a TP/PP/DP cluster run")
    parser.add_argument("--config", required=True, help="Cluster YAML config")
    args = parser.parse_args()

    plan = load_cluster_run_plan(args.config)
    for line in plan.summary_lines():
        print(line)


if __name__ == "__main__":
    main()
