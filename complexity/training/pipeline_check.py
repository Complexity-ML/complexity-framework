"""Sanity checks for export-based Token-Routed pipeline parallelism."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import yaml

from complexity.models import ComplexityModel
from complexity.parallel.pipeline_export import pipeline_split_spec, trace_pipeline
from complexity.training.cluster_plan import load_cluster_run_plan
from complexity.training.o200k_pretrain import PROFILES, make_config


def _load_raw_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Cluster config must be a mapping: {path}")
    return raw


def _profile_args(raw: dict[str, Any]) -> SimpleNamespace:
    model = raw.get("model", {})
    run = raw.get("run", {})
    if not isinstance(model, dict) or not isinstance(run, dict):
        raise ValueError("Cluster config sections 'model' and 'run' must be mappings")

    profile_name = str(model.get("profile", "100m"))
    if profile_name not in PROFILES:
        raise ValueError(f"Unknown o200k profile: {profile_name}")

    profile = PROFILES[profile_name]
    return SimpleNamespace(
        **profile,
        vocab_size=int(model.get("vocab_size", 200019)),
        use_mu_guidance=bool(run.get("use_mu_guidance", False)),
        learn_shared_routed_gates=bool(run.get("learn_shared_routed_gates", True)),
        shared_gate_init=float(run.get("shared_gate_init", 1.0)),
        routed_gate_init=float(run.get("routed_gate_init", 0.1)),
        top_k=int(run.get("top_k", 2)),
        top_k_primary_weight=float(run.get("top_k_primary_weight", 0.5)),
        static_expert_capacity=bool(run.get("static_expert_capacity", False)),
        routing_strategy=str(run.get("routing_strategy", "zipf")),
        mu_clamp=bool(run.get("mu_clamp", False)),
        mu_norm=bool(run.get("mu_norm", False)),
        mu_alpha_init=float(run.get("mu_alpha_init", 1.0)),
        mu_init_value=float(run.get("mu_init_value", 0.0)),
        mu_context_min=float(run.get("mu_context_min", -2.0)),
        mu_context_max=float(run.get("mu_context_max", 2.0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Token-Routed pipeline tracing")
    parser.add_argument("--config", required=True, help="Cluster YAML config")
    parser.add_argument(
        "--trace-profile",
        action="store_true",
        help="Instantiate the configured profile and trace it. Use this on a machine with enough RAM/VRAM.",
    )
    parser.add_argument(
        "--trace-toy",
        action="store_true",
        help="Trace a tiny TR model with the same PP size. Useful on laptops/CI.",
    )
    args = parser.parse_args()

    raw = _load_raw_config(args.config)
    plan = load_cluster_run_plan(args.config)
    pp_size = plan.parallel.pp_size
    profile_args = _profile_args(raw)
    profile_config = make_config(profile_args)

    split_spec = pipeline_split_spec(profile_config.num_hidden_layers, pp_size)
    print(f"run_name         {plan.name}")
    print(f"profile_layers   {profile_config.num_hidden_layers}")
    print(f"pp_size          {pp_size}")
    print(f"split_points     {', '.join(split_spec)}")

    if not args.trace_profile and not args.trace_toy:
        print("trace            skipped (pass --trace-toy or --trace-profile)")
        return

    if args.trace_toy:
        profile_config.hidden_size = 32
        profile_config.num_hidden_layers = max(pp_size, 2)
        profile_config.num_attention_heads = 4
        profile_config.num_key_value_heads = 2
        profile_config.intermediate_size = 64
        profile_config.shared_intermediate_size = 64
        profile_config.vocab_size = 128

    model = ComplexityModel(profile_config)
    example_input_ids = torch.randint(
        0,
        profile_config.vocab_size,
        (1, min(16, plan.seq_len)),
        dtype=torch.long,
    )
    trace_pipeline(model, example_input_ids, pp_size=pp_size)
    print("trace            ok")
    print("static_dispatch  enabled")


if __name__ == "__main__":
    main()
