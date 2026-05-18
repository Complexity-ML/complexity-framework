"""YAML run configuration, summaries, and resume guards."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


RUN_CONFIG_FILE = "run_config.json"

VOLATILE_RESUME_KEYS = {
    "config",
    "resume",
    "force_resume",
    "run_name",
    "save_dir",
    "save_steps",
    "save_total_limit",
    "log_steps",
    "eval_steps",
    "eval_batches",
    "empty_cache_every",
    "num_workers",
}


def parse_args_with_yaml_config(
    parser: argparse.ArgumentParser,
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse CLI args with optional YAML defaults from ``--config``.

    CLI flags always win over YAML values. YAML keys may use either underscores
    or hyphens, and may be nested under a top-level ``run`` key.
    """

    parser.add_argument("--config", type=str, default=None, help="YAML file with run defaults")
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)
    if pre_args.config:
        parser.set_defaults(**load_yaml_run_config(pre_args.config, parser))
    return parser.parse_args(argv)


def load_yaml_run_config(path: str | Path, parser: argparse.ArgumentParser) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML config must be a mapping: {config_path}")

    data = raw.get("run", raw)
    if not isinstance(data, dict):
        raise ValueError(f"YAML 'run' section must be a mapping: {config_path}")

    valid = parser_destinations(parser)
    normalized: dict[str, Any] = {}
    unknown: list[str] = []
    for key, value in data.items():
        dest = str(key).replace("-", "_")
        if dest not in valid:
            unknown.append(str(key))
        else:
            normalized[dest] = value

    if unknown:
        raise ValueError(f"Unknown YAML config keys in {config_path}: {', '.join(sorted(unknown))}")
    return normalized


def parser_destinations(parser: argparse.ArgumentParser) -> set[str]:
    return {action.dest for action in parser._actions if action.dest != argparse.SUPPRESS}


def args_to_run_config(
    args: argparse.Namespace,
    *,
    model_config: dict[str, Any],
    params: int,
    world_size: int,
    backend: dict[str, Any] | None = None,
) -> dict[str, Any]:
    args_dict = vars(args).copy()
    tokens_per_step = int(args_dict["batch_size"]) * int(args_dict["seq_len"]) * int(world_size)
    total_tokens = tokens_per_step * int(args_dict["steps"])
    return {
        "schema_version": 1,
        "git_commit": current_git_commit(),
        "params": params,
        "world_size": world_size,
        "tokens_per_step": tokens_per_step,
        "total_tokens": total_tokens,
        "args": args_dict,
        "model_config": model_config,
        "backend": backend or {},
    }


def write_or_validate_run_config(
    run_dir: str | Path,
    run_config: dict[str, Any],
    *,
    resume: bool,
    force_resume: bool = False,
) -> None:
    path = Path(run_dir) / RUN_CONFIG_FILE
    if not resume or not path.exists():
        write_run_config(path, run_config)
        return

    previous = read_run_config(path)
    mismatches = compare_run_configs(previous, run_config)
    if mismatches and not force_resume:
        detail = "\n".join(f"- {key}: previous={old!r}, current={new!r}" for key, old, new in mismatches[:20])
        more = "" if len(mismatches) <= 20 else f"\n... and {len(mismatches) - 20} more"
        raise ValueError(
            "Resume config mismatch. Use --force-resume only if this is intentional.\n"
            f"{detail}{more}"
        )
    write_run_config(path, run_config)


def read_run_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_run_config(path: str | Path, run_config: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, sort_keys=True)
        f.write("\n")


def compare_run_configs(previous: dict[str, Any], current: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    mismatches: list[tuple[str, Any, Any]] = []
    previous_args = previous.get("args", {})
    current_args = current.get("args", {})
    keys = sorted(set(previous_args) | set(current_args))
    for key in keys:
        if key in VOLATILE_RESUME_KEYS:
            continue
        if key == "steps":
            continue
        old = previous_args.get(key)
        new = current_args.get(key)
        if old != new:
            mismatches.append((f"args.{key}", old, new))

    if previous.get("model_config") != current.get("model_config"):
        mismatches.append(("model_config", previous.get("model_config"), current.get("model_config")))
    return mismatches


def format_run_summary(run_config: dict[str, Any]) -> list[str]:
    args = run_config["args"]
    lines = [
        f"Run: {args['run_name']}",
        f"Model: {run_config['params'] / 1e6:.1f}M params, profile={args.get('profile')}",
        f"Data: dataset={args['dataset']}, tokenizer={args['tokenizer']}, vocab={args['vocab_size']}",
        (
            "Tokens: "
            f"{run_config['tokens_per_step']:,}/step "
            f"x {args['steps']:,} steps = {run_config['total_tokens']:,}"
        ),
        (
            "Schedule: "
            f"lr={args['lr']}, batch/GPU={args['batch_size']}, seq={args['seq_len']}, "
            f"world={run_config['world_size']}"
        ),
        (
            "Checkpoints: "
            f"every {args['save_steps']} steps, keep {args['save_total_limit']}, "
            f"dir={args['save_dir']}"
        ),
    ]
    backend = run_config.get("backend") or {}
    if backend:
        lines.append(
            "Backend: "
            f"{backend.get('backend')} "
            f"device={backend.get('device_name')} "
            f"matmul={backend.get('matmul')} "
            f"distributed={backend.get('distributed')} "
            f"sdpa={backend.get('sdpa')} "
            f"flash={backend.get('flash_attention')} "
            f"custom_triton={backend.get('custom_triton')}"
        )
    return lines


def current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None
