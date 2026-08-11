#!/usr/bin/env python3
"""Live terminal dashboard for a detector training metrics.jsonl file."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--steps-per-epoch", type=int, required=True)
    parser.add_argument("--supervisor-program")
    parser.add_argument("--refresh", type=float, default=2.0)
    return parser.parse_args()


def read_metrics(path: Path) -> tuple[dict | None, dict | None, dict | None]:
    latest_train = None
    latest_validation = None
    latest_record = None
    try:
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                latest_record = record
                if "validation" in record:
                    latest_validation = record
                else:
                    latest_train = record
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return latest_train, latest_validation, latest_record


def supervisor_status(program: str | None) -> tuple[str, int | None]:
    if not program:
        return "UNKNOWN", None
    try:
        result = subprocess.run(
            ["supervisorctl", "status", program],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "UNKNOWN", None
    fields = result.stdout.split()
    status = fields[1] if len(fields) > 1 else "UNKNOWN"
    pid = None
    if "pid" in fields:
        try:
            pid = int(fields[fields.index("pid") + 1].rstrip(","))
        except (ValueError, IndexError):
            pass
    return status, pid


def process_elapsed(pid: int | None) -> float | None:
    if pid is None:
        return None
    try:
        result = subprocess.run(
            ["ps", "-o", "etimes=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return float(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None


def gpu_stats() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        values = [value.strip() for value in result.stdout.splitlines()[0].split(",")]
        util, used, total, watts, temperature = values
        return f"GPU {util}% | VRAM {used}/{total} MiB | {watts} W | {temperature} C"
    except (FileNotFoundError, subprocess.TimeoutExpired, IndexError, ValueError):
        return "GPU unavailable"


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "--:--:--"
    return str(timedelta(seconds=max(0, round(seconds))))


def main() -> None:
    args = parse_args()
    bar_width = max(20, min(50, shutil.get_terminal_size((100, 30)).columns - 42))
    print("\033[?1049h\033[?25l", end="", flush=True)
    try:
        while True:
            train, validation, latest_record = read_metrics(args.metrics)
            status, pid = supervisor_status(args.supervisor_program)
            elapsed = process_elapsed(pid)
            step = int(train.get("step", 0)) if train else 0
            progress = min(step / args.total_steps, 1.0)
            filled = round(progress * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            age = time.time() - args.metrics.stat().st_mtime if args.metrics.exists() else None
            # Anchor throughput to the instant the latest metric was written.
            # Otherwise the displayed rate falls and ETA grows while validation
            # legitimately leaves the training step unchanged.
            metric_elapsed = max(elapsed - age, 1.0) if elapsed is not None and age is not None else elapsed
            rate = step / metric_elapsed if metric_elapsed and step else None
            projected_total = args.total_steps / rate if rate else None
            eta = max(projected_total - elapsed, 0.0) if projected_total and elapsed else None
            finish = (
                (datetime.now() + timedelta(seconds=eta)).strftime("%Y-%m-%d %H:%M:%S")
                if eta is not None
                else "--"
            )
            total_epochs = (args.total_steps + args.steps_per_epoch - 1) // args.steps_per_epoch
            metric_epoch = int(train.get("epoch", 0)) + 1 if train else 1
            is_validating = (
                train is not None
                and step % args.steps_per_epoch == 0
                and not (latest_record and "validation" in latest_record)
            )
            phase = f"VALIDATION epoch {metric_epoch}/{total_epochs}" if is_validating else f"TRAINING epoch {metric_epoch}/{total_epochs}"

            print("\033[H", end="")
            print(f"TR-Hash detector | {status} | PID {pid or '-'}")
            print(f"Phase: {phase}")
            print(f"[{bar}] {progress * 100:6.2f}%")
            print(f"Step {step}/{args.total_steps} | {rate:.3f} step/s" if rate else f"Step {step}/{args.total_steps} | -- step/s")
            print(f"Elapsed {duration(elapsed)} | ETA {duration(eta)} | finish ~ {finish}")
            print(gpu_stats())
            if train:
                print(
                    f"Loss {train['loss']:.4f} | obj {train['objectness_loss']:.4f} | "
                    f"box {train['box_loss']:.4f} | cls {train['class_loss']:.4f}"
                )
                print(f"LR {train['lr']:.2e} | expert LR {train['expert_lr']:.2e}")
            else:
                print("Waiting for the first metrics record...")
            if validation:
                metrics = validation["validation"]
                print(
                    f"Last validation: epoch {int(validation['epoch']) + 1} | "
                    f"mAP50 {metrics['map50']:.4f} | best F1 {metrics['best_f1']:.4f}"
                )
            print(f"Metrics updated {duration(age)} ago | refresh {args.refresh:g}s | Ctrl+C to quit")
            print(" " * bar_width)
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        pass
    finally:
        print("\033[?25h\033[?1049l", end="", flush=True)


if __name__ == "__main__":
    main()
