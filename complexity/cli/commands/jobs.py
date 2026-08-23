"""Generic job lifecycle commands."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, TypeVar

import typer

from complexity.jobs import Job, JobManager, parse_job_status

from ..utils import HAS_RICH, console, error, print_table, success

jobs = typer.Typer(name="jobs", help="Run and control long-lived framework jobs")

_T = TypeVar("_T")
_STATE_STYLES = {
    "RUNNING": "bold green",
    "STARTING": "cyan",
    "STOPPING": "yellow",
    "STOPPED": "yellow",
    "BACKOFF": "bold red",
    "EXITED": "yellow",
    "FATAL": "bold red",
    "UNKNOWN": "red",
}


def _call(action: Callable[[], _T]) -> _T:
    try:
        return action()
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        message = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        console.print(error(message or exc.__class__.__name__))
        raise typer.Exit(1) from exc


def _print_status(raw_status: str) -> None:
    rows: list[tuple[str, str, str]] = []
    for line in raw_status.splitlines():
        name, state, details = parse_job_status(line)
        if not name:
            continue
        style = _STATE_STYLES[state]
        rendered_state = f"[{style}]{state}[/{style}]" if HAS_RICH else state
        rows.append((name, rendered_state, details))
    if rows:
        print_table(
            "Framework jobs",
            [
                {"name": "Name", "style": "cyan"},
                {"name": "State"},
                {"name": "Details", "style": "dim"},
            ],
            rows,
        )
    else:
        console.print("[dim]No managed jobs.[/dim]")


def _log_style(line: str) -> str | None:
    lowered = line.lower()
    if any(marker in lowered for marker in ("traceback", "error", "fatal", "outofmemory")):
        return "bold red"
    if "warning" in lowered or "warn" in lowered:
        return "yellow"
    if "[preflight]" in lowered or "checkpoint saved" in lowered:
        return "green"
    if "tr-hash moe sft" in lowered or "%|" in line:
        return "cyan"
    if " | info | " in lowered:
        return "bright_blue"
    return None


def _print_log_line(line: str) -> None:
    console.print(line.rstrip("\n"), style=_log_style(line), markup=False)


@jobs.command("submit")
def submit_job(
    name: str = typer.Argument(..., help="Stable job name"),
    command: list[str] = typer.Argument(..., help="Command argv; use -- before option-like args"),
    directory: Path = typer.Option(..., "--directory", "-C", help="Absolute working directory"),
    log_path: Path = typer.Option(..., "--log", help="Absolute combined output log"),
    no_restart: bool = typer.Option(False, "--no-restart", help="Do not restart after failure"),
) -> None:
    """Submit a shell-free command to the managed backend.

    Example: complexity jobs submit demo -C /workspace --log /workspace/demo.log -- /bin/bash run.sh
    """

    job = _call(
        lambda: Job(
            name=name,
            command=tuple(command),
            directory=directory,
            log_path=log_path,
            autorestart=False if no_restart else "unexpected",
        )
    )
    _call(lambda: JobManager().submit(job))
    console.print(success(f"Submitted {name}"))


@jobs.command("list")
def list_jobs() -> None:
    """List all Supervisor-managed framework jobs."""

    _print_status(_call(lambda: JobManager().list()))


@jobs.command("status")
def status_job(name: str = typer.Argument(..., help="Job name")) -> None:
    """Show one managed job."""

    _print_status(_call(lambda: JobManager().status(name)))


@jobs.command("logs")
def logs_job(
    name: str = typer.Argument(..., help="Job name"),
    lines: int = typer.Option(100, "--lines", "-n", min=1, help="Initial line count"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow appended output"),
) -> None:
    """Read or follow the configured log for one managed job."""

    manager = JobManager()
    try:
        if follow:
            for line in _call(lambda: manager.follow_logs(name, lines=lines)):
                _print_log_line(line)
            return
        for line in _call(lambda: manager.logs(name, lines=lines)).splitlines():
            _print_log_line(line)
    except KeyboardInterrupt:
        raise typer.Exit(130) from None


def _lifecycle(action: str, name: str) -> None:
    manager = JobManager()
    result = _call(lambda: getattr(manager, action)(name))
    if result:
        console.print(result)
    console.print(success(f"{action.capitalize()} completed for {name}"))


@jobs.command("start")
def start_job(name: str = typer.Argument(..., help="Job name")) -> None:
    _lifecycle("start", name)


@jobs.command("stop")
def stop_job(name: str = typer.Argument(..., help="Job name")) -> None:
    _lifecycle("stop", name)


@jobs.command("restart")
def restart_job(name: str = typer.Argument(..., help="Job name")) -> None:
    _lifecycle("restart", name)


@jobs.command("remove")
def remove_job(
    name: str = typer.Argument(..., help="Job name"),
    missing_ok: bool = typer.Option(False, "--missing-ok", help="Ignore a missing job"),
) -> None:
    """Remove one generated job definition and apply the change."""

    _call(lambda: JobManager().remove(name, missing_ok=missing_ok))
    console.print(success(f"Removed {name}"))
