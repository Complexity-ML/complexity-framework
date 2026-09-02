"""Generic job lifecycle commands."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional, TypeVar

import typer

from complexity.jobs import (
    Job,
    JobManager,
    load_job_manifest,
    parse_job_status,
    wait_for_job_artifacts,
)

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


def _parse_variables(values: list[str]) -> dict[str, str]:
    variables: dict[str, str] = {}
    for value in values:
        name, separator, resolved = value.partition("=")
        if not separator or not name:
            raise ValueError(f"job variable must use NAME=VALUE syntax: {value!r}")
        variables[name] = resolved
    return variables


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


@jobs.command("render-file")
def render_job_file(
    manifest: Path = typer.Argument(..., help="Portable TOML job manifest"),
    root: Optional[Path] = typer.Option(
        None,
        "--root",
        help="Runtime project root; defaults to the manifest directory",
    ),
    variable: list[str] = typer.Option(
        [],
        "--set",
        help="Manifest variable override in NAME=VALUE form; repeat as needed",
    ),
) -> None:
    """Validate and render generated Supervisor programs without installing them."""

    loaded = _call(
        lambda: load_job_manifest(
            manifest,
            root=root,
            variables=_parse_variables(variable),
        )
    )
    for index, job in enumerate(loaded.jobs):
        if index:
            typer.echo()
        typer.echo(job.as_supervisor_program().render(), nl=False)


@jobs.command("submit-file")
def submit_job_file(
    manifest: Path = typer.Argument(..., help="Portable TOML job manifest"),
    root: Optional[Path] = typer.Option(
        None,
        "--root",
        help="Runtime project root; defaults to the manifest directory",
    ),
    variable: list[str] = typer.Option(
        [],
        "--set",
        help="Manifest variable override in NAME=VALUE form; repeat as needed",
    ),
) -> None:
    """Install a portable group of Supervisor jobs and apply it once."""

    loaded = _call(
        lambda: load_job_manifest(
            manifest,
            root=root,
            variables=_parse_variables(variable),
        )
    )
    handles = _call(lambda: JobManager().submit_many(loaded.jobs))
    console.print(success(f"Submitted {len(handles)} job(s) from {manifest}"))
    for handle in handles:
        console.print(f"  {handle.name}")


@jobs.command("run-after")
def run_after_job(
    predecessor: str = typer.Argument(..., help="Managed job that must exit successfully"),
    command: list[str] = typer.Argument(
        ...,
        help="Command argv to execute after readiness; use -- before option-like args",
    ),
    artifact: list[Path] = typer.Option(
        ...,
        "--artifact",
        help="Required non-empty artifact path; repeat as needed",
    ),
    poll_seconds: float = typer.Option(30.0, "--poll-seconds", min=0.1),
    delay_seconds: float = typer.Option(0.0, "--delay-seconds", min=0.0),
    timeout_seconds: Optional[float] = typer.Option(None, "--timeout-seconds", min=0.1),
) -> None:
    """Wait for a verified managed predecessor, then replace this process."""

    _call(
        lambda: wait_for_job_artifacts(
            JobManager(),
            predecessor,
            artifact,
            poll_seconds=poll_seconds,
            timeout_seconds=timeout_seconds,
            on_poll=lambda message: print(f"[run-after] {message}", flush=True),
        )
    )
    if delay_seconds:
        print(f"[run-after] readiness verified; delaying {delay_seconds:g}s", flush=True)
        time.sleep(delay_seconds)
    print(f"[run-after] exec: {' '.join(command)}", flush=True)
    os.execvpe(command[0], command, os.environ.copy())
    raise RuntimeError("exec unexpectedly returned")  # pragma: no cover


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
