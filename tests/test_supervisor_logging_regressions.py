"""Framework-wide guards against the stdout-buffering regression class.

An ad-hoc Supervisor deployment once piped a training launcher through
`pty ... | tee` into a portal-captured `/dev/stdout` log, and separately a
tqdm bar was pinned to `file=sys.stdout`. Both silently swallow live
progress: stdout is fully buffered once it isn't a tty (i.e. once piped),
while stderr is not — so nothing surfaced even though training was
progressing normally. These checks apply to every script/conf/callback in
the repo, not just the one training run where it was first found.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SUPERVISOR_CONF_DIRS = ("configs/supervisor", "deploy/supervisor")


def _supervisor_confs() -> list[Path]:
    confs: list[Path] = []
    for directory in SUPERVISOR_CONF_DIRS:
        confs.extend((REPO_ROOT / directory).glob("*.conf"))
    return confs


def _launcher_scripts() -> list[Path]:
    return sorted((REPO_ROOT / "scripts").glob("vast_*.sh"))


def test_no_supervisor_conf_logs_through_the_portal_stdout_capture() -> None:
    confs = _supervisor_confs()
    assert confs, "expected at least one committed supervisor conf"
    for conf in confs:
        text = conf.read_text(encoding="utf-8")
        assert "/dev/stdout" not in text, f"{conf} logs through /dev/stdout"


def test_no_vast_launcher_pipes_torchrun_through_pty_or_tee() -> None:
    launchers = _launcher_scripts()
    assert launchers, "expected at least one committed vast_*.sh launcher"
    for launcher in launchers:
        text = launcher.read_text(encoding="utf-8")
        assert "pty " not in text, f"{launcher} pipes through pty"
        assert "| tee" not in text, f"{launcher} pipes through tee"


def test_no_tqdm_bar_is_pinned_to_stdout() -> None:
    for path in (REPO_ROOT / "complexity").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "tqdm(" not in text:
            continue
        assert "file=sys.stdout" not in text, f"{path} pins a tqdm bar to stdout"
