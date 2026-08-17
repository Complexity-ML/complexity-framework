#!/usr/bin/env python3
"""Watch a supervisord-managed program's log file; force-restart it if the log
goes stale for too long.

Complements timeout-based recovery that only covers specific failure modes
(e.g. the 600s NCCL collective timeout in complexity/parallel/data_parallel.py,
which only fires when a rank is stuck *inside a collective*). A rank stuck
elsewhere -- e.g. a data loader hung mid-download after a network blip, with
no exception and no NCCL call in flight -- has no equivalent watchdog: the
process stays alive, burns CPU, and never crashes, so supervisord's
autorestart=unexpected never has anything to react to. This script is the
backstop: it doesn't care *why* the log went stale, only that it did.

Usage (as its own supervisord program, so it survives independently of the
program it watches):
    python3 scripts/watch_and_restart_stalled_process.py \
        --log-file artifacts/tr_hash_200m_70b_replay.log \
        --program tr_hash_200m_70b_replay \
        --stall-seconds 900 \
        --poll-interval 60
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import time
from pathlib import Path

logger = logging.getLogger("watch_and_restart_stalled_process")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def seconds_since_last_write(log_file: Path, now: float | None = None) -> float | None:
    """Seconds since log_file's mtime, or None if it doesn't exist yet."""
    if not log_file.exists():
        return None
    now = time.time() if now is None else now
    return now - log_file.stat().st_mtime


def restart_program(program: str) -> bool:
    """Run `supervisorctl restart <program>`. Returns True on a clean exit."""
    logger.error(f"{program}: log stale past threshold, restarting via supervisorctl")
    result = subprocess.run(
        ["supervisorctl", "restart", program],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"restart failed: {result.stdout.strip()} {result.stderr.strip()}")
        return False
    logger.info(f"restart output: {result.stdout.strip()}")
    return True


def watch_once(log_file: Path, program: str, stall_seconds: float, now: float | None = None) -> bool:
    """Check staleness once; restart and return True if a restart was triggered."""
    idle = seconds_since_last_write(log_file, now=now)
    if idle is None:
        logger.info(f"{log_file} does not exist yet, waiting")
        return False
    if idle <= stall_seconds:
        return False
    restart_program(program)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-file", required=True, type=Path)
    parser.add_argument("--program", required=True, help="supervisord program name to restart")
    parser.add_argument(
        "--stall-seconds",
        type=float,
        default=900.0,
        help="restart the program if its log hasn't been written to in this many "
        "seconds (default: %(default)s -- comfortably above the 600s NCCL "
        "collective timeout so the two mechanisms don't race each other)",
    )
    parser.add_argument("--poll-interval", type=float, default=60.0)
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=None,
        help="minimum time between restarts (default: stall-seconds, so a "
        "genuinely broken program isn't restart-looped faster than it can "
        "possibly produce a fresh log line)",
    )
    args = parser.parse_args()
    cooldown = args.cooldown_seconds if args.cooldown_seconds is not None else args.stall_seconds

    last_restart = 0.0
    while True:
        now = time.time()
        if now - last_restart >= cooldown:
            if watch_once(args.log_file, args.program, args.stall_seconds, now=now):
                last_restart = now
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
