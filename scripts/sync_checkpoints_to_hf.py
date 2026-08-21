#!/usr/bin/env python3
"""Poll a local checkpoint directory, mirror completed checkpoints to a
private HF Hub model repo, then prune local copies once they are backed up.

Watches every checkpoint tag the trainer can produce (token_pack_*, final_*,
interrupted_*, best_*, step_*, ...), so a run that finishes cleanly or
crashes still gets its last checkpoint backed up, not just token-pack
boundaries.

Only ever deletes a local checkpoint AFTER its upload is confirmed, and
always keeps the N most recently uploaded checkpoints locally so a crash can
resume without waiting on a Hub download.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger("sync_checkpoints_to_hf")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

STATE_FILENAME = ".synced_checkpoints.json"


def is_complete_checkpoint(path: Path) -> bool:
    # runner.py's plain "final" export (no step suffix) is a lightweight
    # HF-compatible snapshot: model.safetensors only, no checkpoint.pt.
    if path.name == "final":
        return (path / "model.safetensors").exists()
    return (path / "checkpoint.pt").exists() or any(path.glob("*.metadata"))


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.rsplit("_", 1)[-1])
    except ValueError:
        return -1


def load_state(state_path: Path) -> dict:
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {"uploaded": []}


def save_state(state_path: Path, state: dict) -> None:
    state_path.write_text(json.dumps(state, indent=2))


def sync_once(
    checkpoint_dir: Path,
    repo_id: str,
    token: str,
    private: bool,
    keep_local: int,
    path_prefix: str = "",
) -> None:
    from huggingface_hub import HfApi, create_repo

    state_path = checkpoint_dir / STATE_FILENAME
    state = load_state(state_path)
    uploaded = set(state["uploaded"])

    # Watch every checkpoint tag the trainer can produce (token_pack_*,
    # final_*, interrupted_*, best_*, step_*, ...), not just token-pack
    # boundaries — a run that finishes cleanly or crashes must still get its
    # last checkpoint backed up.
    candidates = sorted(
        (
            p
            for p in checkpoint_dir.iterdir()
            if p.is_dir() and p.name != "tensorboard" and is_complete_checkpoint(p)
        ),
        key=checkpoint_step,
    )

    api = HfApi(token=token)
    create_repo(repo_id, repo_type="model", private=private, token=token, exist_ok=True)

    for pack_dir in candidates:
        if pack_dir.name in uploaded:
            continue
        logger.info(f"Uploading {pack_dir.name} to {repo_id} ...")
        api.upload_folder(
            folder_path=str(pack_dir),
            repo_id=repo_id,
            repo_type="model",
            path_in_repo="/".join(
                part for part in (path_prefix.strip("/"), pack_dir.name) if part
            ),
            token=token,
        )
        uploaded.add(pack_dir.name)
        state["uploaded"] = sorted(uploaded)
        save_state(state_path, state)
        logger.info(f"Uploaded {pack_dir.name}")

    # "final" has no step suffix (checkpoint_step returns -1 for it), which
    # would make pruning treat it as the OLDEST checkpoint and delete it
    # first — it must never be pruned locally, cleanup_tr_hash_200m_checkpoints.py
    # is the only thing allowed to manage its lifecycle.
    uploaded_local = sorted(
        (p for p in candidates if p.name in uploaded and p.name != "final"),
        key=checkpoint_step,
    )
    to_prune = uploaded_local[:-keep_local] if keep_local > 0 else uploaded_local
    for pack_dir in to_prune:
        logger.info(f"Pruning local {pack_dir.name} (already backed up)")
        shutil.rmtree(pack_dir, ignore_errors=True)


def run_pass_with_timeout(pass_args: list[str], pass_timeout: float) -> bool:
    """Run one sync pass as a subprocess, killing it if it exceeds pass_timeout.

    A network drop mid-upload can stall the underlying socket read with no
    exception ever raised in-process, so a bare try/except can't catch it --
    only killing the process (which tears down its sockets) reliably unwedges
    it. Returns True on a clean pass, False if it timed out or failed.
    """
    try:
        subprocess.run(pass_args, timeout=pass_timeout, check=True)
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"sync pass exceeded {pass_timeout:.0f}s (stalled connection?), killed it, retrying")
        return False
    except subprocess.CalledProcessError:
        logger.exception("sync pass failed, will retry")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--repo-id",
        required=True,
        help="e.g. AETHORIA-AI/TR-HASH-MoE-200M-130B-Checkpoints",
    )
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--keep-local", type=int, default=1, help="most-recent uploaded checkpoints to keep on disk")
    parser.add_argument("--poll-interval", type=float, default=60.0)
    parser.add_argument(
        "--path-prefix",
        default="",
        help="Optional repository subdirectory for uploaded checkpoints.",
    )
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--once", action="store_true", help="run a single pass instead of polling forever")
    parser.add_argument(
        "--pass-timeout",
        type=float,
        default=1800.0,
        help="hard wall-clock limit (seconds) for one sync pass. A network drop "
        "mid-upload can stall the underlying socket read with no exception ever "
        "raised, so a bare try/except never fires -- each pass runs in a "
        "subprocess that gets killed and retried if it exceeds this budget.",
    )
    args = parser.parse_args()

    token = os.environ.get(args.hf_token_env)
    if not token:
        raise SystemExit(f"{args.hf_token_env} is not set in the environment")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.once:
        sync_once(
            checkpoint_dir,
            args.repo_id,
            token,
            args.private,
            args.keep_local,
            args.path_prefix,
        )
        return

    pass_args = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--checkpoint-dir", str(checkpoint_dir),
        "--repo-id", args.repo_id,
        "--hf-token-env", args.hf_token_env,
        "--keep-local", str(args.keep_local),
        "--private" if args.private else "--no-private",
        "--once",
    ]
    if args.path_prefix:
        pass_args.extend(["--path-prefix", args.path_prefix])
    while True:
        run_pass_with_timeout(pass_args, args.pass_timeout)
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
