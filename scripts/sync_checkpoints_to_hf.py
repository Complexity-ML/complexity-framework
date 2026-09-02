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
import importlib.util
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
_XET_RETRY_AFTER_MONOTONIC = 0.0


def load_hf_token(token_file: Path | None, env_name: str) -> str:
    """Load an HF token without exposing it in Supervisor configuration."""
    token = None
    if token_file is not None:
        if not token_file.is_file():
            raise SystemExit(f"HF token file does not exist: {token_file}")
        mode = token_file.stat().st_mode & 0o777
        if mode & 0o077:
            raise SystemExit(f"HF token file must not be group/world accessible: {token_file}")
        token = token_file.read_text(encoding="utf-8").strip()
    if not token:
        token = os.environ.get(env_name)
    if not token:
        raise SystemExit(f"neither --hf-token-file nor {env_name} supplied a token")
    return token


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


def repository_checkpoint_name(path: Path, steps_per_epoch: int | None) -> str:
    """Return a stable, human-readable Hub directory name.

    Epoch-boundary ``step_*`` checkpoints are published as ``epoch_N`` when
    the run supplies its exact number of optimizer steps per epoch. Other
    checkpoint kinds retain their local name so interrupted and intermediate
    recovery points remain unambiguous.
    """
    if steps_per_epoch is None or steps_per_epoch <= 0 or not path.name.startswith("step_"):
        return path.name
    step = checkpoint_step(path)
    if step <= 0 or step % steps_per_epoch:
        return path.name
    return f"epoch_{step // steps_per_epoch}"


def load_state(state_path: Path) -> dict:
    if state_path.exists():
        state = json.loads(state_path.read_text())
        state.setdefault("uploaded", [])
        state.setdefault("destinations", {})
        return state
    return {"uploaded": [], "destinations": {}}


def save_state(state_path: Path, state: dict) -> None:
    state_path.write_text(json.dumps(state, indent=2))


def sync_once(
    checkpoint_dir: Path,
    repo_id: str,
    token: str,
    private: bool,
    keep_local: int,
    path_prefix: str = "",
    steps_per_epoch: int | None = None,
) -> None:
    from huggingface_hub import HfApi, create_repo

    state_path = checkpoint_dir / STATE_FILENAME
    state = load_state(state_path)
    uploaded = set(state["uploaded"])
    destinations = dict(state["destinations"])

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
        repository_name = repository_checkpoint_name(pack_dir, steps_per_epoch)
        repository_path = "/".join(
            part for part in (path_prefix.strip("/"), repository_name) if part
        )
        # A checkpoint is only current when both its local identity and exact
        # Hub destination match. This prevents a stale state file from
        # silently preserving an obsolete, deeply nested repository layout.
        if pack_dir.name in uploaded and destinations.get(pack_dir.name) == repository_path:
            continue
        logger.info(f"Uploading {pack_dir.name} as {repository_name} to {repo_id} ...")
        api.upload_folder(
            folder_path=str(pack_dir),
            repo_id=repo_id,
            repo_type="model",
            path_in_repo=repository_path,
            token=token,
        )
        uploaded.add(pack_dir.name)
        destinations[pack_dir.name] = repository_path
        state["uploaded"] = sorted(uploaded)
        state["destinations"] = destinations
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


def run_pass_with_timeout(
    pass_args: list[str],
    pass_timeout: float,
    *,
    environment: dict[str, str] | None = None,
) -> bool:
    """Run one sync pass as a subprocess, killing it if it exceeds pass_timeout.

    A network drop mid-upload can stall the underlying socket read with no
    exception ever raised in-process, so a bare try/except can't catch it --
    only killing the process (which tears down its sockets) reliably unwedges
    it. Returns True on a clean pass, False if it timed out or failed.
    """
    try:
        subprocess.run(pass_args, timeout=pass_timeout, check=True, env=environment)
        return True
    except subprocess.TimeoutExpired:
        logger.error(
            f"sync pass exceeded {pass_timeout:.0f}s (stalled connection?), killed it, retrying"
        )
        return False
    except subprocess.CalledProcessError:
        logger.exception("sync pass failed, will retry")
        return False


def _environment_flag_enabled(environment: dict[str, str], name: str) -> bool:
    return environment.get(name, "").strip().lower() in {"1", "on", "true", "yes"}


def xet_is_available(environment: dict[str, str] | None = None) -> bool:
    """Return whether the preferred Hub Xet transport can be attempted."""
    current_environment = os.environ if environment is None else environment
    if _environment_flag_enabled(current_environment, "HF_HUB_DISABLE_XET"):
        return False
    if time.monotonic() < _XET_RETRY_AFTER_MONOTONIC:
        return False
    try:
        return importlib.util.find_spec("hf_xet") is not None
    except (ImportError, ValueError):
        return False


def xet_circuit_is_open() -> bool:
    return time.monotonic() < _XET_RETRY_AFTER_MONOTONIC


def run_pass_with_transport_fallback(
    pass_args: list[str],
    pass_timeout: float,
    xet_timeout: float,
    *,
    xet_cooldown: float = 900.0,
    environment: dict[str, str] | None = None,
) -> bool:
    """Prefer Xet, then retry the same pass over HTTP if Xet fails or stalls.

    The HTTP override is scoped to child subprocesses. After a transient CAS
    failure, HTTP remains selected for a bounded cooldown and Xet is then
    probed again instead of being permanently disabled.
    """
    preferred_environment = dict(os.environ if environment is None else environment)
    if xet_circuit_is_open():
        preferred_environment["HF_HUB_DISABLE_XET"] = "1"
    should_try_xet = xet_is_available(preferred_environment)
    preferred_timeout = min(pass_timeout, xet_timeout) if should_try_xet else pass_timeout
    if run_pass_with_timeout(
        pass_args,
        preferred_timeout,
        environment=preferred_environment,
    ):
        return True
    if not should_try_xet:
        return False

    global _XET_RETRY_AFTER_MONOTONIC
    _XET_RETRY_AFTER_MONOTONIC = time.monotonic() + max(0.0, xet_cooldown)
    logger.warning("Xet sync pass failed or stalled; retrying this pass over HTTP")
    http_environment = dict(preferred_environment)
    http_environment["HF_HUB_DISABLE_XET"] = "1"
    return run_pass_with_timeout(
        pass_args,
        pass_timeout,
        environment=http_environment,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--repo-id",
        required=True,
        help="e.g. AETHORIA-AI/TR-HASH-MoE-200M-130B-Checkpoints",
    )
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument(
        "--hf-token-file",
        type=Path,
        help="Protected local token file; preferred for Supervisor-managed jobs.",
    )
    parser.add_argument(
        "--keep-local", type=int, default=1, help="most-recent uploaded checkpoints to keep on disk"
    )
    parser.add_argument("--poll-interval", type=float, default=60.0)
    parser.add_argument(
        "--path-prefix",
        default="",
        help="Optional repository subdirectory for uploaded checkpoints.",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        help="Rename exact step_N epoch boundaries to epoch_N in the Hub repository.",
    )
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--once", action="store_true", help="run a single pass instead of polling forever"
    )
    parser.add_argument(
        "--pass-timeout",
        type=float,
        default=1800.0,
        help="hard wall-clock limit (seconds) for one sync pass. A network drop "
        "mid-upload can stall the underlying socket read with no exception ever "
        "raised, so a bare try/except never fires -- each pass runs in a "
        "subprocess that gets killed and retried if it exceeds this budget.",
    )
    parser.add_argument(
        "--xet-timeout",
        type=float,
        default=300.0,
        help="maximum seconds to wait for the preferred Xet pass before retrying "
        "the same pass over HTTP. Xet is probed again on the next polling pass.",
    )
    parser.add_argument(
        "--xet-cooldown",
        type=float,
        default=900.0,
        help="seconds to keep using HTTP after an Xet failure before probing Xet again.",
    )
    args = parser.parse_args()

    token = load_hf_token(args.hf_token_file, args.hf_token_env)

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
            args.steps_per_epoch,
        )
        return

    pass_args = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--repo-id",
        args.repo_id,
        "--hf-token-env",
        args.hf_token_env,
        "--keep-local",
        str(args.keep_local),
        "--private" if args.private else "--no-private",
        "--once",
    ]
    if args.hf_token_file is not None:
        pass_args.extend(["--hf-token-file", str(args.hf_token_file)])
    if args.path_prefix:
        pass_args.extend(["--path-prefix", args.path_prefix])
    if args.steps_per_epoch is not None:
        pass_args.extend(["--steps-per-epoch", str(args.steps_per_epoch)])
    while True:
        run_pass_with_transport_fallback(
            pass_args,
            args.pass_timeout,
            args.xet_timeout,
            xet_cooldown=args.xet_cooldown,
        )
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
