#!/usr/bin/env python3
"""Interactive REPL on top of MPSOnlineRLEngine.

You type questions, the model responds, and the engine silently accumulates
monitoring events. Every ``--min-events`` events it auto-runs an RL update,
saves a checkpoint, and hot-reloads the server from it. You keep typing.

Arithmetic questions are scored via the verified-tool path (Python computes
the truth, the engine trains on the correct trace). Anything else flows
through entropy/confidence-triggered monitoring with no ground truth — those
events only fire if the model itself was uncertain.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from complexity.inference.mps_online_rl_engine import (
    MPSOnlineRLEngine,
    MPSOnlineRLEngineConfig,
)


PROMPT_HELP = """\
Commands:
  :q              quit
  :force          force an RL update on the next message
  :feedback wrong  flag the LAST response as incorrect (negative reward)
  :feedback good   flag the LAST response as correct (positive reward)
  :help            show this help
Anything else is treated as a user question.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, default=REPO_ROOT / "tokenizer-o200k")
    ap.add_argument("--output-dir", type=Path, default=Path("runs/online_rl_repl"))
    ap.add_argument("--min-events", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-7)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    engine = MPSOnlineRLEngine(MPSOnlineRLEngineConfig(
        checkpoint=args.checkpoint,
        tokenizer=args.tokenizer,
        output_dir=args.output_dir,
        min_events_before_update=args.min_events,
        learning_rate=args.lr,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        reload_after_save=True,
        seed=args.seed,
    ))

    print(f"\nLoaded {args.checkpoint}")
    print(f"Output dir: {args.output_dir}")
    print(f"Auto-RL fires every {args.min_events} monitored events.\n")
    print(PROMPT_HELP)

    pending_feedback: str | None = None
    last_prompt: str | None = None
    force_update = False
    served = 0

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line in (":q", ":quit", ":exit"):
            break
        if line == ":help":
            print(PROMPT_HELP)
            continue
        if line == ":force":
            force_update = True
            print("(next message will force an RL update)")
            continue
        if line.startswith(":feedback "):
            pending_feedback = line.removeprefix(":feedback ").strip()
            print(f"(feedback queued for next message: {pending_feedback!r})")
            continue

        served += 1
        response, stats, ckpt = engine.infer_with_verified_tools(
            line,
            force_update=force_update,
            user_feedback=pending_feedback,
        )
        pending_feedback = None
        force_update = False
        last_prompt = line

        pending = len(engine.loop.pending_events)
        threshold = engine.loop.config.min_events_before_update
        print(f"model> {response}")
        print(f"       [served={served} pending_events={pending}/{threshold}]")

        if stats is not None:
            print(
                f"       [auto-RL fired: events={stats['num_events']:.0f} "
                f"mean_reward={stats['mean_reward']:.3f} "
                f"loss={stats['loss']:.4f} grad_norm={stats['grad_norm']:.3f}]"
            )
            if ckpt is not None:
                print(f"       [new checkpoint: {ckpt}]")
            print("       (model hot-reloaded; continuing to serve)\n")
        else:
            print()


if __name__ == "__main__":
    main()
