"""
Distributed cleanup — prevents broken pipe loops on crash.

Usage:
    from complexity.gpu.distributed_cleanup import safe_main

    if __name__ == "__main__":
        safe_main(main)

This wraps main() with proper NCCL cleanup on ANY exit (crash, Ctrl+C, success).
Without this, crashed NCCL processes spam "broken pipe" forever.

Complexity-ML — 2026
"""

import os
import sys
import logging

logger = logging.getLogger(__name__)


def safe_main(main_fn):
    """
    Wrap main() with NCCL cleanup on any exit.

    Ensures dist.destroy_process_group() is called even on crashes,
    then os._exit(0) to kill NCCL heartbeat threads cleanly.

    For multi-rank failures, abort the NCCL communicator first so
    other ranks waiting on a collective op die immediately instead
    of looping on broken-pipe warnings.
    """
    exit_code = 0
    try:
        main_fn()
    except KeyboardInterrupt:
        pass
    except SystemExit:
        pass
    except Exception as e:
        logger.error(f"Fatal: {e}", exc_info=True)
        exit_code = 1
    finally:
        # Suppress NCCL/C10D warnings during shutdown to avoid broken pipe spam
        logging.getLogger("torch.distributed").setLevel(logging.CRITICAL)
        logging.getLogger().setLevel(logging.CRITICAL)
        # Disable NCCL heartbeat before cleanup
        os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "0"
        os.environ["NCCL_TIMEOUT"] = "1"
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                # Abort in-flight collective ops first — this unblocks other
                # ranks waiting on this rank, so they raise instead of looping.
                try:
                    pg = dist.group.WORLD
                    if hasattr(pg, "abort"):
                        pg.abort()
                    elif hasattr(pg, "_shutdown"):
                        pg._shutdown()
                except Exception:
                    pass
                dist.destroy_process_group()
        except Exception:
            pass
        # os._exit kills NCCL heartbeat threads immediately — no broken pipe loop
        os._exit(exit_code)
