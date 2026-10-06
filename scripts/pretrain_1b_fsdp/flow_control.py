"""Coordinated upload backpressure at optimizer-update boundaries."""

import time

CHECKPOINT = 1
STOP = 2
WAIT = 4


def upload_control_state(output, runtime, *, limited=False, max_pending=3):
    """Read rank zero's shared files; requests take priority over backpressure."""
    request = CHECKPOINT if (runtime / "request-checkpoint").exists() else 0
    if (runtime / "request-stop").exists():
        request |= STOP
    if request or limited:
        return request
    pending = sum(
        (path / "complete.json").is_file()
        and not (runtime / (path.name + ".uploaded.json")).is_file()
        for path in output.glob("step_*")
    )
    return WAIT if pending >= max_pending else 0


def wait_for_upload_capacity(read_state, synchronize, *, is_main, on_wait,
                             poll_seconds=30, sleep=time.sleep):
    """Every rank polls and sleeps together, with no long-lived collective.

    synchronize broadcasts rank zero's integer to every rank. All ranks must
    call this function, including ranks that cannot read the shared files.
    """
    while True:
        state = synchronize(read_state() if is_main else 0)
        if not state & WAIT:
            return state
        if is_main:
            on_wait()
        sleep(poll_seconds)
