"""POSIX wall-clock guard for the fixed public challenge command."""

import signal
from pathlib import Path
from types import FrameType

from r0.modules.replay import write_challenge_report


def _expired(signum: int, frame: FrameType | None) -> None:
    raise TimeoutError("public challenge command exceeded 300 seconds")


def run_challenges(output: Path) -> str:
    if not hasattr(signal, "SIGALRM"):
        raise ValueError("challenge command requires POSIX SIGALRM for its wall-clock cap")
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise ValueError("challenge command requires no existing process alarm")
    previous = signal.signal(signal.SIGALRM, _expired)
    signal.setitimer(signal.ITIMER_REAL, 300.0)
    try:
        return write_challenge_report(output)["public_status"]
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)
