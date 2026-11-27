"""Run a generation coordinator in a killable process group."""

import json
import math
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from r0.modules.generation.lib.requests import RunRequest


@dataclass(frozen=True, slots=True)
class JobResult:
    returncode: int
    timed_out: bool
    elapsed_seconds: float
    summary: dict[str, object] | None
    stdout: str
    stderr: str


def run_guarded(
    request: RunRequest,
    out: Path,
    *,
    workers: int = 1,
    retry_failed: bool = False,
    sync_path: Path | None = None,
) -> JobResult:
    """Bound child wall time including startup; kill its process group on expiry.

    sync_path enables `sync PATH` persistence after failures/groups, intended for a
    mounted Modal Volume v2. Provider durability still requires target-runtime proof.
    """
    return run_guarded_document(
        asdict(request), out, workers=workers, retry_failed=retry_failed, sync_path=sync_path
    )


def run_guarded_document(
    request: dict[str, Any],
    out: Path,
    *,
    workers: int = 1,
    retry_failed: bool = False,
    sync_path: Path | None = None,
) -> JobResult:
    """Decode and replay a raw request only inside the timeout-protected worker."""
    if type(request) is not dict:
        raise ValueError("guarded request must be an object")
    seconds = request.get("max_seconds")
    if (
        type(seconds) not in (int, float)
        or not math.isfinite(cast(float, seconds))
        or cast(float, seconds) <= 0
    ):
        raise ValueError("guarded request needs a finite positive timeout")
    if os.name != "posix":
        raise ValueError("guarded generation requires POSIX process groups")
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("workers must be an integer in 1..16")
    if sync_path is not None and not out.resolve().is_relative_to(sync_path.resolve()):
        raise ValueError("output must be inside the requested persistence mount")
    payload = json.dumps(
        {
            "request": request,
            "out": str(out.resolve()),
            "workers": workers,
            "retry_failed": retry_failed,
            "sync_path": str(sync_path.resolve()) if sync_path is not None else None,
        },
        allow_nan=False,
    )
    start = time.monotonic()
    process = subprocess.Popen(
        [sys.executable, "-m", "r0.modules.generation.worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(payload, timeout=seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
    finally:
        # A notebook interrupt must not leave a detached child generating tasks.
        if process.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
    elapsed = time.monotonic() - start
    summary: dict[str, object] | None = None
    if not timed_out and stdout.strip():
        summary = json.loads(stdout)
    return JobResult(process.returncode, timed_out, elapsed, summary, stdout, stderr)
