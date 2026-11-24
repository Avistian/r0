"""Linux cgroup v2 attempt boundary. No fallback is an enforced attempt."""

import os
import platform
import signal
import subprocess
import sys
import time
import uuid
from contextlib import suppress
from importlib.metadata import distributions
from pathlib import Path

from .prepare import LIMITS
from .storage import Document, file_hash

THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def environment() -> Document:
    root = Path(__file__).parents[1]
    return {
        "python": sys.version,
        "executable_sha256": file_hash(Path(sys.executable)),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": Path("/proc/cpuinfo").read_text().split("model name", 1)[-1].splitlines()[0],
        "libraries": dict(
            sorted((d.metadata["Name"], d.version) for d in distributions() if d.metadata["Name"])
        ),
        "implementation": {
            str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*.py"))
        },
        "thread_environment": {key: os.environ.get(key) for key in THREAD_ENV},
        "required_thread_environment": {key: "1" for key in THREAD_ENV},
    }


def run_process(
    command: list[str], *, cwd: Path, cgroup_root: Path, limits: Document | None = None
) -> Document:
    """Fail closed unless a delegated cgroup enforces all caps before exec.

    Lower limits are accepted only for harmless enforcement probes. Production
    callers use the frozen limits. The child joins before importing any predictor.
    """
    caps = LIMITS if limits is None else limits
    if (
        caps["memory_max"] > LIMITS["memory_max"]
        or caps["wall_seconds"] > 300
        or caps["cpu_max"] != LIMITS["cpu_max"]
        or caps["swap_max"] != 0
        or caps["threads"] != 1
    ):
        raise ValueError("resource caps cannot be widened")
    group = cgroup_root / f"r0-comparison-{uuid.uuid4().hex}"
    started = time.monotonic()
    process: subprocess.Popen[str] | None = None
    try:
        if sys.platform != "linux" or not (cgroup_root / "cgroup.controllers").is_file():
            raise ValueError("cgroup v2 enforcement unavailable")
        group.mkdir()
        for name, value in (
            ("cpu.max", caps["cpu_max"]),
            ("memory.max", caps["memory_max"]),
            ("memory.swap.max", 0),
            ("memory.oom.group", 1),
        ):
            path = group / name
            if not path.exists():
                raise ValueError(f"cgroup controller unavailable: {name}")
            path.write_text(str(value))
            if path.read_text().strip() != str(value):
                raise ValueError(f"cgroup cap verification failed: {name}")
        if not (group / "cgroup.kill").exists():
            raise ValueError("whole-cgroup termination unavailable")
        bootstrap = (
            "import os,sys; "
            "open(sys.argv[1], 'w').write(str(os.getpid())); "
            "os.execvpe(sys.argv[2],sys.argv[2:],os.environ)"
        )
        env = {
            key: value
            for key, value in os.environ.items()
            if key in ("PATH", "PYTHONPATH", "LANG", "LC_ALL", "VIRTUAL_ENV")
        }
        env.update({key: "1" for key in THREAD_ENV})
        env.update(TABPFN_DISABLE_TELEMETRY="1", PYTHONDONTWRITEBYTECODE="1")
        process = subprocess.Popen(
            [sys.executable, "-c", bootstrap, str(group / "cgroup.procs"), *command],
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        status = "COMPLETED"
        try:
            stdout, stderr = process.communicate(timeout=caps["wall_seconds"])
        except subprocess.TimeoutExpired:
            (group / "cgroup.kill").write_text("1")
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            status = "TIMEOUT"
        events = (group / "memory.events").read_text()
        if any(
            line.startswith("oom_kill ") and int(line.split()[1]) for line in events.splitlines()
        ):
            status = "MEMORY_LIMIT"
        elif process.returncode and status == "COMPLETED":
            status = "FAILED"
        return {
            "status": status,
            "returncode": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "seconds": time.monotonic() - started,
            "limits": caps,
            "enforcement": "linux-cgroup-v2",
            "memory_events": events,
            "cpu_stat": (group / "cpu.stat").read_text(),
        }
    except OSError as error:
        raise ValueError(f"resource enforcement unavailable: {error}") from error
    finally:
        if group.exists():
            kill = group / "cgroup.kill"
            if kill.exists():
                kill.write_text("1")
            if process is not None and process.poll() is None:
                process.wait()
            group.rmdir()
