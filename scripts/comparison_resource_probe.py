"""Harmless kernel probes; run in a dedicated systemd user service with Delegate=yes."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from r0.modules.comparison import run_process


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--walkthrough-out", type=Path)
    args = parser.parse_args()
    relative = Path("/proc/self/cgroup").read_text().strip().split("::")[1]
    if not relative.endswith(".service") or "r0-comparison-probes-" not in relative:
        raise ValueError("run in a dedicated r0-comparison-probes-* delegated service")
    root = Path("/sys/fs/cgroup" + relative)
    (root / "supervisor").mkdir()
    (root / "supervisor/cgroup.procs").write_text(str(os.getpid()))
    (root / "cgroup.subtree_control").write_text("+cpu +memory")
    limits = {
        "cpu_max": "100000 100000",
        "threads": 1,
        "memory_max": 64 * 1024**2,
        "swap_max": 0,
        "wall_seconds": 1,
    }
    results = {}
    for name, code in (
        ("membership", "print(open('/proc/self/cgroup').read())"),
        ("timeout", "import time; time.sleep(10)"),
        ("memory", "x=bytearray(128*1024**2); print(len(x))"),
        (
            "descendant_timeout",
            "import subprocess,time; subprocess.Popen(['sleep','10']); time.sleep(10)",
        ),
    ):
        results[name] = run_process(
            [sys.executable, "-c", code], cwd=Path.cwd(), cgroup_root=root, limits=limits
        )
    # A child created by the predictor must not survive whole-cgroup termination.
    results["groups_removed"] = not any(root.glob("r0-comparison-*"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        json.dump(results, output, indent=2)
        output.write("\n")
    assert results["timeout"]["status"] == "TIMEOUT"
    assert results["descendant_timeout"]["status"] == "TIMEOUT"
    assert results["memory"]["status"] == "MEMORY_LIMIT"
    assert "r0-comparison-" in results["membership"]["stdout"]
    assert results["groups_removed"]
    print("PASS: membership, wall deadline, memory limit, descendant termination")
    if args.walkthrough_out:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("comparison_external_walkthrough.py")),
                "--out",
                str(args.walkthrough_out),
                "--cgroup-root",
                str(root),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
