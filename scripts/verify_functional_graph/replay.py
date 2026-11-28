"""Portable exact-head bounded SPEC verification. Never modifies the source tree."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--source-root", required=True, type=Path)
ap.add_argument("--output-dir", required=True, type=Path)
ap.add_argument("--expected-head", required=True)
ap.add_argument("--python", default=sys.executable)
args = ap.parse_args()
source = args.source_root.resolve()
out = args.output_dir.resolve()


def git(*argv):
    return subprocess.check_output(["git", *argv], cwd=source, text=True).strip()


assert git("rev-parse", "HEAD") == args.expected_head, "Wrong source HEAD"
initial_status = git("status", "--short")
assert initial_status in ("", "?? .venv"), initial_status
out.mkdir(parents=True, exist_ok=False)
env = dict(
    os.environ, PYTHONPATH=str(source / "src"), PYTHONDONTWRITEBYTECODE="1", R0_VERIFY_OUT=str(out)
)
# One CPU thread keeps tiny verification work bounded and reproducible.
env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
programs = Path(__file__).resolve().parent / "portable_programs"
files = [
    "repaired_diagnostic_reproducer.py",
    "family_oracle.py",
    "moments_ordering.py",
    "decoding_admission.py",
    "root_compatibility.py",
    "materialization_compatibility.py",
    "session_affine.py",
    "diagnostics_resources.py",
]
commands = []
status = "PASS"
started = time.time()
for name in files:
    destination = out / name
    destination.write_bytes((programs / name).read_bytes())
    command = [args.python, str(destination)]
    run = subprocess.run(command, cwd=source, env=env, text=True, capture_output=True)
    (out / (name + ".stdout")).write_text(run.stdout)
    (out / (name + ".stderr")).write_text(run.stderr)
    commands.append(
        {
            "argv": command,
            "cwd": str(source),
            "returncode": run.returncode,
            "stdout": name + ".stdout",
            "stderr": name + ".stderr",
        }
    )
    (out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    print(name, run.returncode, flush=True)
    if run.returncode:
        status = "REJECT_OR_FIXTURE_FAILURE"
        break
tracked = git(
    "ls-files",
    "src/r0/modules/priors",
    "src/r0/modules/explorer",
    "docs/issue-28-functional-graph-integration.md",
    "docs/specs/0028-causal-network-generators.md",
    "AGENTS.md",
    "docs/requirements.md",
).splitlines()
pins = {
    "expected_head": args.expected_head,
    "observed_head": git("rev-parse", "HEAD"),
    "source_root": str(source),
    "initial_status": initial_status,
    "final_status": git("status", "--short"),
    "python": args.python,
    "driver_python": sys.version,
    "platform": platform.platform(),
    "environment_overrides": {
        k: env[k]
        for k in (
            "PYTHONPATH",
            "PYTHONDONTWRITEBYTECODE",
            "R0_VERIFY_OUT",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
        )
    },
    "source_sha256": {f: hashlib.sha256((source / f).read_bytes()).hexdigest() for f in tracked},
}
(out / "source_pins.json").write_text(json.dumps(pins, indent=2) + "\n")
assert pins["observed_head"] == args.expected_head
assert pins["final_status"] == initial_status
(out / "verification_result.json").write_text(
    json.dumps(
        {
            "status": status,
            "scope": (
                "independent bounded authored functional graph verification; not "
                "full issue 28 acceptance"
            ),
            "programs_completed": len(commands),
            "elapsed_seconds": time.time() - started,
        },
        indent=2,
    )
    + "\n"
)
(out / "SHA256SUMS").write_text(
    "".join(
        hashlib.sha256(p.read_bytes()).hexdigest() + "  " + str(p.relative_to(out)) + "\n"
        for p in sorted(out.rglob("*"))
        if p.is_file() and p != out / "SHA256SUMS"
    )
)
print(status, flush=True)
sys.exit(0 if status == "PASS" else 1)
