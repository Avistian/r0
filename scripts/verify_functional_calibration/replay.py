"""Run the bounded independent target-calibration programs without installs or sync."""

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if actual != args.expected_head:
        raise ValueError(f"source HEAD differs: {actual} != {args.expected_head}")
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True)
    if any(line != "?? .venv" for line in status.splitlines()):
        raise ValueError("source checkout must be clean apart from the environment symlink")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    programs = Path(__file__).resolve().parent
    for name in ("probe.py", "supplement.py", "v1_snapshot.py"):
        shutil.copy2(programs / name, output / name)
    baseline = "6a70ecd2289ff4e9c208a0d7f0d09acebe5ada53"
    raw = subprocess.check_output(["git", "archive", baseline, "src"], cwd=source)
    baseline_dir = output / "baseline-source"
    baseline_dir.mkdir()
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        archive.extractall(baseline_dir, filter="data")
    (output / "baseline-source.tar").write_bytes(raw)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(source / "src")}
    steps = [
        ("probe", [str(output / "probe.py")]),
        ("v1-base", [str(output / "v1_snapshot.py"), str(baseline_dir), str(output / "v1-base")]),
        (
            "v1-candidate",
            [str(output / "v1_snapshot.py"), str(source), str(output / "v1-candidate")],
        ),
        ("supplement", [str(output / "supplement.py")]),
    ]
    results = []
    for name, arguments in steps:
        command = [sys.executable, *arguments]
        with (
            (output / f"{name}.stdout").open("w") as stdout,
            (output / f"{name}.stderr").open("w") as stderr,
        ):
            process = subprocess.run(
                command, cwd=source, env=env, stdout=stdout, stderr=stderr, check=False
            )
        results.append({"name": name, "command": command, "returncode": process.returncode})
        (output / "replay-results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(name, process.returncode, flush=True)
        if process.returncode:
            raise SystemExit(process.returncode)
    pins = {
        "source": actual,
        "baseline": baseline,
        "python": sys.version,
        "baseline_archive_sha256": hashlib.sha256(raw).hexdigest(),
    }
    (output / "source-pins.json").write_text(json.dumps(pins, indent=2) + "\n")
    manifest = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(output)}\n"
        for path in sorted(output.rglob("*"))
        if path.is_file()
    )
    (output / "SHA256SUMS").write_text(manifest)


if __name__ == "__main__":
    main()
