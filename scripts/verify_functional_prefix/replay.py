"""Run the bounded independent ancestral-prefix programs without installs or sync."""

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
    for name in ("verify.py",):
        shutil.copy2(programs / name, output / name)
    baseline = "09cdbbab88de5cc9b04702bb7433aea47f8ae160"
    raw = subprocess.check_output(["git", "archive", baseline, "src"], cwd=source)
    baseline_dir = output / "baseline-source"
    baseline_dir.mkdir()
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        archive.extractall(baseline_dir, filter="data")
    (output / "baseline-source.tar").write_bytes(raw)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(source / "src")}
    steps = [
        (
            "base",
            [str(output / "verify.py"), str(baseline_dir / "src"), str(output / "base-results")],
        ),
        (
            "candidate",
            [str(output / "verify.py"), str(source / "src"), str(output / "candidate-results")],
        ),
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
    baseline_files = sorted((output / "base-results").glob("*.json"))
    if len(baseline_files) != 24:
        raise ValueError("baseline must produce all 24 graph snapshots")
    for path in baseline_files:
        if path.read_bytes() != (output / "candidate-results" / path.name).read_bytes():
            raise ValueError(f"baseline snapshot differs: {path.name}")
    if not (output / "candidate-results" / "prefix-results.json").is_file():
        raise ValueError("candidate prefix checks did not produce their results")
    print("PASS 24 byte-identical graph snapshots and candidate prefix checks", flush=True)
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
