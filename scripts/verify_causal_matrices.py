"""Run portable public-contract oracles against frozen git archives, retaining raw evidence."""

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

CANDIDATE = "7eea0bd5ab6f553aa07f014ed5a353b4d0db69e9"
BASE = "9871fdaf7037c7ed9b09926f5073f63f5cf6ff31"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--candidate", default=CANDIDATE)
    parser.add_argument("--base", default=BASE)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    programs = Path(__file__).resolve().parent
    receipt = {"python": sys.executable, "commands": [], "archives": {}, "programs": {}}
    for name in (
        "verify_causal_matrices.py",
        "causal_matrix_numeric.py",
        "causal_matrix_cases.py",
        "causal_matrix_legacy.py",
    ):
        file = programs / name
        assert file.stat().st_size < 20000
        receipt["programs"][name] = {"sha256": sha(file), "bytes": file.stat().st_size}
    for name, revision in (("candidate", args.candidate), ("base", args.base)):
        head = subprocess.check_output(
            ["git", "-C", str(args.repo), "rev-parse", revision], text=True
        ).strip()
        command = ["git", "-C", str(args.repo), "archive", head]
        receipt["commands"].append(command)
        raw = subprocess.check_output(command)
        archive = out / (name + ".tar")
        archive.write_bytes(raw)
        archive.chmod(0o444)
        folder = out / name
        folder.mkdir()
        with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
            tar.extractall(folder, filter="data")
        for file in folder.rglob("*"):
            if file.is_file():
                file.chmod(0o444)
        for directory in sorted((p for p in folder.rglob("*") if p.is_dir()), reverse=True):
            directory.chmod(0o555)
        folder.chmod(0o555)
        receipt["archives"][name] = {"head": head, "path": str(folder), "tar_sha256": sha(archive)}

    def run(name, program, source):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(out / source / "src")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        command = [sys.executable, str(programs / program), str(out / name)]
        receipt["commands"].append({"argv": command, "PYTHONPATH": env["PYTHONPATH"]})
        with (
            (out / (name + ".stdout")).open("w") as stdout,
            (out / (name + ".stderr")).open("w") as stderr,
        ):
            process = subprocess.run(
                command, env=env, cwd=out, stdout=stdout, stderr=stderr, check=False
            )
        if process.returncode:
            receipt["status"] = "REJECT_OR_HARNESS_ERROR"
            receipt["failed_command"] = command
            (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            raise SystemExit(process.returncode)

    run("numeric", "causal_matrix_numeric.py", "candidate")
    run("cases", "causal_matrix_cases.py", "candidate")
    run("legacy-base", "causal_matrix_legacy.py", "base")
    run("legacy-candidate", "causal_matrix_legacy.py", "candidate")
    old = json.loads((out / "legacy-base/legacy.json").read_text())
    new = json.loads((out / "legacy-candidate/legacy.json").read_text())
    receipt["legacy"] = {
        "equal": old == new,
        "cases": len(old),
        "failed_draws": sum(r["status"] == "FAILED" for r in old),
        "base_sha256": sha(out / "legacy-base/legacy.json"),
        "candidate_sha256": sha(out / "legacy-candidate/legacy.json"),
    }
    receipt["numeric"] = json.loads((out / "numeric/numeric-summary.json").read_text())
    receipt["cases"] = json.loads((out / "cases/workflow-summary.json").read_text())
    receipt["status"] = "PASS" if old == new else "REJECT"
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "legacy": receipt["legacy"],
                "numeric_checks": receipt["numeric"]["checks"],
                "case_checks": receipt["cases"]["checks"],
            }
        )
    )
    if old != new:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
