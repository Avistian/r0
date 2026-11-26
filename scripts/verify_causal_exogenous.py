"""Run retained public-interface probes against isolated, exact Git archives."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import traceback
from pathlib import Path


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(repo, revision, dest):
    dest.mkdir()
    with tempfile.TemporaryFile() as stream:
        subprocess.run(["git", "-C", str(repo), "archive", revision], stdout=stream, check=True)
        stream.seek(0)
        with tarfile.open(fileobj=stream, mode="r:") as bundle:
            bundle.extractall(dest, filter="data")


def worker(probes, snapshot, output, extra):
    # -I drops ambient PYTHONPATH, cwd and user site; prepend only this exact archive.
    code = (
        "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));"
        'p=sys.argv.pop(1);sys.argv[0]=p;runpy.run_path(p,run_name="__main__")'
    )
    command = [
        sys.executable,
        "-I",
        "-c",
        code,
        str(snapshot / "src"),
        str(probes),
        str(output),
        *extra,
    ]
    environment = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        command, cwd=snapshot, env=environment, capture_output=True, text=True, check=False
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "report": json.loads(output.read_text()) if output.exists() else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; refusing to replace retained evidence")
    repo = args.repo.resolve()
    probes = Path(__file__).with_name("verify_causal_exogenous_probes.py").resolve()
    result = {
        "status": "RUNNING",
        "command": [sys.executable, *sys.argv],
        "repo": str(repo),
        "expected_head": args.expected_head,
        "base": args.base,
        "python": sys.version,
        "scripts": {
            str(Path(__file__).resolve()): digest(Path(__file__)),
            str(probes): digest(probes),
            str(probes.with_name("verify_causal_exogenous_repair.py")): digest(
                probes.with_name("verify_causal_exogenous_repair.py")
            ),
        },
    }
    try:
        actual = git(repo, "rev-parse", "HEAD")
        expected = git(repo, "rev-parse", args.expected_head + "^{commit}")
        base = git(repo, "rev-parse", args.base + "^{commit}")
        result.update(actual_head=actual, expected_head=expected, base=base)
        if actual != expected:
            raise RuntimeError(f"HEAD mismatch: expected {expected}, got {actual}")
        result["initial_status"] = git(repo, "status", "--short")
        if git(repo, "diff", "--name-only", expected, "--", "src", "tests", "scripts"):
            raise RuntimeError("tracked source/tests/scripts differ from expected revision")
        with tempfile.TemporaryDirectory(prefix="r0-exogenous-oracle-") as temp:
            folder = Path(temp)
            baseline, candidate = folder / "base", folder / "candidate"
            archive(repo, base, baseline)
            archive(repo, expected, candidate)
            before = worker(probes, baseline, folder / "baseline.json", ["--legacy-only"])
            result["baseline_execution"] = before
            if before["returncode"]:
                raise RuntimeError("baseline worker failed")
            after = worker(
                probes,
                candidate,
                folder / "candidate.json",
                ["--legacy-baseline", str(folder / "baseline.json")],
            )
            result["candidate_execution"] = after
            for execution, snapshot in ((before, baseline), (after, candidate)):
                report = execution["report"]
                imported = Path(report["public_import_path"]).resolve()
                if not imported.is_relative_to(snapshot / "src"):
                    raise RuntimeError(f"wrong source import: {imported}")
                if report["script_sha256"] != digest(probes):
                    raise RuntimeError("probe script changed during execution")
            result["status"] = "PASS" if after["returncode"] == 0 else "REJECT"
        result["final_actual_head"] = git(repo, "rev-parse", "HEAD")
        if result["final_actual_head"] != expected:
            raise RuntimeError("HEAD changed during verification")
        result["final_status"] = git(repo, "status", "--short")
    except Exception as error:
        result["status"] = "HARNESS_ERROR"
        result["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"status": result["status"], "output": str(args.output)}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
