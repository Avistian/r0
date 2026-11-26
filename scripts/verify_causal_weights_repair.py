#!/usr/bin/env python3
"""Archive exact revisions and independently check factorized-weight public contracts."""

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import traceback
from pathlib import Path

SCRIPTS = ("verify_causal_weights_repair.py", "causal_weight_cases.py", "causal_weight_legacy.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def worker(args):
    from causal_weight_legacy import snapshot

    import r0.modules.priors as priors

    root, out = Path(args.repo).resolve(), Path(args.output)
    assert Path(priors.__file__).resolve().is_relative_to(root / "src")
    report = {
        "status": "RUNNING",
        "checks": [],
        "root": str(root),
        "expected_head": args.expected_head,
        "interpreter": sys.executable,
        "python": sys.version,
    }

    def mark(name, **details):
        report["checks"].append({"name": name, "status": "PASS", **details})
        print("PASS:", name, flush=True)
        save(out / "report.json", report)

    try:
        if args.worker == "candidate":
            from causal_weight_cases import (
                authored_checks,
                explorer_checks,
                numerical_checks,
                rng_and_admission_checks,
                sampled_checks,
            )

            numerical_checks(mark, out)
            authored_checks(mark, out)
            sampled_checks(mark, out)
            rng_and_admission_checks(mark, out)
            explorer_checks(mark, out)
        records = snapshot(out)
        mark(
            "legacy capture preserves success and failure outcomes",
            records=len(records),
            failures=sum(r["status"] == "FAILED" for r in records),
        )
        report["status"] = "PASS"
        return 0
    except Exception:
        report["status"] = "REJECT_OR_HARNESS_ERROR"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], file=sys.stderr, flush=True)
        return 1
    finally:
        report["imports"] = {
            name: {"path": str(Path(module.__file__).resolve()), "sha256": sha(module.__file__)}
            for name, module in sorted(sys.modules.items())
            if name == "r0" or name.startswith("r0.")
            if getattr(module, "__file__", None)
        }
        assert all(Path(v["path"]).is_relative_to(root / "src") for v in report["imports"].values())
        report["scripts"] = {name: sha(Path(__file__).parent / name) for name in SCRIPTS}
        save(out / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("repo", "expected-head", "base", "output"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--worker", choices=("candidate", "baseline"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    repo, out = Path(args.repo).resolve(), Path(args.output).resolve()
    actual = git(repo, "rev-parse", "HEAD")
    if actual != args.expected_head:
        raise SystemExit("HEAD mismatch: " + actual)
    out.mkdir(parents=True, exist_ok=False)
    scripts = out / "verifier"
    scripts.mkdir()
    for name in SCRIPTS:
        source = Path(__file__).parent / name
        assert source.stat().st_size < 20_000, source
        shutil.copyfile(source, scripts / name)
    receipt = {
        "head_before": actual,
        "expected_head": args.expected_head,
        "base": git(repo, "rev-parse", args.base),
        "repo": str(repo),
        "status_before": git(repo, "status", "--short"),
        "argv": sys.argv,
        "scripts": {name: sha(scripts / name) for name in SCRIPTS},
        "archives": {},
        "workers": {},
    }
    for label, revision in (("candidate", actual), ("baseline", receipt["base"])):
        archive = subprocess.check_output(
            ["git", "-C", str(repo), "archive", "--format=tar", revision]
        )
        destination = out / (label + "_source")
        destination.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(destination, filter="data")
        receipt["archives"][label] = {
            "revision": revision,
            "tar_sha256": hashlib.sha256(archive).hexdigest(),
            "path": str(destination),
        }
        save(
            out / (label + "_source_hashes.json"),
            {
                str(p.relative_to(destination)): sha(p)
                for p in sorted((destination / "src").rglob("*.py"))
            },
        )
    save(out / "receipt.json", receipt)
    env = {
        **os.environ,
        "UV_PROJECT_ENVIRONMENT": str(
            Path(os.environ.get("UV_PROJECT_ENVIRONMENT", str(repo / ".venv"))).resolve()
        ),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for label in ("candidate", "baseline"):
        source = out / (label + "_source")
        evidence = out / label
        evidence.mkdir()
        command = [
            sys.executable,
            str(source / "scripts/run_check.py"),
            "python",
            "-B",
            str(scripts / SCRIPTS[0]),
            "--worker",
            label,
            "--repo",
            str(source),
            "--expected-head",
            receipt["archives"][label]["revision"],
            "--base",
            receipt["base"],
            "--output",
            str(evidence),
        ]
        receipt["workers"][label] = {
            "command": command,
            "wrapper_sha256": sha(source / "scripts/run_check.py"),
        }
        save(out / "receipt.json", receipt)
        with (
            (evidence / "stdout.log").open("w") as stdout,
            (evidence / "stderr.log").open("w") as stderr,
        ):
            result = subprocess.run(
                command, cwd=source, env=env, stdout=stdout, stderr=stderr, check=False
            )
        receipt["workers"][label]["returncode"] = result.returncode
        if result.returncode:
            receipt["status"] = "REJECT_OR_HARNESS_ERROR"
            break
    else:
        candidate = json.loads((out / "candidate/legacy.json").read_text())
        baseline = json.loads((out / "baseline/legacy.json").read_text())
        receipt["legacy_comparison"] = {
            "records": len(candidate),
            "equal": candidate == baseline,
            "failed_cases": sum(r["status"] == "FAILED" for r in candidate),
        }
        receipt["status"] = "PASS" if candidate == baseline else "REJECT"
        if candidate != baseline:
            save(
                out / "legacy_differences.json",
                [
                    {"candidate": c, "baseline": b}
                    for c, b in zip(candidate, baseline, strict=True)
                    if c != b
                ],
            )
    receipt["head_after"] = git(repo, "rev-parse", "HEAD")
    receipt["status_after"] = git(repo, "status", "--short")
    assert receipt["head_after"] == actual
    assert receipt["status_after"] == receipt["status_before"]
    save(out / "receipt.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("status", "head_after")}, indent=2))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
