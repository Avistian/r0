"""Retain pinned import identities and compare bounded legacy outcomes without filtering."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from r0.modules import priors


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    args.output.mkdir(parents=True)
    source_root = Path(priors.__file__).resolve().parents[4]
    tracked = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", args.commit, "--", "src"],
        cwd=args.repo,
        text=True,
    ).splitlines()
    source_files = []
    for name in tracked:
        expected = subprocess.check_output(["git", "show", args.commit + ":" + name], cwd=args.repo)
        actual = source_root / name
        source_files.append(
            {
                "path": name,
                "actual_path": str(actual),
                "sha256": sha(actual),
                "pinned_sha256": hashlib.sha256(expected).hexdigest(),
            }
        )
        assert source_files[-1]["sha256"] == source_files[-1]["pinned_sha256"]
    imports = []
    for name, module in sorted(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if name.startswith("r0.") and file:
            path = Path(file).resolve()
            assert path.is_relative_to(source_root)
            imports.append({"module": name, "path": str(path), "sha256": sha(path)})
    base = json.loads((args.baseline / "legacy.json").read_text())
    candidate = json.loads((args.runs / "legacy/legacy.json").read_text())
    assert len(base) == len(candidate) == 504
    mismatches = []
    for before, after in zip(base, candidate, strict=True):
        name = before["key"] + ".json"
        if before != after or (
            json.loads((args.baseline / "legacy_cases" / name).read_text())
            != json.loads((args.runs / "legacy/legacy_cases" / name).read_text())
        ):
            mismatches.append(before["key"])
    assert not mismatches
    checks = {}
    for name, report in (
        ("numeric", "summary.json"),
        ("matrices", "summary.json"),
        ("boundaries", "summary.json"),
        ("extreme", "result.json"),
    ):
        raw = json.loads((args.runs / name / report).read_text())
        assert raw["status"] == "PASS"
        checks[name] = {
            "report": str(args.runs / name / report),
            "sha256": sha(args.runs / name / report),
            "status": raw["status"],
            "probes": raw.get("probes", len(raw.get("outcomes", []))),
        }
    # Tight independent check of the retained Decimal oracle, without re-generating.
    extreme = json.loads((args.runs / "extreme/result.json").read_text())
    log_errors = [
        abs(a - b)
        for probe in extreme["outcomes"]
        for a, b in zip(probe["log_probabilities"], probe["independent_decimal_logs"], strict=True)
    ]
    assert max(log_errors) <= 2e-13
    scripts = {
        p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
        for p in args.runs.glob("shared_preference_*.py")
    }
    assert all(v["bytes"] < 20_000 for v in scripts.values())
    result = {
        "status": "PASS_SCOPED_REPAIR",
        "commit": args.commit,
        "python": sys.executable,
        "source_root": str(source_root),
        "source_archive_sha256": sha(args.runs / "source.tar"),
        "pinned_source_files": source_files,
        "actual_imports": imports,
        "scripts": scripts,
        "runs": checks,
        "legacy": {
            "cases": 504,
            "exact_matches": 504,
            "mismatches": mismatches,
            "base_failed": sum(r["status"] == "FAILED" for r in base),
            "candidate_failed": sum(r["status"] == "FAILED" for r in candidate),
            "baseline": str(args.baseline),
        },
        "maximum_log_probability_absolute_error": max(log_errors),
        "historical_reject": "47630e8: constructor and decoder rejected a valid extreme law",
        "limitations": [
            "No whole-issue acceptance",
            "No browser, Colab, GPU or paid execution",
            "Parent owns full-suite checks and census identity qualification",
            "Controlled transformed-matrix arithmetic uses identity activation",
        ],
    }
    (args.output / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "legacy": result["legacy"],
                "source_files": len(source_files),
                "imports": len(imports),
                "maximum_log_error": max(log_errors),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
