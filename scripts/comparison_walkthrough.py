"""Retain fixture-only comparisons and resource probe receipts; never load real assets."""

import argparse
import json
import sys
from pathlib import Path

from r0.modules.comparison import compare, prepare, run_process


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cgroup-root", type=Path, default=Path("/sys/fs/cgroup"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    fixtures = Path(__file__).resolve().parents[1] / "docs/evidence/issue-26/fixtures"
    rows: list[dict[str, object]] = []
    for task in ("binary", "multiclass", "regression"):
        prepared = args.out / f"{task}-prepared"
        manifest = prepare(fixtures / f"{task}.json", prepared)
        report = compare(
            prepared, args.out / f"{task}-results", diagnostic=True, fixture_predictors=True
        )
        rows.append(
            {
                "task": task,
                "query_id": manifest["query_ids"][0],
                "fixture_checks": report["fixture_checks"],
                "status": report["status"],
                "summaries": report["summaries"],
            }
        )
    probes: dict[str, object] = {}
    limits = {
        "cpu_max": "100000 100000",
        "threads": 1,
        "memory_max": 64 * 1024**2,
        "swap_max": 0,
        "wall_seconds": 1,
    }
    for name, code in (
        ("timeout", "import time; time.sleep(10)"),
        ("memory", "x=bytearray(128*1024**2); print(len(x))"),
    ):
        try:
            probes[name] = run_process(
                [sys.executable, "-c", code],
                cwd=args.out,
                cgroup_root=args.cgroup_root,
                limits=limits,
            )
        except ValueError as error:
            probes[name] = {"status": "NOT_RUN", "reason": str(error)}
    receipt = {
        "evidence": "AUTHORED_FIXTURE_UNENFORCED",
        "comparisons": rows,
        "resource_probes": probes,
        "real_tabpfn": "NOT_RUN",
        "real_learner": "NOT_RUN",
    }
    (args.out / "walkthrough.json").write_text(json.dumps(receipt, indent=2) + "\n")
    for row in rows:
        print(row["task"], row["fixture_checks"], row["status"], row["query_id"])
    print(json.dumps(probes, indent=2))
    if any(row["fixture_checks"] != "PASS" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
