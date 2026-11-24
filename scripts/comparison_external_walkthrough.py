"""Exercise learner/TabPFN roles using explicit fixture subprocesses and saved ingestion."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from r0.modules.comparison import ExternalPredictor, attempt, compare, environment, prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cgroup-root", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/comparison_fixture_backend.py"
    runtime = environment()
    runtime["thread_environment"] = runtime["required_thread_environment"]
    identity = {
        "implementation": "authored-mean-frequency-backend-v1",
        "environment": runtime,
        "settings": {"algorithm": "context-mean-or-class-frequency"},
        "checkpoint": "FIXTURE_NO_CHECKPOINT",
        "evidence": "FIXTURE_ONLY",
    }
    identity_path = args.out.resolve() / "identity.json"
    identity_path.write_text(json.dumps({"identity": identity}, indent=2) + "\n")
    rows: list[dict[str, Any]] = []
    for task in ("binary", "multiclass", "regression"):
        prepared = args.out / f"{task}-prepared"
        prepare(root / f"docs/evidence/issue-26/fixtures/{task}.json", prepared)
        learner = ExternalPredictor("learner", Path(sys.executable), identity_path, script)
        tabpfn = ExternalPredictor("tabpfn", Path(sys.executable), identity_path, script)
        saved = [args.out / f"{task}-tabpfn-{draw}" for draw in (1, 2, 3)]
        for draw, path in enumerate(saved, 1):
            result = attempt(
                prepared,
                path,
                tabpfn,
                draw,
                diagnostic=args.cgroup_root is None,
                cgroup_root=args.cgroup_root,
            )
            assert result["status"] == "SUCCESS", result
        report = compare(
            prepared,
            args.out / f"{task}-results",
            external_predictors=[learner, tabpfn],
            saved_attempts={"tabpfn": saved},
            diagnostic=args.cgroup_root is None,
            cgroup_root=args.cgroup_root,
        )
        assert report["comparison"]["status"] == "COMPLETE", report
        assert report["status"] == "INCOMPLETE"
        rows.append(
            {
                "task": task,
                "evidence": "FIXTURE_ONLY",
                "comparison": report["comparison"],
                "resource_enforcement": report["resource_enforcement"],
            }
        )
    (args.out / "external-walkthrough.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("PASS: three task types, separate subprocesses, saved result ingestion, fixture gaps")


if __name__ == "__main__":
    main()
