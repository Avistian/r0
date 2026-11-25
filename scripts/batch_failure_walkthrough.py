"""Retain saved-task failure identity and reject incomplete or over-counted receipts."""

import argparse
import json
import runpy
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

from r0.modules.generation import load_group, save_group
from r0.modules.priors import LinearRecipe, generate_task


def forbidden(*args, **kwargs):
    raise AssertionError("saved consumption called generate_task")


def main(out: Path, saved: Path):
    out.mkdir(parents=True, exist_ok=False)
    scripts = Path(__file__).parent
    archive = out / "oversized.zip"
    save_group(
        archive,
        (
            (i, generate_task(LinearRecipe(rows=rows, context_rows=20), seed=92, task_id=i))
            for i, rows in enumerate((32, 256, 32))
        ),
        max_bytes=1024 * 1024,
    )
    expected = list(load_group(archive))[1]
    sys.argv = [
        "consume_generated_batches.py",
        "--archive",
        str(archive),
        "--out",
        str(out / "failed"),
    ]
    with patch("r0.modules.priors.generate_task", side_effect=forbidden):
        try:
            runpy.run_path(str(scripts / "consume_generated_batches.py"), run_name="__main__")
        except ValueError as error:
            if "at most 128 rows" not in str(error):
                raise
        else:
            raise AssertionError("oversized saved task was accepted")
    failure = json.loads((out / "failed" / "failure.json").read_text())
    assert failure["status"] == "FAILED"
    assert failure["attempt"]["source_task_id"] == expected.source_task_id
    assert failure["attempt"]["recipe"] == json.loads(expected.recipe_json)
    assert failure["produced"] == 3
    assert failure["consumed"] == 0  # The incomplete batch did not update the optimizer.
    assert not (out / "failed" / "summary.json").exists()
    verify = runpy.run_path(str(scripts / "verify_generated_batches.py"))["main"]
    results = {"saved_failure_identity": "PASS"}
    for name in ("premature-stop", "excess-consumption"):
        target = out / name
        shutil.copytree(saved, target)
        (target / "verified.json").unlink(missing_ok=True)
        summary = json.loads((target / "summary.json").read_text())
        lines = (target / "batches.jsonl").read_text().splitlines(keepends=True)
        if name == "premature-stop":
            first = json.loads(lines[0])
            summary.update(
                produced=4, consumed=4, batches=1, final_log_bandwidth=first["parameter_after"]
            )
            (target / "batches.jsonl").write_text(lines[0])
        else:
            summary["requested_episodes"] = 6
        (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        try:
            verify(target, None, target / "verified.json")
        except AssertionError:
            assert not (target / "verified.json").exists()
            results[name] = "REJECTED_AS_EXPECTED"
        else:
            raise AssertionError(f"verifier accepted {name}")
    (out / "verified.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--saved", type=Path, required=True)
    args = parser.parse_args()
    main(args.out, args.saved)
