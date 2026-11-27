"""Qualify retained census evidence by comparing every newly sampled recipe identity.

This does not regenerate task rows or count as a fresh generation census.
"""

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import r0
from r0.modules.priors import LinearRecipe, sample_recipe


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def run(headers, output):
    output.mkdir(parents=True, exist_ok=False)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    outcomes = []
    for profile in ("shared", "refreshed", "local", "global"):
        files = sorted(headers.glob(f"0028-shared-headers-{profile}-*.json"))
        if len(files) != 8:
            raise ValueError(f"Expected all eight header parts for {profile}")
        entries = [row for file in files for row in json.loads(file.read_text())]
        expected = {
            (family, task_id)
            for family in ("forward", "layered", "redirection")
            for task_id in range(64)
        }
        if len(entries) != 192 or {(r["family"], r["task_id"]) for r in entries} != expected:
            raise ValueError(f"Missing or duplicate census identities for {profile}")
        for row in entries:
            task_id = row["task_id"]
            kind = ("regression", "binary", "multiclass")[task_id % 3]
            controls = LinearRecipe(
                rows=16,
                context_rows=8,
                features=4,
                active_features=4,
                signal_features=4,
                task_type=kind,
                classes={"regression": 0, "binary": 2, "multiclass": 3}[kind],
            )
            recipe = sample_recipe(
                row["family"],
                task=controls,
                seed=row["seed"],
                task_id=task_id,
                profile=f"causal-{profile}-matrices-v1",
            )
            actual = digest(json.dumps(asdict(recipe), sort_keys=True, allow_nan=False).encode())
            outcome = {
                "profile": profile,
                "family": row["family"],
                "task_id": task_id,
                "original_generation_status": row["status"],
                "expected_recipe_sha256": row["recipe_sha256"],
                "actual_recipe_sha256": actual,
                "equal": actual == row["recipe_sha256"],
            }
            outcomes.append(outcome)
            with (output / "outcomes.jsonl").open("a") as stream:
                stream.write(json.dumps(outcome, sort_keys=True) + "\n")
            if not outcome["equal"]:
                raise ValueError(f"Recipe identity mismatch: {profile}/{row['family']}/{task_id}")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() != head:
        raise ValueError("Source HEAD changed during identity qualification")
    receipt = {
        "status": "PASS_RECIPE_IDENTITY_ONLY",
        "head": head,
        "imported_r0": r0.__file__,
        "records": len(outcomes),
        "equal": sum(row["equal"] for row in outcomes),
        "prior_generation_failures": sum(
            row["original_generation_status"] == "FAILED" for row in outcomes
        ),
        "program_sha256": digest(Path(__file__).read_bytes()),
        "outcomes_sha256": digest((output / "outcomes.jsonl").read_bytes()),
        "source_diff_from_census": subprocess.check_output(
            ["git", "diff", "--name-only", "47630e8", head, "--", "src"], text=True
        ).splitlines(),
        "not_run": ["fresh task generation", "browser", "Colab", "GPU"],
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("headers", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.headers, args.output)
