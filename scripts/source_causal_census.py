"""Unfiltered authored census of source causal profiles, not a throughput benchmark."""

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from r0.modules.generation import save_group
from r0.modules.priors import LinearRecipe, SourceCausalRecipe, generate_task, sample_recipe


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run(output, profile):
    output.mkdir(parents=True, exist_ok=False)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    report = {
        "head": head,
        "profile": profile,
        "seed": 28,
        "filter": "none",
        "timing": "NOT_MEASURED",
        "peak_rss": "NOT_MEASURED",
        "families": {},
        "program_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    for family in ("forward", "layered", "redirection"):
        records, tasks = [], []
        for task_id in range(64):
            kind = ("regression", "binary", "multiclass")[task_id % 3]
            controls = LinearRecipe(
                rows=16,
                context_rows=8,
                features=4,
                signal_features=4,
                active_features=4,
                task_type=kind,
                classes={"regression": 0, "binary": 2, "multiclass": 3}[kind],
            )
            entry = {
                "family": family,
                "profile": profile,
                "seed": 28,
                "task_id": task_id,
                "task": asdict(controls),
                "phase": "sampling",
            }
            try:
                recipe = sample_recipe(
                    family, task=controls, seed=28, task_id=task_id, profile=profile
                )
                assert isinstance(recipe, SourceCausalRecipe)
                entry.update(recipe=asdict(recipe), phase="generation")
                task = generate_task(recipe, seed=28, task_id=task_id)
                card = json.loads(task.mechanism_json)
                std = [float(x) for x in task.scores.std(dim=0, unbiased=False)]
                entry.update(
                    status="VALID",
                    flags=card["flags"],
                    source_task_id=task.source_task_id,
                    score_std=[v if math.isfinite(v) else str(v) for v in std],
                )
                tasks.append((task_id, task))
            except Exception as error:
                entry.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            records.append(entry)
            # Incremental raw retention, including every failure, without retries.
            with (output / f"{family}.jsonl").open("a") as handle:
                handle.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")
        if tasks:
            save_group(output / f"{family}.zip", tasks, max_bytes=32 * 1024**2)
        valid = [r for r in records if r["status"] == "VALID"]
        selected = [
            s
            for r in records
            for s in r.get("recipe", {}).get("source_selections", {}).get("selections", [])
        ]
        prior_records = [r["recipe"]["source_prior"] for r in records if "recipe" in r]
        report["families"][family] = {
            "attempted": len(records),
            "valid": len(valid),
            "failed": len(records) - len(valid),
            "failures": [
                {k: r[k] for k in ("task_id", "phase", "error")}
                for r in records
                if r["status"] == "FAILED"
            ],
            "missing_complete_recipes": sum("recipe" not in r for r in records),
            "task_types": dict(Counter(r["task"]["task_type"] for r in records)),
            "flags": dict(Counter(f for r in valid for f in r["flags"])),
            "placements": dict(Counter(p["placement"] for p in prior_records)),
            "noise_after_probabilities": dict(
                Counter(str(p["noise_after_probability"]) for p in prior_records)
            ),
            "primitive_kinds": dict(Counter(s["activation"]["kind"] for s in selected)),
            "stage_bindings": sum(
                len(r.get("recipe", {}).get("source_bindings", ())) for r in records
            ),
            "rescale_draws": sum(
                len(r.get("recipe", {}).get("rescale_preferences", {}).get("draws", ()))
                for r in records
            ),
        }
        dump(output / "summary.json", report)
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == head
    print(
        json.dumps(
            {
                "profile": profile,
                "outcomes": {
                    k: (v["valid"], v["attempted"]) for k, v in report["families"].items()
                },
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()
    run(args.output, args.profile)
