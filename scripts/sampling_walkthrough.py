"""Retain sampled settings, generated examples, replay and graph checks for both profiles."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from r0.modules.priors import (
    LinearRecipe,
    explain_generation,
    generate_task,
    recipe_from_dict,
    sample_recipe,
)

PROFILES = {
    "broad-v1": (
        "linear",
        "grouped",
        "polynomial",
        "quadratic",
        "rff",
        "periodic",
        "radial",
        "sinusoidal",
        "gp",
        "composition",
    ),
    "tabiclv2-adapted-v1": ("power", "quadratic", "rff", "product"),
}


def depth(expression):
    if isinstance(expression, int):
        return 0
    return 1 + max(map(depth, expression["children"]))


def main(out: Path):
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "src"], check=False).returncode:
        raise ValueError("commit source before retaining revision-labelled evidence")
    out.mkdir(parents=True, exist_ok=False)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    coverage, results = [], []
    for profile, families in PROFILES.items():
        for family in families:
            controls = LinearRecipe(rows=32, context_rows=20)
            histograms = {}
            with (out / f"{profile}-{family}-settings.jsonl").open("x") as stream:
                for task_id in range(128):
                    recipe = sample_recipe(
                        family, task=controls, seed=27, task_id=task_id, profile=profile
                    )
                    fields = asdict(recipe)
                    request = {
                        "profile": profile,
                        "family": family,
                        "task": asdict(controls),
                        "seed": 27,
                        "task_id": task_id,
                    }
                    stream.write(json.dumps({**request, "recipe": fields}, allow_nan=False) + "\n")
                    assert recipe_from_dict(fields) == recipe
                    assert (
                        sample_recipe(
                            family, task=controls, seed=27, task_id=task_id, profile=profile
                        )
                        == recipe
                    )
                    observed = {
                        key: fields[key]
                        for key in (
                            "degree",
                            "terms",
                            "rank",
                            "group_size",
                            "projection",
                            "combine",
                            "calibration_rows",
                        )
                        if key in fields
                    }
                    observed["input_distribution"] = fields.get("inputs", {}).get(
                        "distribution", "gaussian"
                    )
                    if "components" in fields:
                        observed.update(
                            components=len(fields["components"]), depth=depth(fields["expression"])
                        )
                    if "kernels" in fields:
                        observed["kernels"] = len(fields["kernels"])
                    for key, value in observed.items():
                        histograms.setdefault(key, Counter())[str(value)] += 1
            coverage.append(
                {
                    "profile": profile,
                    "family": family,
                    "settings_draws": 128,
                    "histograms": histograms,
                }
            )
            for kind, classes in (("regression", 0), ("binary", 2), ("multiclass", 3)):
                task_controls = LinearRecipe(
                    rows=32, context_rows=20, task_type=kind, classes=classes
                )
                recipe = sample_recipe(
                    family, task=task_controls, seed=31, task_id=0, profile=profile
                )
                name = f"{profile}-{family}-{kind}"
                record = {
                    "name": name,
                    "profile": profile,
                    "family": family,
                    "seed": 31,
                    "task_id": 0,
                    "recipe": asdict(recipe),
                }
                (out / f"{name}-request.json").write_text(json.dumps(record, indent=2) + "\n")
                try:
                    task = generate_task(recipe, seed=31, task_id=0)
                    graph = explain_generation(task, [0, 1, 3])
                    assert graph["signal_max_error"] <= 1e-10
                    assert graph["target_max_error"] <= 1e-10
                    replay = generate_task(recipe_from_dict(record["recipe"]), seed=31, task_id=0)
                    assert task.episode() == replay.episode()
                    assert task.mechanism_json == replay.mechanism_json
                    assert bool(torch.isfinite(task.scores).all())
                    (out / f"{name}-card.json").write_text(
                        json.dumps(
                            {
                                "recipe": json.loads(task.recipe_json),
                                "mechanism": json.loads(task.mechanism_json),
                                "graph": graph,
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    np.savez(
                        out / f"{name}.npz",
                        inputs=task.inputs.numpy(),
                        scores=task.scores.numpy(),
                        targets=task.targets.numpy(),
                    )
                    results.append(
                        {
                            "name": name,
                            "status": "PASS",
                            "source_task_id": task.source_task_id,
                            "episode_id": task.episode_id,
                            "graph_signal_max_error": graph["signal_max_error"],
                            "graph_target_max_error": graph["target_max_error"],
                        }
                    )
                except (ValueError, RuntimeError, AssertionError) as error:
                    results.append({"name": name, "status": "FAILED", "error": repr(error)})
                (out / f"{name}-result.json").write_text(json.dumps(results[-1], indent=2) + "\n")
    summary = {
        "source": source,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "coverage": coverage,
        "examples": results,
        "successful": sum(r["status"] == "PASS" for r in results),
        "failed": sum(r["status"] == "FAILED" for r in results),
        "bulk_eligibility": "NOT_MEASURED",
        "hosted_execution": "NOT_RUN",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key not in ("coverage", "examples")}
        )
    )
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out)
