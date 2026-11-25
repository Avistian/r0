"""Audit sampled-suite saved tasks and distinguish selected from completed diversity."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch

from r0.modules.generation import load_group
from r0.modules.priors import recipe_from_dict, sample_recipe


def depth(expression):
    if isinstance(expression, int):
        return 0
    return 1 + max(depth(child) for child in expression["children"])


def fields(recipe):
    """Top-level settings; composed child families are counted separately below."""
    task = recipe.get("task", recipe)
    values = {
        key: task[key]
        for key in (
            "rows",
            "features",
            "active_features",
            "context_rows",
            "classes",
            "noise_ratio",
            "separation",
        )
    }
    values["class_offsets"] = json.dumps(task["class_offsets"])
    inputs = recipe.get("inputs", {})
    values.update(
        {
            "input_distribution": inputs.get("distribution", "gaussian"),
            "input_location": inputs.get("location", 0),
            "input_scale": inputs.get("scale", 1),
            "input_bound": inputs.get("bound"),
        }
    )
    if inputs.get("distribution") == "student_t":
        values["student_t_df"] = inputs["df"]
    for key in (
        "degree",
        "rank",
        "terms",
        "group_size",
        "projection",
        "combine",
        "calibration_rows",
        "interaction_strength",
        "linear_strength",
        "length_scale",
    ):
        if key in recipe:
            values[key] = recipe[key]
    if "kernels" in recipe:
        values["kernel_count"] = len(recipe["kernels"])
    if "components" in recipe:
        values["component_count"] = len(recipe["components"])
        values["expression_depth"] = depth(recipe["expression"])
    return values


def audit(root, output):
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "src", "scripts"], check=True)
    manifest = json.loads((root / "manifest.json").read_text())
    report = json.loads((root / "report.json").read_text())
    assert report["source"] == manifest["source"]
    assert len({case["id"] for case in manifest["cases"]}) == len(manifest["cases"])
    for name, expected in manifest["drivers"].items():
        committed = subprocess.check_output(["git", "show", f"{manifest['source']}:scripts/{name}"])
        assert hashlib.sha256(committed).hexdigest() == expected
    identities, counts, families, failures = set(), Counter(), {}, []
    confirmed_saved = 0
    for case in manifest["cases"]:
        directory = root / case["id"]
        request = json.loads((directory / "request.json").read_text())
        assert request == case["request"]
        selection = dict(case["selection"])
        selection["task"] = recipe_from_dict(selection["task"])
        selected = json.loads(json.dumps(asdict(sample_recipe(**selection))))
        assert selected == request["recipe"], case["id"]
        result = json.loads((directory / "result.json").read_text())
        measurement = result["measurement"]
        saved = measurement is not None and measurement["summary"] == {
            "attempted": 1,
            "completed": 1,
            "failed": 0,
            "pending": 0,
            "status": "COMPLETE",
        }
        confirmed_saved += int(saved)
        status = result["status"]
        if status == "PASS" and result["worker"].get("verification") != "PASS":
            status = "FAILED"
        counts[status] += 1
        group = families.setdefault(
            case["family"],
            {
                "outcomes": Counter(),
                "settings": {},
                "component_families": {},
                "kernel_kinds": {},
            },
        )
        group["outcomes"][status] += 1
        for key, value in fields(request["recipe"]).items():
            bins = group["settings"].setdefault(key, {}).setdefault(str(value), Counter())
            bins["SELECTED"] += 1
            bins["GENERATION_SAVED"] += int(saved)
            bins[status] += 1
        for component in request["recipe"].get("components", []):
            child = component["recipe"]
            name = child.get("family", child["version"])
            bins = group["component_families"].setdefault(name, Counter())
            bins["SELECTED_COMPONENTS"] += 1
            bins[f"{status}_PARENT_COMPONENTS"] += 1
        for kernel in request["recipe"].get("kernels", []):
            bins = group["kernel_kinds"].setdefault(kernel["kind"], Counter())
            bins["SELECTED_KERNELS"] += 1
            bins[f"{status}_TASK_KERNELS"] += 1
        if status != "PASS":
            failures.append(
                {
                    "case": case["id"],
                    "status": status,
                    "saved_before_worker_verification": saved,
                    "stderr_tail": (directory / "stderr.txt").read_text().splitlines()[-8:],
                    "generation_measurement": result["measurement"],
                }
            )
            continue
        assert result["returncode"] == 0
        assert saved
        assert not result["timed_out"]
        assert not result["receipt_errors"]
        archives = list((directory / "run").rglob("data-*.zip"))
        assert len(archives) == 1
        assert archives[0].stat().st_size == result["worker"]["archive_bytes"]
        tasks = list(load_group(archives[0]))
        assert len(tasks) == 1
        task = tasks[0]
        card = json.loads(task.recipe_json)
        assert card["recipe"] == selected
        assert card["seed"] == request["seed"]
        assert card["task_id"] == request["start"]
        assert task.episode_id == result["worker"]["episode_id"]
        assert task.source_task_id not in identities
        identities.add(task.source_task_id)
        controls = selected.get("task", selected)
        assert list(task.inputs.shape) == [controls["rows"], controls["features"]]
        assert len(task.context_indices) == controls["context_rows"]
        assert len(json.loads(task.mechanism_json)["active"]) == controls["active_features"]
        for values in (task.inputs, task.scores, task.targets, task.probabilities):
            if values is not None:
                assert torch.isfinite(values).all()
    assert dict(counts) == {key: value for key, value in report["totals"].items() if value}
    assert len(identities) == counts["PASS"]
    result = {
        "source": manifest["source"],
        "suite": manifest["suite"],
        "audit_source": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256((root / "report.json").read_bytes()).hexdigest(),
        "audit_status": "PASS",
        "totals": counts,
        "confirmed_saved_tasks": confirmed_saved,
        "verified_source_tasks": len(identities),
        "new_generations": 0,
        "families": families,
        "failures": failures,
        "scope": (
            "Selected settings and verified saved tasks; "
            "not an independent mathematical oracle or bulk eligibility"
        ),
    }
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "suite",
                    "audit_status",
                    "totals",
                    "verified_source_tasks",
                    "new_generations",
                )
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    audit(args.root, args.output)
