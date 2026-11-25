"""Retain selected-row generation graphs and before/after data fingerprints."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch

from r0.modules.explorer import ExplorerSession
from r0.modules.priors import TrainingTask, explain_generation, generate_task, recipe_from_dict


def digest(task: TrainingTask) -> str:
    value = hashlib.sha256(task.recipe_json.encode() + task.mechanism_json.encode())
    for tensor in (
        task.inputs,
        task.targets,
        task.scores,
        task.probabilities,
        task.context_indices,
        task.query_indices,
    ):
        if tensor is not None:
            value.update(tensor.detach().cpu().numpy().tobytes())
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    examples = ExplorerSession(args.out).request("examples", {})["examples"]
    cases: list[tuple[str, dict[str, Any]]] = []
    for name, config in examples.items():
        for task_type, classes in (("regression", 0), ("binary", 2), ("multiclass", 3)):
            case = json.loads(json.dumps(config))
            controls = case.get("task", case)
            controls.update(
                task_type=task_type,
                classes=classes,
                noise_ratio=0.7 if task_type == "regression" else 0,
                separation=0.8 if task_type != "regression" else 1,
            )
            for child in case.get("components", []):
                child_controls = child["recipe"].get("task", child["recipe"])
                child_controls.update(task_type=task_type, classes=classes)
            cases.append((f"{name}: {task_type}", case))
    wide = json.loads(json.dumps(examples["Random Fourier features"]))
    wide.update(terms=4096, calibration_rows=256)
    cases.append(("4096 Fourier terms", wide))
    many = json.loads(json.dumps(examples["Composition"]))
    many["components"] = [many["components"][0] for _ in range(64)]
    many["expression"] = {"operation": "sum", "children": list(range(64)), "strength": 1}
    cases.append(("64 linear components", many))
    for combine in ("sum", "product"):
        gp = json.loads(json.dumps(examples["Exact Gaussian process"]))
        gp.update(
            combine=combine,
            calibration_rows=32,
            kernels=[
                {"kind": "rbf"},
                {"kind": "matern12"},
                {"kind": "matern32"},
                {"kind": "matern52"},
                {"kind": "rational_quadratic", "alpha": 0.5},
                {"kind": "spectral", "frequencies": [0.1, 0.2, 0.3, 0.4]},
                {
                    "kind": "gibbs",
                    "length_slopes": [0.1, -0.2, 0.3, 0.4],
                    "amplitude_slopes": [0.1, 0, 0, 0],
                },
            ],
        )
        cases.append((f"all seven GP kernels: {combine}", gp))
    large = json.loads(json.dumps(examples["Linear"]))
    large.update(
        rows=20000, context_rows=15000, features=256, signal_features=256, active_features=128
    )
    cases.append(("20000 rows and 256 features", large))
    records: list[dict[str, Any]] = []
    for i, (name, config) in enumerate(cases):
        task = generate_task(recipe_from_dict(config), seed=27, task_id=i)
        before = digest(task)
        rows = [0, 3, len(task.inputs) - 1]
        graph = explain_generation(task, rows)
        assert all(
            parent in {node["id"] for node in graph["nodes"]}
            for node in graph["nodes"]
            for parent in node["parents"]
        )
        assert len({node["id"] for node in graph["nodes"]}) == len(graph["nodes"])
        for j in range(graph["component_count"]):
            child = explain_generation(task, rows, component=j)
            (args.out / f"case-{i}-component-{j}.json").write_text(
                json.dumps(child, allow_nan=False)
            )
        if name == "4096 Fourier terms":
            tail = explain_generation(task, rows, column_start=4090)
            basis = next(node for node in tail["nodes"] if node["id"] == "basis")
            assert basis["shape"][1] == 4096
            assert basis["column_indices"] == list(range(4090, 4096))
            (args.out / f"case-{i}-tail.json").write_text(json.dumps(tail, allow_nan=False))
        assert before == digest(task)
        (args.out / f"case-{i}.json").write_text(json.dumps(graph, allow_nan=False))
        records.append(
            {
                "case": i,
                "name": name,
                "recipe": config,
                "seed": 27,
                "task_id": i,
                "identity": task.episode_id,
                "data_sha256": before,
                "signal_max_error": graph["signal_max_error"],
                "target_max_error": graph["target_max_error"],
                "nodes": len(graph["nodes"]),
            }
        )
        print(name, "PASS", flush=True)
    (args.out / "summary.json").write_text(
        json.dumps(
            {
                "source": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "scope": "local CPU selected-row graph reconstruction; not UI or hosted proof",
                "cases": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
