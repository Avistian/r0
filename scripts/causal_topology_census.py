"""Retain a small CPU-only topology census; not a performance benchmark."""

import argparse
import json
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch

from r0.modules.generation import save_group
from r0.modules.priors import CausalRecipe, LinearRecipe, generate_task, sample_recipe


def run(destination: Path, profile: str = "causal-affine-v1") -> None:
    destination.mkdir(parents=True, exist_ok=False)
    summary = {}
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
            record = {"family": family, "seed": 28, "task_id": task_id}
            try:
                recipe = sample_recipe(
                    family, task=controls, seed=28, task_id=task_id, profile=profile
                )
                assert isinstance(recipe, CausalRecipe)
                record["recipe"] = asdict(recipe)
                task = generate_task(recipe, seed=28, task_id=task_id)
                tasks.append((task_id, task))
                depth, connected = [], list(range(len(recipe.nodes)))
                for child, node in enumerate(recipe.nodes):
                    depth.append(1 + max((depth[p] for p in node.parents), default=-1))
                    for parent in node.parents:
                        old, new = connected[child], connected[parent]
                        connected = [new if c == old else c for c in connected]
                card = json.loads(task.mechanism_json)
                record.update(
                    status="VALID",
                    nodes=len(recipe.nodes),
                    edges=sum(len(n.parents) for n in recipe.nodes),
                    depth=max(depth),
                    components=len(set(connected)),
                    flags=card["flags"],
                    score_std=task.scores.std(dim=0, unbiased=False).tolist(),
                    source_task_id=task.source_task_id,
                )
            except Exception as exc:
                record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
            records.append(record)
        (destination / f"{family}.jsonl").write_text(
            "".join(
                json.dumps(record, sort_keys=True, allow_nan=False) + "\n" for record in records
            )
        )
        if tasks:
            save_group(destination / f"{family}.zip", tasks, max_bytes=32 * 1024**2)
        valid = [r for r in records if r["status"] == "VALID"]
        summary[family] = {
            "attempted": len(records),
            "valid": len(valid),
            "failed": len(records) - len(valid),
            "failure_reasons": dict(
                Counter(r["error"] for r in records if r["status"] == "FAILED")
            ),
            "activation_modes": dict(
                Counter(
                    r["recipe"]["activation_prior"]["mode"]
                    for r in records
                    if "activation_prior" in r.get("recipe", {})
                )
            ),
            "exogenous_laws": dict(
                Counter(
                    node["exogenous"]["inputs"]["distribution"]
                    for r in records
                    for node in r.get("recipe", {}).get("nodes", [])
                    if "exogenous" in node
                )
            ),
            "exogenous_geometry": dict(
                Counter(
                    node["exogenous"]["geometry"]
                    for r in records
                    for node in r.get("recipe", {}).get("nodes", [])
                    if "exogenous" in node
                )
            ),
            "noise_strengths": dict(
                Counter(
                    str(node["noise_response"]["strength"])
                    for r in records
                    for node in r.get("recipe", {}).get("nodes", [])
                    if "noise_response" in node
                )
            ),
            "weight_constructions": dict(
                Counter(
                    node["weight_construction"]["version"]
                    for r in records
                    for node in r.get("recipe", {}).get("nodes", [])
                    if "weight_construction" in node
                )
            ),
            "factor_inner_dimensions": dict(
                Counter(
                    str(len(node["weight_construction"]["diagonal"]))
                    for r in records
                    for node in r.get("recipe", {}).get("nodes", [])
                    if "diagonal" in node.get("weight_construction", {})
                )
            ),
            "importance_vectors": dict(
                Counter(
                    vector["version"]
                    for r in records
                    for node in r.get("recipe", {}).get("nodes", [])
                    for vector in node.get("weight_construction", {}).get("importance", [])
                )
            ),
            "flag_counts": dict(Counter(flag for r in valid for flag in r["flags"])),
            "ranges": {
                name: [min(r[name] for r in valid), max(r[name] for r in valid)]
                for name in ("nodes", "edges", "depth", "components")
                if valid
            },
            "task_types": dict(Counter(r["recipe"]["task"]["task_type"] for r in valid)),
        }
    report = {
        "seed": 28,
        "profile": profile,
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "device": "cpu",
        "torch": torch.__version__,
        "filter": "none",
        "timing": "NOT_MEASURED",
        "families": summary,
    }
    (destination / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--profile",
        choices=(
            "causal-affine-v1",
            "causal-activated-v1",
            "causal-preferences-v1",
            "causal-exogenous-v1",
            "causal-factorized-v1",
            "causal-importance-v1",
        ),
        default="causal-affine-v1",
    )
    args = parser.parse_args()
    run(args.destination, args.profile)
