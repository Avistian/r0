"""Capture old-profile outcomes without filtering failed draws."""

import hashlib
import json
from dataclasses import asdict

from r0.modules.priors import LinearRecipe, explain_generation, generate_task, sample_recipe


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def snapshot(out):
    records = []
    folder = out / "legacy_cases"
    folder.mkdir()
    for profile in (
        "causal-affine-v1",
        "causal-activated-v1",
        "causal-preferences-v1",
        "causal-exogenous-v1",
        "causal-factorized-v1",
    ):
        for family in ("forward", "layered", "redirection"):
            for task_type in ("regression", "binary", "multiclass"):
                for seed, task_id in (*((s, 10) for s in (0, 1, 2, 3, 28, 45)), (28, 50)):
                    key = f"{profile}_{family}_{task_type}_{seed}_{task_id}"
                    entry = {
                        "key": key,
                        "profile": profile,
                        "family": family,
                        "task_type": task_type,
                        "seed": seed,
                        "task_id": task_id,
                    }
                    recipe = None
                    try:
                        recipe = sample_recipe(
                            family,
                            task=LinearRecipe(
                                rows=16,
                                context_rows=8,
                                task_type=task_type,
                                classes={"regression": 0, "binary": 2, "multiclass": 3}[task_type],
                            ),
                            seed=seed,
                            task_id=task_id,
                            profile=profile,
                        )
                        entry["recipe"] = asdict(recipe)
                        task = generate_task(recipe, seed=seed, task_id=task_id)
                        entry["status"] = "PASS"
                        entry["recipe_json"] = task.recipe_json
                        entry["mechanism_json"] = task.mechanism_json
                        entry["source_task_id"] = task.source_task_id
                        entry["episode_id"] = task.episode_id
                        entry["episode_sha256"] = digest(asdict(task.episode()))
                        for field in (
                            "inputs",
                            "targets",
                            "scores",
                            "probabilities",
                            "context_indices",
                            "query_indices",
                        ):
                            tensor = getattr(task, field)
                            entry[field] = (
                                None
                                if tensor is None
                                else {
                                    "shape": list(tensor.shape),
                                    "dtype": str(tensor.dtype),
                                    "bytes_sha256": hashlib.sha256(
                                        tensor.cpu().numpy().tobytes()
                                    ).hexdigest(),
                                }
                            )
                        entry["trace_sha256"] = digest(explain_generation(task, [0, 15]))
                    except ValueError as error:
                        entry["status"] = "FAILED"
                        entry["stage"] = "generation" if recipe is not None else "sampling"
                        entry["error"] = type(error).__name__ + ": " + str(error)
                    (folder / (key + ".json")).write_text(
                        json.dumps(entry, sort_keys=True, indent=2) + "\n"
                    )
                    records.append(
                        {
                            "key": key,
                            "status": entry["status"],
                            "sha256": digest(entry),
                            "error": entry.get("error"),
                        }
                    )
    (out / "legacy.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


if __name__ == "__main__":
    import inspect
    import sys
    from pathlib import Path

    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    source = Path(inspect.getfile(sample_recipe))
    (out / "import-receipt.json").write_text(
        json.dumps(
            {
                "python": sys.executable,
                "source": str(source),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    rows = snapshot(out)
    print({"checks": len(rows), "failed_draws": sum(r["status"] == "FAILED" for r in rows)})
