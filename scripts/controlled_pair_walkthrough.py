"""Retain paired variants and context-only linear-model query measurements."""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from r0.modules.explorer import ExplorerSession
from r0.modules.priors import LinearRecipe, TaskPair, generate_pair, generate_task, recipe_from_dict


def prediction_errors(pair: TaskPair) -> tuple[np.ndarray, np.ndarray]:
    """OLS uses only public Episode context rows/labels; errors use query targets."""
    errors = []
    for task in (pair.before, pair.after):
        episode = task.episode()
        view = episode.predictor_view()
        context = np.array([row.values for row in view.context], dtype=np.float64)
        queries = np.array([row.values for row in view.queries], dtype=np.float64)
        context = np.column_stack((np.ones(len(context)), context))
        queries = np.column_stack((np.ones(len(queries)), queries))
        weights = np.linalg.lstsq(context, np.asarray(view.context_targets), rcond=None)[0]
        errors.append(queries @ weights - np.asarray(episode.query_targets))
    return errors[0], errors[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    examples = ExplorerSession(args.out).request("examples", {})["examples"]
    cases: list[dict[str, Any]] = []
    for name, original in examples.items():
        for task_type, classes in (("regression", 0), ("binary", 2), ("multiclass", 3)):
            config = json.loads(json.dumps(original))
            controls = config.get("task", config)
            controls.update(
                task_type=task_type, classes=classes, noise_ratio=0.2 if classes == 0 else 0
            )
            for component in config.get("components", []):
                child = component["recipe"].get("task", component["recipe"])
                child.update(task_type=task_type, classes=classes)
            settings = [("noise", 0.8)] if classes == 0 else [("separation", 0.0)]
            settings.extend((("irrelevant", controls["features"] + 2), ("sparsity", 2)))
            for intervention, value in settings:
                case = len(cases)
                print(case, name, task_type, intervention, flush=True)
                recipe = recipe_from_dict(config)
                allow = intervention == "sparsity" and not isinstance(recipe, LinearRecipe)
                started = time.perf_counter()
                pair = generate_pair(
                    recipe,
                    intervention=intervention,
                    value=value,
                    seed=27,
                    task_id=case,
                    allow_estimated_variance=allow,
                )
                elapsed = time.perf_counter() - started
                card = json.loads(pair.card_json)
                assert all(card["required_equalities"].values())
                # Replay directly from the retained pair card, independently of pair construction.
                for side in ("before", "after"):
                    record = card[side]["recipe"]
                    replay = generate_task(
                        recipe_from_dict(record["recipe"]),
                        seed=record["seed"],
                        task_id=record["task_id"],
                    )
                    assert replay.episode() == getattr(pair, side).episode()
                    assert replay.mechanism_json == getattr(pair, side).mechanism_json
                if intervention == "noise":
                    left = pair.before.targets.numpy() - pair.before.scores[:, 0].numpy()
                    right = pair.after.targets.numpy() - pair.after.scores[:, 0].numpy()
                    np.testing.assert_allclose(right, 4 * left, rtol=1e-12, atol=1e-12)
                elif intervention == "separation":
                    assert pair.after.probabilities is not None
                    np.testing.assert_allclose(pair.after.probabilities.numpy(), 1 / classes)
                (args.out / f"case-{case}.json").write_text(
                    json.dumps(card, indent=2, allow_nan=False) + "\n"
                )
                if task_type == "regression" and intervention == "noise":
                    np.savez(
                        args.out / f"case-{case}.npz",
                        before_inputs=pair.before.inputs.numpy(),
                        after_inputs=pair.after.inputs.numpy(),
                        before_targets=pair.before.targets.numpy(),
                        after_targets=pair.after.targets.numpy(),
                        scores=pair.before.scores.numpy(),
                        context=pair.before.context_indices.numpy(),
                        query=pair.before.query_indices.numpy(),
                    )
                cases.append(
                    {
                        "case": case,
                        "family": name,
                        "task_type": task_type,
                        "intervention": intervention,
                        "pair_id": card["pair_id"],
                        "variance_control": card["variance_control"],
                        "seconds": elapsed,
                    }
                )
    # No parameter selection: one OLS model definition across 30 paired root draws.
    errors: dict[str, list[np.ndarray]] = {name: [] for name in ("noise", "irrelevant", "sparsity")}
    baseline = LinearRecipe(
        rows=256,
        context_rows=192,
        features=16,
        signal_features=16,
        active_features=8,
        noise_ratio=0.2,
    )
    model_records = []
    for task_id in range(30):
        for intervention, value in (("noise", 0.8), ("irrelevant", 64), ("sparsity", 2)):
            pair = generate_pair(
                baseline, intervention=intervention, value=value, seed=27, task_id=task_id
            )
            left, right = prediction_errors(pair)
            errors[intervention].append(np.stack((left, right)))
            model_records.append(
                {
                    "task_id": task_id,
                    "intervention": intervention,
                    "pair_id": json.loads(pair.card_json)["pair_id"],
                    "before_query_mse": float(np.mean(left**2)),
                    "after_query_mse": float(np.mean(right**2)),
                }
            )
    np.savez(
        args.out / "ols-query-errors.npz", **{key: np.stack(value) for key, value in errors.items()}
    )
    (args.out / "ols-records.json").write_text(json.dumps(model_records, indent=2) + "\n")
    aggregate = {
        key: {
            "root_draws": len(value),
            "before_query_mse": float(np.mean(np.stack(value)[:, 0] ** 2)),
            "after_query_mse": float(np.mean(np.stack(value)[:, 1] ** 2)),
        }
        for key, value in errors.items()
    }
    (args.out / "summary.json").write_text(
        json.dumps(
            {
                "source": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "scope": "local CPU pairs and OLS; no UI, hosted or frontier-model evidence",
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "cases": cases,
                "model": "OLS/intercept; context fit, observed-query MSE, no tuning",
                "ols": aggregate,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "PASS",
        len(cases),
        "generation cases; 30 shared root draws per OLS intervention",
        flush=True,
    )


if __name__ == "__main__":
    main()
