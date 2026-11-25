"""Retained before/after fingerprints for the shared analytic signal evaluator."""

import hashlib
import json
import subprocess
from dataclasses import asdict

import torch

from r0.modules.episodes import TaskType
from r0.modules.priors import (
    FourierRecipe,
    GroupedLinearRecipe,
    LinearRecipe,
    PolynomialRecipe,
    QuadraticRecipe,
    Recipe,
    SmoothRecipe,
    generate_task,
)


def recipes() -> list[Recipe]:
    cases: list[Recipe] = []
    configurations: tuple[tuple[int, TaskType, int], ...] = (
        (16, "regression", 0),
        (16, "binary", 2),
        (16, "multiclass", 3),
        (513, "regression", 0),
    )
    for rows, task_type, classes in configurations:
        task = LinearRecipe(rows=rows, context_rows=8, task_type=task_type, classes=classes)
        cases.extend(
            [
                task,
                GroupedLinearRecipe(task=task, calibration_rows=256),
                PolynomialRecipe(task=task, calibration_rows=256),
                QuadraticRecipe(task=task, rank=4, calibration_rows=256),
                FourierRecipe(task=task, terms=8, calibration_rows=256),
                FourierRecipe(task=task, family="periodic", terms=8, calibration_rows=256),
                SmoothRecipe(task=task, projection="ridge", terms=8, calibration_rows=256),
                SmoothRecipe(task=task, family="sinusoidal", terms=8, calibration_rows=256),
            ]
        )
    return cases


def main() -> None:
    torch.set_num_threads(1)
    records = []
    for recipe in recipes():
        task = generate_task(recipe, seed=27, task_id=3)
        digest = hashlib.sha256(task.recipe_json.encode() + task.mechanism_json.encode())
        for tensor in (
            task.inputs,
            task.scores,
            task.targets,
            task.probabilities,
            task.context_indices,
            task.query_indices,
        ):
            if tensor is not None:
                digest.update(tensor.cpu().numpy().tobytes())
        records.append(
            {
                "recipe": asdict(recipe),
                "source_task_id": task.source_task_id,
                "episode_id": task.episode_id,
                "sha256": digest.hexdigest(),
            }
        )
    print(
        json.dumps(
            {
                "source": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "cases": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
