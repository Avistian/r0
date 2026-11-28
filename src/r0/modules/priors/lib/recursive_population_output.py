"""Direct population forest outputs preserve actual labels and original row ownership."""

import platform
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from r0.modules.priors.lib.recursive_forest_evaluation import evaluate_recursive_forest
from r0.modules.priors.lib.recursive_population_ancestors import population_ancestor_values
from r0.modules.priors.lib.recursive_population_records import (
    RecursivePopulationRecipe,
    RecursivePopulationRequest,
    population_identity,
)
from r0.modules.priors.lib.representatives import encode, record_fingerprint
from r0.modules.priors.lib.training_features import TrainingFeature
from r0.modules.priors.lib.training_task import TrainingTask


def generate_recursive_population(
    recipe: RecursivePopulationRecipe, *, seed: int, task_id: int, device: str = "cpu"
) -> TrainingTask:
    population_identity(seed, task_id)
    if type(recipe) is not RecursivePopulationRecipe or device != "cpu":
        raise ValueError("population forest generation requires its realized CPU record")
    if (seed, task_id) != (recipe.seed, recipe.task_id):
        raise ValueError("population identity changed; explicitly realize a new population request")
    recipe.validate_replay()
    request = recipe.request
    values = population_ancestor_values(request.ancestors, seed=seed, task_id=task_id)
    inputs = np.asarray(recipe.parent_rows, dtype=np.float64)
    emitted = evaluate_recursive_forest(recipe.sample, inputs, limits=recipe.limits)
    return population_output(
        request,
        asdict(recipe),
        values.observed,
        values.schema,
        emitted,
        seed=seed,
        task_id=task_id,
    )


def population_output(
    request: RecursivePopulationRequest,
    config: dict[str, Any],
    observed: torch.Tensor,
    schema: tuple[TrainingFeature, ...],
    emitted: NDArray[Any],
    *,
    seed: int,
    task_id: int,
    intervention: dict[str, Any] | None = None,
) -> TrainingTask:
    result = torch.tensor(emitted, dtype=torch.int64 if request.plan.classes else torch.float64)
    targets = result if request.plan.classes else result[:, request.target_channel]
    scores = (
        torch.nn.functional.one_hot(result, num_classes=request.plan.classes).to(torch.float64)
        if request.plan.classes
        else result
    )
    source = record_fingerprint((config, seed, task_id))
    split = np.random.default_rng(
        int(
            record_fingerprint(
                ("recursive-population-split-v1", seed, task_id, request.task.split)
            ),
            16,
        )
    ).permutation(request.task.rows)
    indices = torch.tensor(split, dtype=torch.int64)
    return TrainingTask(
        source,
        record_fingerprint((source, request.task.split, request.task.context_rows)),
        request.task.task_type,
        encode(
            {
                "recipe": config,
                "seed": seed,
                "task_id": task_id,
                "execution": {
                    "implementation": config["version"],
                    "device": "cpu",
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "torch": str(torch.__version__),
                    "cpu_threads": torch.get_num_threads(),
                    "cuda": torch.version.cuda,
                    "gpu": None,
                },
            }
        ),
        encode(
            {
                "family": "recursive_population",
                "version": config["version"],
                "lifecycle": "fresh-forest-per-row",
                "source": request.plan.source,
                "population_identity": {
                    "seed": seed,
                    "task_id": task_id,
                    "rows": request.task.rows,
                },
                "parent_slices": request.parents,
                "parent_representation": "encoded ancestor states",
                "output_semantics": "majority-label-indicator"
                if request.plan.classes
                else "mean-regression-vector",
                "target_origin": "native-variable-v1",
                "target_channel": request.target_channel,
                "conditional_moments": (
                    "bounded conditional on each retained finite row-owned forest"
                ),
                "unconditional_moments": "separate construction law; no fixed function across rows",
                "reconstruction": (
                    "Algorithm 5 per-row lifecycle with separately named tree law; "
                    "ancestor DAG is r0 extension"
                ),
                "admission": request.admit(),
                "intervention": intervention,
            }
        ),
        observed,
        targets,
        scores,
        None,
        indices[: request.task.context_rows],
        indices[request.task.context_rows :],
        request.vocabulary,
        schema,
        "native-variable-v1",
    )
