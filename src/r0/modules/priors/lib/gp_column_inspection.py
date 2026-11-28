"""Finite GP population traces keep covariance axes distinct from selected task rows."""

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any

import torch

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.gp_column_decoding import gp_column_recipe_from_dict
from r0.modules.priors.lib.gp_column_output import generate_gp_columns
from r0.modules.priors.lib.gp_column_population import GPColumnPopulation, gp_stream
from r0.modules.priors.lib.gp_task_validation import validate_gp_task
from r0.modules.priors.lib.target_inspection import target_steps
from r0.modules.priors.lib.training_task import TrainingTask


def gp_column_steps(
    task: TrainingTask,
    indices: Sequence[int],
    add: Callable[..., None],
    *,
    limits: CausalLimits | None,
) -> dict[str, Any]:
    record = json.loads(task.recipe_json)
    recipe = gp_column_recipe_from_dict(record["recipe"], limits=limits or CausalLimits())
    if not isinstance(recipe.source, GPColumnPopulation):
        raise ValueError("GP trace requires the retained realized population")
    expected = generate_gp_columns(recipe, seed=record["seed"], task_id=record["task_id"])
    validate_gp_task(task, expected, recipe.version)
    p = recipe.source
    selected = list(indices)
    add(
        "locations",
        "Latent row locations",
        [],
        "retained population locations",
        torch.tensor(p.draw.locations)[selected],
        {"row_ids": selected, "law": p.request.location_family},
    )
    add(
        "innovations",
        "Column innovations",
        [],
        "retained iid standard normals; full row vector drives Cholesky",
        torch.tensor(p.draw.innovations)[selected],
        {"row_ids": selected, "dependency_rows": list(p.row_ids)},
    )
    add(
        "inputs",
        "Joint GP feature columns",
        ["locations", "innovations"],
        "X[:,j] = L[j] @ Z[:,j]; all original rows",
        expected.inputs[selected],
        {
            "population_id": p.population_id,
            "shared_kernel": len(p.draw.kernels) == 1,
            "row_ids": selected,
        },
    )
    add(
        "raw-target",
        "Frozen target mechanism",
        ["inputs"],
        "linear, full quadratic, or tanh neural combination",
        torch.tensor(p.raw_scores)[selected],
        asdict(p.target),
    )
    signal = expected.scores[selected]
    add(
        "signal",
        "Independent target calibration",
        ["raw-target"],
        "(raw - independent marginal mean) / independent marginal scale",
        signal,
        {
            "centers": p.centers,
            "scales": p.scales,
            "calibration_rows": p.request.calibration_rows,
            "query_fit": False,
        },
    )
    target_steps(
        asdict(recipe.task),
        signal,
        selected,
        lambda role: gp_stream(p.seed, p.task_id, "adapter", role),
        add,
    )
    return {
        "kind": "gp_column_population",
        "rows": selected,
        "population_id": p.population_id,
        "shared_matrices": gp_population_matrices(p),
    }


def gp_population_matrices(p: GPColumnPopulation):
    """Retained covariance/factor matrices with explicit original-row axes."""
    return [
        {
            "kernel_columns": list(range(p.request.columns)) if len(p.draw.kernels) == 1 else [j],
            "row_ids": list(p.row_ids),
            "column_row_ids": list(p.row_ids),
            "covariance": [list(row) for row in k],
            "factor": [list(row) for row in p.draw.factors[j]],
            "axis_semantics": "original population row by original population row",
        }
        for j, k in enumerate(p.draw.covariance)
    ]
