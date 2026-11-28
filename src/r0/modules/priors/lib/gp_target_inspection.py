"""Original-row traces for matched/fresh GP target environments."""

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any

import torch

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.gp_column_target import evaluate_gp_column_target
from r0.modules.priors.lib.gp_target_environment import (
    generate_gp_target_environment,
    gp_column_target_recipe_from_dict,
    gp_target_adapter_stream,
    gp_target_record_metadata,
)
from r0.modules.priors.lib.gp_task_validation import validate_gp_task
from r0.modules.priors.lib.target_inspection import target_steps
from r0.modules.priors.lib.training_task import TrainingTask


def gp_target_steps(
    task: TrainingTask,
    indices: Sequence[int],
    add: Callable[..., None],
    *,
    limits: CausalLimits | None,
) -> dict[str, Any]:
    cap = limits or CausalLimits()
    record = json.loads(task.recipe_json)
    gp_target_record_metadata(record, limits=cap)
    recipe = gp_column_target_recipe_from_dict(record["recipe"], limits=cap)
    expected = generate_gp_target_environment(
        recipe, seed=record["seed"], task_id=record["task_id"]
    )
    validate_gp_task(task, expected, recipe.version)
    c = recipe.comparison
    p = c.original
    rows = list(indices)
    replacement = recipe.side == "replacement"
    target = c.replacement if replacement else p.target
    raw = (
        c.replacement_raw
        if replacement
        else evaluate_gp_column_target(target, c.draw.values, limits=cap)
    )
    centers, scales = (c.centers, c.scales) if replacement else (p.centers, p.scales)
    add(
        "locations",
        "Fixed original locations",
        [],
        "original population locations",
        torch.tensor(c.draw.locations)[rows],
        {"row_ids": rows},
    )
    add(
        "innovations",
        "Shared comparison innovations",
        [],
        "matched originals or fixed-seed fresh normals",
        torch.tensor(c.draw.innovations)[rows],
        {"mode": c.mode, "fresh_seed": c.fresh_seed, "dependency_rows": list(p.row_ids)},
    )
    add(
        "inputs",
        "Joint GP feature columns",
        ["locations", "innovations"],
        "X[:,j] = L[j] @ Z[:,j]; all original rows",
        expected.inputs[rows],
        {"population_id": p.population_id},
    )
    add(
        "raw-target",
        "Original target" if not replacement else "Replacement target",
        ["inputs"],
        "frozen target equation",
        torch.tensor(raw)[rows],
        asdict(target),
    )
    signal = expected.scores[rows]
    add(
        "signal",
        "Independent target calibration",
        ["raw-target"],
        "(raw - center) / scale; original independent marginal inputs",
        signal,
        {
            "centers": centers,
            "scales": scales,
            "query_fit": False,
            "calibration_rows": p.request.calibration_rows,
        },
    )
    target_steps(
        asdict(recipe.task), signal, rows, lambda role: gp_target_adapter_stream(recipe, role), add
    )
    return {
        "kind": "gp_target_environment",
        "rows": rows,
        "population_id": p.population_id,
        "side": recipe.side,
        "mode": c.mode,
        "shared_matrices": [
            {
                "kernel_columns": list(range(p.request.columns))
                if len(c.draw.kernels) == 1
                else [j],
                "row_ids": list(p.row_ids),
                "column_row_ids": list(p.row_ids),
                "covariance": [list(row) for row in matrix],
                "factor": [list(row) for row in c.draw.factors[j]],
                "axis_semantics": "original population row by original population row",
            }
            for j, matrix in enumerate(c.draw.covariance)
        ],
    }
