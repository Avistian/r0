"""Bounded raw network response grids, decoded and evaluated inside the worker."""

import json
import math
from typing import Any, cast

import numpy as np
import torch

from r0.modules import priors


def network_curve(data: dict[str, Any]) -> dict[str, Any]:
    maximum = data.get("max_bytes", 256 * 1024**2)
    if type(maximum) is not int or not 1 <= maximum <= 256 * 1024**2:
        raise ValueError("network curve byte cap must be an integer in 1..268435456")
    if len(json.dumps(data, allow_nan=False).encode()) > maximum:
        raise ValueError("network curve request exceeds byte cap")
    limits = priors.CausalLimits(**data["causal_limits"])
    from r0.modules.generation.lib.topology_curve_identity import topology_curve_identity

    topology_identity = topology_curve_identity(data)
    recipe = priors.recipe_from_dict(data["recipe"], limits=limits)
    topology_source = (
        recipe
        if isinstance(recipe, (priors.HyperpriorGraph, priors.JointHyperpriorGraph))
        else None
    )
    if topology_source is not None:
        recipe = topology_source.runtime
    if isinstance(recipe, priors.RealizedPrototypeBatchNetworkRecipe):
        # Observed grid coordinates already are convex mixtures; do not redraw
        # prototypes, row weights, or source normalization for response queries.
        recipe = recipe.request.source
    if not isinstance(
        recipe, (priors.NetworkRecipe, priors.NetworkInputView, priors.FunctionalCausalRecipe)
    ):
        raise ValueError("response requires a standalone network recipe")
    anchor, feature, points, bounds = (
        data["anchor"],
        data["feature"],
        data["points"],
        data["bounds"],
    )
    if type(anchor) is not list or type(bounds) is not list:
        raise ValueError("invalid network response grid")
    anchor = cast(list[Any], anchor)
    bounds = cast(list[Any], bounds)
    if (
        len(anchor) != recipe.task.features
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in anchor)
        or type(feature) is not int
        or not 0 <= feature < len(anchor)
        or type(points) is not int
        or not 2 <= points <= 512
        or len(bounds) != 2
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds)
        or bounds[0] > bounds[1]
    ):
        raise ValueError("invalid network response grid")
    # Admit the query geometry before allocating the repeated anchor/grid.
    if isinstance(recipe, priors.FunctionalCausalRecipe):
        recipe.admit(evaluation_rows=recipe.task.rows + points)
    elif isinstance(recipe, priors.NetworkInputView):
        recipe.admit(rows=points)
    else:
        priors.admit_frozen_function(recipe.function, points, recipe.limits)
    x = torch.tensor(anchor, dtype=torch.float64).repeat(points, 1)
    fraction = torch.linspace(0, 1, points, dtype=torch.float64)
    values = (1 - fraction) * bounds[0] + fraction * bounds[1]
    x[:, feature] = values
    reference: dict[str, Any] = {}
    if isinstance(recipe, priors.NetworkInnovationRecipe):
        mode = data.get("reference", "zero")
        if mode not in ("zero", "original-row"):
            raise ValueError("network reference must be zero or original-row")
        innovations = None
        if mode == "original-row":
            row = priors.network_innovation_row(
                recipe, seed=data["seed"], task_id=data["task_id"], row=data["anchor_row"]
            )
            innovations = {key: value.repeat(points, 1) for key, value in row.items()}
            reference["fixed_innovations"] = {
                str(key): value.numpy().tolist() for key, value in row.items()
            }
        scores = priors.evaluate_network_innovation_reference(recipe, x, innovations=innovations)
        reference.update(
            reference=mode, original_task_id=str(data["task_id"]), original_row=data["anchor_row"]
        )
    elif isinstance(recipe, priors.FunctionalCausalRecipe):
        mode = data.get("reference", "original-row")
        scores = priors.evaluate_functional_reference(
            recipe,
            x,
            seed=data["seed"],
            task_id=data["task_id"],
            row=data["anchor_row"],
            reference=mode,
        )
        reference.update(
            reference=mode,
            original_task_id=str(data["task_id"]),
            original_row=data["anchor_row"],
            clamp_location="state_and_readout",
        )
    else:
        scores = (
            priors.view_signal(recipe, x)
            if isinstance(recipe, priors.NetworkInputView)
            else priors.evaluate_frozen_function(recipe.function, x, limits=recipe.limits)
        )
    target_reference = (
        priors.topology_reference_targets(
            topology_source,
            scores,
            seed=data["seed"],
            task_id=data["task_id"],
            row=data["anchor_row"],
        )
        if topology_source is not None
        else {}
    )
    result = {
        **target_reference,
        **reference,
        **topology_identity,
        "values": np.asarray(values.numpy(), dtype=np.float64).tolist(),
        "scores": np.asarray(scores.numpy(), dtype=np.float64).tolist(),
    }
    if len(json.dumps(result, allow_nan=False).encode()) > maximum:
        raise ValueError("network response exceeds byte cap")
    return result
