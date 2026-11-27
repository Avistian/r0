"""Synthetic fitted-tree graph construction inside the guarded generation worker."""

from typing import Any, cast

import numpy as np

from r0.modules import priors


def construct_fitted(
    fields: dict[str, Any], *, seed: int, limits: priors.CausalLimits
) -> priors.FittedFunctionalCausalRecipe:
    if set(fields) != {"skeleton", "fits"}:
        raise ValueError("fitted construction requires exactly a frozen skeleton and fit plans")
    skeleton = priors.recipe_from_dict(fields["skeleton"], limits=limits)
    if type(skeleton) is not priors.FunctionalCausalRecipe:
        raise ValueError(
            "fitted construction requires an authored uncalibrated functional skeleton"
        )
    raw_fits = fields["fits"]
    if (
        type(raw_fits) is not tuple
        or not raw_fits
        or len(cast(tuple[Any, ...], raw_fits)) > limits.nodes
    ):
        raise ValueError("fitted construction requires bounded selected node plans")
    plans: list[tuple[int, priors.FittedTreePlan]] = []
    for raw in cast(tuple[Any, ...], raw_fits):
        if type(raw) is not dict or set(cast(dict[str, Any], raw)) != {"node", "plan"}:
            raise ValueError("each fitted selection requires its node and plan")
        entry = cast(dict[str, Any], raw)
        plans.append((entry["node"], priors.FittedTreePlan(**entry["plan"])))
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    return priors.sample_fitted_graph(
        skeleton,
        tuple(plans),
        prefix_rng=streams[0],
        input_rng=streams[1],
        target_rng=streams[2],
        backend_rng=streams[3],
    )
