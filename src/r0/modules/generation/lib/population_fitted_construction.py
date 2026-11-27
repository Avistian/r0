"""Fit explicit populations inside the bounded worker, bound to actual episode IDs."""

from typing import Any, cast

import numpy as np

from r0.modules import priors


def construct_population_fitted(
    fields: dict[str, Any],
    *,
    construction_seed: int,
    seed: int,
    task_id: int,
    limits: priors.CausalLimits,
) -> priors.PopulationFittedGraph:
    if set(fields) != {"topology", "skeleton", "fits", "population", "mixed"}:
        raise ValueError("population construction requires its skeleton, plans and population")
    skeleton = priors.recipe_from_dict(fields["skeleton"], limits=limits)
    if type(skeleton) is not priors.FunctionalCausalRecipe:
        raise ValueError("population fitting requires an authored uncalibrated skeleton")
    entries = fields["fits"]
    if type(entries) is not list or not 1 <= len(cast(list[Any], entries)) <= limits.nodes:
        raise ValueError("population fits must be a bounded list")
    mixed = None if fields["mixed"] is None else priors.mixed_population_from_dict(fields["mixed"])
    selected: list[tuple[int, priors.PopulationTreePlan]] = []
    for entry in cast(list[Any], entries):
        if type(entry) is not dict or set(cast(dict[str, Any], entry)) != {"node", "plan"}:
            raise ValueError("population selections require node and plan")
        entry = cast(dict[str, Any], entry)
        plan_fields = dict(cast(dict[str, Any], entry["plan"]))
        # Actual support/batch sizes are determined by the selected episode, not
        # a padded independent fit-size control. The realized plan retains them.
        if fields["population"] != "mixed-independent":
            rows = (
                skeleton.task.context_rows
                if fields["population"] == "support-only"
                else skeleton.task.rows
            )
            if "rows" in plan_fields and plan_fields["rows"] != rows:
                raise ValueError("declared fit rows differ from actual selected population")
            plan_fields["rows"] = rows
        plan = priors.PopulationTreePlan(**plan_fields)
        selected.append((entry["node"], plan))
    streams = [
        np.random.default_rng(s)
        for s in np.random.SeedSequence([construction_seed, seed, task_id]).spawn(3)
    ]
    return priors.fit_population_graph(
        skeleton,
        tuple(selected),
        population=fields["population"],
        seed=seed,
        task_id=task_id,
        input_rng=streams[0],
        target_rng=streams[1],
        backend_rng=streams[2],
        mixed=mixed,
    )
