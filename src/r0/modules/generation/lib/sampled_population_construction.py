"""Bound effective episode rows before drawing any population-fit parameters."""

from typing import Any, cast

import numpy as np

from r0.modules import priors


def construct_sampled_population(
    fields: dict[str, Any],
    *,
    construction_seed: int,
    seed: int,
    task_id: int,
    limits: priors.CausalLimits,
) -> priors.SampledPopulationFittedGraph:
    if set(fields) != {"topology", "skeleton", "fits", "population", "mixed"}:
        raise ValueError("sampled population construction needs skeleton, priors and lifecycle")
    skeleton = priors.recipe_from_dict(fields["skeleton"], limits=limits)
    if type(skeleton) is not priors.FunctionalCausalRecipe:
        raise ValueError("sampled population fitting requires an authored uncalibrated skeleton")
    entries = fields["fits"]
    if type(entries) is not list or not 1 <= len(cast(list[Any], entries)) <= limits.nodes:
        raise ValueError("sampled population fits must be a bounded list")
    mixed = None if fields["mixed"] is None else priors.mixed_population_from_dict(fields["mixed"])
    selected: list[tuple[int, priors.PopulationTreeParameterPrior]] = []
    for entry in cast(list[Any], entries):
        if type(entry) is not dict or set(cast(dict[str, Any], entry)) != {"node", "prior"}:
            raise ValueError("sampled population selections require node and prior")
        entry = cast(dict[str, Any], entry)
        if type(entry["prior"]) is not dict:
            raise ValueError("population parameter prior must be an object")
        raw = dict(cast(dict[str, Any], entry["prior"]))
        if raw.get("population", fields["population"]) != fields["population"]:
            raise ValueError("parameter prior and construction lifecycles differ")
        raw["population"] = fields["population"]
        if type(raw.get("base")) is not dict:
            raise ValueError("population parameter prior requires its base plan")
        base = dict(cast(dict[str, Any], raw["base"]))
        if fields["population"] != "mixed-independent":
            rows = (
                skeleton.task.context_rows
                if fields["population"] == "support-only"
                else skeleton.task.rows
            )
            if "rows" in base and base["rows"] != rows:
                raise ValueError("prior base fit rows differ from actual selected population")
            base["rows"] = rows
        raw["base"] = base
        selected.append(
            (entry["node"], priors.population_parameter_prior_from_dict(raw, limits=limits))
        )
    streams = [
        np.random.default_rng(s)
        for s in np.random.SeedSequence([construction_seed, seed, task_id]).spawn(4)
    ]
    return priors.sample_parameterized_population_graph(
        skeleton,
        tuple(selected),
        population=fields["population"],
        seed=seed,
        task_id=task_id,
        mixed=mixed,
        parameter_rng=streams[0],
        input_rng=streams[1],
        target_rng=streams[2],
        backend_rng=streams[3],
    )
