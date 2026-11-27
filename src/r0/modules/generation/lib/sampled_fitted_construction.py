"""Sample parameter laws and fit synthetic ancestors within the guarded worker."""

from typing import Any, cast

import numpy as np

from r0.modules import priors


def construct_sampled_fitted(
    fields: dict[str, Any], *, seed: int, limits: priors.CausalLimits
) -> priors.SampledFittedFunctionalCausalRecipe:
    if set(fields) != {"skeleton", "fits"}:
        raise ValueError("sampled fitted construction requires skeleton and selected priors")
    skeleton = priors.recipe_from_dict(fields["skeleton"], limits=limits)
    if type(skeleton) is not priors.FunctionalCausalRecipe:
        raise ValueError("sampled fitting requires an authored uncalibrated skeleton")
    entries = fields["fits"]
    if type(entries) is not tuple or not 1 <= len(cast(tuple[Any, ...], entries)) <= limits.nodes:
        raise ValueError("sampled fitting requires bounded node/prior selections")
    selected: list[tuple[int, priors.FittedTreeParameterPrior]] = []
    for entry in cast(tuple[Any, ...], entries):
        if type(entry) is not dict or set(cast(dict[str, Any], entry)) != {"node", "prior"}:
            raise ValueError("each sampled fit requires node and parameter prior")
        values = cast(dict[str, Any], entry)
        selected.append(
            (
                values["node"],
                priors.fitted_parameter_prior_from_dict(values["prior"], limits=limits),
            )
        )
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)]
    return priors.sample_parameterized_fitted_graph(
        skeleton,
        tuple(selected),
        parameter_rng=streams[0],
        prefix_rng=streams[1],
        input_rng=streams[2],
        target_rng=streams[3],
        backend_rng=streams[4],
    )
