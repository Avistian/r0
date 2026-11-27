"""Guarded composition of network architecture, frozen statistics and row noise."""

from typing import Any, cast

import numpy as np

from r0.modules import priors


def construct_network_innovations(
    fields: dict[str, Any], *, seed: int, limits: priors.CausalLimits
) -> priors.NetworkInnovationRecipe:
    from r0.modules.generation.lib.construction import construct_graph

    if set(fields) != {"source", "statistics", "plan"} or (fields["statistics"] is None) == (
        fields["plan"] is None
    ):
        raise ValueError(
            "network innovations require source and exactly one statistics request or authored plan"
        )
    source_controls = cast(dict[str, Any], fields["source"])
    if type(source_controls) is not dict or source_controls.get("topology") not in (
        "network",
        "network-authored",
        "network-hyperprior",
        "source-root-network",
    ):
        raise ValueError("network innovations wrap an explicit standalone network construction")
    source = construct_graph(source_controls, seed=seed, limits=limits)
    if type(source) is not priors.NetworkRecipe:
        raise ValueError("network innovation source must be a standalone network")
    if fields["statistics"] is not None:
        request = priors.network_innovation_request_from_dict(fields["statistics"], limits=limits)
        # Domain-separated from existing architecture constructor streams.
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0x4E4F4953]))
        return priors.sample_network_innovations(source, request, statistics_rng=rng)
    plan = priors.network_innovation_plan_from_dict(fields["plan"], limits=limits)
    return priors.with_network_innovations(source, plan)
