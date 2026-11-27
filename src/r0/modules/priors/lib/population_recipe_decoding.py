"""Distinct explicit population recipe versions share strict archive dispatch."""

from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits


def population_recipe_from_dict(fields: dict[str, Any], *, limits: CausalLimits):
    if (
        "parent_rows" in fields
        and "observed_rows" in fields
        and fields.get("version")
        not in ("recursive-population-v1", "recursive-population-intervention-v1")
    ):
        raise ValueError("population archive requires its explicit version")
    if (
        "bindings" in fields
        and "request" in fields
        and fields.get("version") != "recursive-population-graph-v1"
    ):
        raise ValueError("population DAG archive requires its explicit version")
    if {"base", "change", "samples", "inputs"} <= fields.keys() and fields.get(
        "version"
    ) != "recursive-population-graph-intervention-v1":
        raise ValueError("population DAG intervention requires its explicit version")
    version = fields.get("version", "linear-v1")
    if version == "recursive-population-graph-intervention-v1":
        from r0.modules.priors.lib.population_graph_intervention_decoding import (
            population_graph_intervention_from_dict,
        )

        return population_graph_intervention_from_dict(fields, limits=limits)
    if version == "recursive-population-graph-v1":
        from r0.modules.priors.lib.recursive_population_graph_decoding import (
            recursive_population_graph_from_dict,
        )

        return recursive_population_graph_from_dict(fields, limits=limits)
    if version == "recursive-population-intervention-v1":
        from r0.modules.priors.lib.recursive_population_decoding import (
            recursive_population_intervention_from_dict,
        )

        return recursive_population_intervention_from_dict(fields, limits=limits)
    if version == "recursive-population-v1":
        from r0.modules.priors.lib.recursive_population_decoding import (
            recursive_population_from_dict,
        )

        return recursive_population_from_dict(fields, limits=limits)
    return None
