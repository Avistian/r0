"""Construct a comparison only after complete outer change semantics are admitted."""

from typing import Any

from r0.modules.priors.lib.causal_intervention_decoding import intervention_from_dict
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.population_graph_intervention_decoding import (
    preflight_population_graph_intervention,
)
from r0.modules.priors.lib.recursive_population_graph_decoding import (
    recursive_population_graph_from_dict,
)
from r0.modules.priors.lib.recursive_population_graph_intervention import (
    RecursivePopulationGraphIntervention,
    intervene_recursive_population_graph,
)


def construct_population_graph_change(
    base: dict[str, Any], interventions: Any, *, mode: str, population: int, limits: CausalLimits
) -> RecursivePopulationGraphIntervention:
    admit_function_metadata((base, interventions), limits)
    try:
        change = {
            "version": "causal-intervention-v1",
            "base": base["request"]["base"],
            "interventions": interventions,
            "mode": mode,
            "population": population,
        }
        # A metadata-only envelope reuses original receipt headers and input geometry.
        # No proposed post-change values are claimed: this preview is discarded.
        envelope = {
            "version": "recursive-population-graph-intervention-v1",
            "base": base,
            "change": change,
            "samples": [b["sample"] for b in base["bindings"]],
            "inputs": [(b["node"], b["inputs"]) for b in base["bindings"]],
        }
        preflight_population_graph_intervention(envelope, limits=limits)
        original = recursive_population_graph_from_dict(base, limits=limits)
        return intervene_recursive_population_graph(
            original, intervention_from_dict(change, limits=limits)
        )
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError(
            "population change requires complete original metadata and semantics"
        ) from exc
