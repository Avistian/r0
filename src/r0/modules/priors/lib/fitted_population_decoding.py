"""Bounded reconstruction of population fitting custody without backend imports."""

from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.fitted_fit_decoding import decode_fitted_raw_draw
from r0.modules.priors.lib.fitted_population_fit import PopulationTreeFit, admit_population_fit
from r0.modules.priors.lib.fitted_population_graph import PopulationNodeFit
from r0.modules.priors.lib.fitted_population_plan import MixedFitPopulation, PopulationTreePlan
from r0.modules.priors.lib.fitted_tree_decoding import (
    decode_fitted_tree,
    fitted_array,
    fitted_object,
)
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot

_DEFAULT_LIMITS = CausalLimits()


def mixed_population_from_dict(data: Any) -> MixedFitPopulation:
    fields = fitted_object(data, MixedFitPopulation)
    if "categorical_fraction" in fields:
        fields["categorical_fraction"] = fitted_array(fields["categorical_fraction"], 2)
    return MixedFitPopulation(**fields)


def population_fit_from_dict(
    data: Any,
    *,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> PopulationTreeFit:
    admit_function_metadata(data, limits)
    fields = fitted_object(data, PopulationTreeFit)
    fields["plan"] = PopulationTreePlan(**fitted_object(fields.get("plan"), PopulationTreePlan))
    admit_population_fit(fields["plan"], limits)
    fields["function"] = decode_fitted_tree(fields["function"])
    if fields.get("mixed") is not None:
        fields["mixed"] = mixed_population_from_dict(fields["mixed"])
    for key in ("inputs", "targets"):
        fields[key] = tuple(fitted_array(row, 256) for row in fitted_array(fields.get(key), 1024))
    for key in ("backend_seeds", "backend_parameters", "cardinalities"):
        fields[key] = fitted_array(fields.get(key), 256)
    for key in ("initial_rng", "final_rng"):
        fields[key] = tuple(
            GeneratorSnapshot(**fitted_object(v, GeneratorSnapshot))
            for v in fitted_array(fields.get(key), 3)
        )
    fields["raw_draws"] = tuple(
        decode_fitted_raw_draw(v) for v in fitted_array(fields.get("raw_draws"), 2048)
    )
    return PopulationTreeFit(**fields)


def population_node_fit_from_dict(data: Any, *, limits: CausalLimits) -> PopulationNodeFit:
    fields = fitted_object(data, PopulationNodeFit)
    fields["realization"] = population_fit_from_dict(fields.get("realization"), limits=limits)
    return PopulationNodeFit(**fields)
