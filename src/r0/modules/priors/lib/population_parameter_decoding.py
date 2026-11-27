"""Bounded population-law decoding, including exact effective-plan/raw replay."""

from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.fitted_parameter_decoding import (
    decode_fitted_parameter_laws,
    decode_scalar_parameter_draw,
)
from r0.modules.priors.lib.fitted_population_plan import PopulationTreePlan
from r0.modules.priors.lib.fitted_tree_decoding import fitted_array, fitted_object
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.population_parameter_prior import PopulationTreeParameterPrior
from r0.modules.priors.lib.population_parameter_sampling import SampledPopulationTreePlan
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits

_DEFAULT_LIMITS = CausalLimits()


def population_parameter_prior_from_dict(
    data: Any,
    *,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> PopulationTreeParameterPrior:
    admit_function_metadata(data, limits)
    values = fitted_object(data, PopulationTreeParameterPrior)
    values["base"] = PopulationTreePlan(**fitted_object(values.get("base"), PopulationTreePlan))
    values["laws"] = decode_fitted_parameter_laws(values.get("laws", ()))
    if "raw_limits" in values:
        values["raw_limits"] = RawDrawLimits(**fitted_object(values["raw_limits"], RawDrawLimits))
    prior = PopulationTreeParameterPrior(**values)
    prior.admit(limits)
    return prior


def sampled_population_plan_from_dict(
    data: Any,
    *,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> SampledPopulationTreePlan:
    admit_function_metadata(data, limits)
    values = fitted_object(data, SampledPopulationTreePlan)
    values["prior"] = population_parameter_prior_from_dict(values["prior"], limits=limits)
    values["plan"] = PopulationTreePlan(**fitted_object(values["plan"], PopulationTreePlan))
    for key in ("initial_rng", "final_rng"):
        values[key] = GeneratorSnapshot(**fitted_object(values[key], GeneratorSnapshot))
    values["raw"] = tuple(decode_scalar_parameter_draw(v) for v in fitted_array(values["raw"], 8))
    values["unused_parameters"] = fitted_array(values.get("unused_parameters", ()), 1)
    return SampledPopulationTreePlan(**values)
