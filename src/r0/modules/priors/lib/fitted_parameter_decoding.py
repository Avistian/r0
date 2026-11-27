"""Strict bounded reconstruction of sampled fit laws and replay-bound plans."""

from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.fitted_fit_decoding import decode_fitted_raw_draw
from r0.modules.priors.lib.fitted_parameter_laws import FittedParameterLaw
from r0.modules.priors.lib.fitted_parameter_prior import FittedTreeParameterPrior
from r0.modules.priors.lib.fitted_parameter_sampling import SampledFittedTreePlan
from r0.modules.priors.lib.fitted_tree_decoding import fitted_array, fitted_object
from r0.modules.priors.lib.fitted_tree_plan import FittedTreePlan
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits, RawDrawRecord

_DEFAULT_LIMITS = CausalLimits()


def fitted_parameter_prior_from_dict(
    data: Any, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> FittedTreeParameterPrior:
    admit_function_metadata(data, limits)
    values = fitted_object(data, FittedTreeParameterPrior)
    values["base"] = FittedTreePlan(**fitted_object(values.get("base"), FittedTreePlan))
    laws: list[tuple[str, FittedParameterLaw]] = []
    for pair in fitted_array(values.get("laws", ()), 8):
        entry = fitted_array(pair, 2)
        if len(entry) != 2:
            raise ValueError("parameter laws require name/law pairs")
        law = fitted_object(entry[1], FittedParameterLaw)
        for key in ("values", "probabilities"):
            if key in law:
                law[key] = fitted_array(law[key], 128)
        laws.append((entry[0], FittedParameterLaw(**law)))
    values["laws"] = tuple(laws)
    if "raw_limits" in values:
        values["raw_limits"] = RawDrawLimits(**fitted_object(values["raw_limits"], RawDrawLimits))
    prior = FittedTreeParameterPrior(**values)
    prior.admit(limits)
    return prior


def sampled_fitted_plan_from_dict(
    data: Any, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> SampledFittedTreePlan:
    admit_function_metadata(data, limits)
    values = fitted_object(data, SampledFittedTreePlan)
    values["prior"] = fitted_parameter_prior_from_dict(values["prior"], limits=limits)
    values["plan"] = FittedTreePlan(**fitted_object(values["plan"], FittedTreePlan))
    for key in ("initial_rng", "final_rng"):
        values[key] = GeneratorSnapshot(**fitted_object(values[key], GeneratorSnapshot))
    values["raw"] = tuple(_scalar_raw(v) for v in fitted_array(values["raw"], 8))
    values["unused_parameters"] = fitted_array(values.get("unused_parameters", ()), 1)
    return SampledFittedTreePlan(**values)


def _scalar_raw(data: Any) -> RawDrawRecord:
    fields = fitted_object(data, RawDrawRecord)
    fitted_array(fields.get("values"), 1)
    fitted_array(fields.get("shape"), 0)
    return decode_fitted_raw_draw(fields)
