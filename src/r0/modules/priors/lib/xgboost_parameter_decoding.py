"""Pre-admitted exact JSON replay of sampled XGBoost parameter plans."""

from dataclasses import asdict, fields
from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.fitted_parameter_decoding import decode_scalar_parameter_draw
from r0.modules.priors.lib.fitted_tree_decoding import fitted_array, fitted_object
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.xgboost_fit_plan import XGBoostFitPlan
from r0.modules.priors.lib.xgboost_fitting import canonical_xgboost_record
from r0.modules.priors.lib.xgboost_parameter_sampling import (
    SampledXGBoostPlan,
    XGBoostParameterPrior,
)

_DEFAULT = CausalLimits()


def _complete(value: Any, kind: type[Any]) -> dict[str, Any]:
    result = fitted_object(value, kind)
    if set(result) != {field.name for field in fields(kind)}:
        raise ValueError("XGBoost parameter archive requires complete fields")
    return result


def sampled_xgboost_plan_from_dict(
    data: Any, *, limits: CausalLimits = _DEFAULT
) -> SampledXGBoostPlan:
    admit_function_metadata(data, limits)
    try:
        fields = _complete(data, SampledXGBoostPlan)
        fields["prior"] = xgboost_parameter_prior_from_dict(fields["prior"], limits=limits)
        fields["plan"] = XGBoostFitPlan(**_complete(fields["plan"], XGBoostFitPlan))
        for key in ("initial_rng", "final_rng"):
            fields[key] = GeneratorSnapshot(**_complete(fields[key], GeneratorSnapshot))
        fields["raw"] = tuple(
            decode_scalar_parameter_draw(r) for r in fitted_array(fields["raw"], 2)
        )
        result = SampledXGBoostPlan(**fields)
        if canonical_xgboost_record(asdict(result)) != canonical_xgboost_record(data):
            raise ValueError("XGBoost parameter archive differs from exact reconstruction")
        return result
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError, RecursionError) as exc:
        raise ValueError("malformed XGBoost parameter archive") from exc


def xgboost_parameter_prior_from_dict(
    data: Any, *, limits: CausalLimits = _DEFAULT
) -> XGBoostParameterPrior:
    admit_function_metadata(data, limits)
    prior = _complete(data, XGBoostParameterPrior)
    prior["base"] = XGBoostFitPlan(**_complete(prior["base"], XGBoostFitPlan))
    prior["raw_limits"] = RawDrawLimits(**_complete(prior["raw_limits"], RawDrawLimits))
    result = XGBoostParameterPrior(**prior)
    result.admit(limits)
    return result
