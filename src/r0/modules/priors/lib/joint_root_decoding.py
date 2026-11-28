"""Bounded reconstruction of a source root's base and full function plan."""

from dataclasses import fields as dataclass_fields
from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.joint_function_decoding import (
    decode_function_prior,
    joint_function_plan_from_dict,
)
from r0.modules.priors.lib.joint_root_plans import JointRootPlan, JointRootPrior
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord
from r0.modules.priors.lib.joint_weight_plans import JointWeightPlan, joint_weight_plan_from_dict

_DEFAULT_LIMITS = CausalLimits()


def _record(value: Any, record_type: type[Any]) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("root plan records must be JSON objects")
    fields = cast(dict[str, Any], value)
    if set(fields) - {field.name for field in dataclass_fields(record_type)}:
        raise ValueError("root plan record contains unknown fields")
    return dict(fields)


def _array(value: Any, maximum: int) -> tuple[Any, ...]:
    if type(value) not in (list, tuple) or len(value) > maximum:
        raise ValueError("root plan arrays must obey their cardinality limits")
    return tuple(value)


def joint_root_plan_from_dict(
    data: dict[str, Any],
    *,
    decisions: JointSamplingRecord,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> JointRootPlan:
    """Admit all raw metadata before decoding; root realization needs a separate gate."""
    if type(decisions) is not JointSamplingRecord or type(limits) is not CausalLimits:
        raise ValueError("root decoding requires frozen decisions and caller-owned limits")
    admit_function_metadata(data, limits)
    fields = _record(data, JointRootPlan)
    if fields.get("version") != "joint-root-plan-v1":
        raise ValueError("unsupported joint root plan version")
    fields["operation"] = _array(fields["operation"], 16)
    prior = _record(fields.get("prior", {}), JointRootPrior)
    if "families" in prior:
        prior["families"] = _array(prior["families"], 4)
    prior["function"] = decode_function_prior(prior.get("function", {}))
    fields["prior"] = JointRootPrior(**prior)
    if fields.get("weights") is not None:
        fields["weights"] = joint_weight_plan_from_dict(
            _record(fields["weights"], JointWeightPlan), decisions=decisions
        )
    fields["function"] = joint_function_plan_from_dict(
        fields["function"], decisions=decisions, limits=limits
    )
    plan = JointRootPlan(**fields)
    plan.validate(decisions)
    return plan
