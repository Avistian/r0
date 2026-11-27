"""Bounded reconstruction of source function plans and their typed child trees."""

from dataclasses import fields as dataclass_fields
from typing import Any, cast

from r0.modules.priors.lib.causal_activations import CausalActivation
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.joint_activation_plans import (
    JointActivationPlan,
    joint_activation_plan_from_dict,
)
from r0.modules.priors.lib.joint_function_plans import JointFunctionPlan
from r0.modules.priors.lib.joint_function_prior import JointFunctionPrior
from r0.modules.priors.lib.joint_matrix_plans import (
    JointMatrixPlan,
    JointMatrixPrior,
    joint_matrix_plan_from_dict,
)
from r0.modules.priors.lib.joint_point_plans import JointPointPlan
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord
from r0.modules.priors.lib.joint_weight_plans import JointWeightPlan, joint_weight_plan_from_dict

_DEFAULT_LIMITS = CausalLimits()
_NESTED: dict[type[Any], dict[str, type[Any] | tuple[type[Any], int]]] = {
    JointFunctionPlan: {
        "prior": JointFunctionPrior,
        "matrices": (JointMatrixPlan, 3),
        "activations": (JointActivationPlan, 4),
        "weights": JointWeightPlan,
        "children": (JointFunctionPlan, 2),
    },
    JointFunctionPrior: {"matrix": JointMatrixPrior},
    JointMatrixPlan: {
        "prior": JointMatrixPrior,
        "weights": JointWeightPlan,
        "points": JointPointPlan,
        "child": JointMatrixPlan,
        "activation": JointActivationPlan,
    },
    JointPointPlan: {"mixture_weights": JointWeightPlan, "factor_weights": JointWeightPlan},
    JointActivationPlan: {"activation": CausalActivation},
}


def _object(value: Any, record_type: type[Any]) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("function child record must be a JSON object")
    fields = cast(dict[str, Any], value)
    if set(fields) - {field.name for field in dataclass_fields(record_type)}:
        raise ValueError("function child record contains unknown fields")
    return fields


def _array(value: Any, maximum: int) -> list[Any] | tuple[Any, ...]:
    if type(value) not in (list, tuple) or len(value) > maximum:
        raise ValueError("function plan arrays must obey their branch cardinality limit")
    return cast(list[Any] | tuple[Any, ...], value)


def _structure(value: Any, record_type: type[Any]) -> None:
    fields = _object(value, record_type)
    for name, schema in _NESTED.get(record_type, {}).items():
        if fields.get(name) is None:
            continue
        if isinstance(schema, tuple):
            child_type, maximum = schema
            for item in _array(fields[name], maximum):
                _structure(item, child_type)
        else:
            _structure(fields[name], schema)


def _prior(data: dict[str, Any]) -> JointFunctionPrior:
    fields = dict(data)
    matrix = dict(fields.get("matrix", {}))
    for name, cap in (("families", 5), ("simple_families", 4)):
        if name in matrix:
            matrix[name] = tuple(_array(matrix[name], cap))
    fields["matrix"] = JointMatrixPrior(**matrix)
    return JointFunctionPrior(**fields)


def _decode(
    data: dict[str, Any], decisions: JointSamplingRecord, *, child: bool = False
) -> JointFunctionPlan:
    fields = dict(data)
    if fields.get("version") != "joint-function-plan-v1":
        raise ValueError("unsupported joint function plan version")
    fields["operation"] = tuple(_array(fields["operation"], 16))
    fields["prior"] = _prior(fields.get("prior", {}))
    parameters = _array(fields.get("parameters", ()), 4)
    if any(len(_array(pair, 2)) != 2 for pair in parameters):
        raise ValueError("function parameter pairs must be two-element arrays")
    fields["parameters"] = tuple(tuple(pair) for pair in parameters)
    children = _array(fields.get("children", ()), 2)
    if child and children:
        raise ValueError("cheap function children cannot contain recursive products")
    fields["children"] = tuple(_decode(item, decisions, child=True) for item in children)
    fields["matrices"] = tuple(
        joint_matrix_plan_from_dict(matrix, decisions=decisions)
        for matrix in _array(fields.get("matrices", ()), 3)
    )
    fields["activations"] = tuple(
        joint_activation_plan_from_dict(activation, decisions=decisions)
        for activation in _array(fields.get("activations", ()), 4)
    )
    if fields.get("weights") is not None:
        fields["weights"] = joint_weight_plan_from_dict(fields["weights"], decisions=decisions)
    plan = JointFunctionPlan(**fields)
    plan.validate(decisions)
    return plan


def joint_function_plan_from_dict(
    data: dict[str, Any],
    *,
    decisions: JointSamplingRecord,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> JointFunctionPlan:
    """Gate raw metadata before coercion; realized arrays need separate admission."""
    if type(decisions) is not JointSamplingRecord or type(limits) is not CausalLimits:
        raise ValueError("function decoding requires frozen decisions and caller-owned limits")
    admit_function_metadata(data, limits)
    _structure(data, JointFunctionPlan)
    return _decode(data, decisions)
