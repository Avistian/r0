"""Bounded reconstruction of retained root bases, fits and calibration outputs."""

from dataclasses import fields as dataclass_fields
from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.joint_function_decoding import joint_function_plan_from_dict
from r0.modules.priors.lib.joint_function_realization_decoding import (
    joint_function_realization_from_dict,
)
from r0.modules.priors.lib.joint_root_admission import admit_root_realization
from r0.modules.priors.lib.joint_root_base import JointRootBase, JointRootBaseSample
from r0.modules.priors.lib.joint_root_decoding import joint_root_plan_from_dict
from r0.modules.priors.lib.joint_root_plans import JointRootPlan
from r0.modules.priors.lib.joint_root_realization import JointRootRealization
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord
from r0.modules.priors.lib.weight_geometry import Matrix
from r0.modules.priors.lib.weight_vectors import (
    PolynomialWeightVector,
    UniformWeightVector,
    weight_vector_from_dict,
)

_DEFAULT_LIMITS = CausalLimits()


def _record(value: Any, record_type: type[Any], version: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("root realization records must be JSON objects")
    result = cast(dict[str, Any], value)
    if set(result) != {field.name for field in dataclass_fields(record_type)}:
        raise ValueError("root realization record must retain exactly its required fields")
    if result.get("version") != version:
        raise ValueError("unsupported root realization record version")
    return dict(result)


def _array(value: Any, count: int) -> tuple[Any, ...]:
    if type(value) not in (list, tuple) or len(value) != count:
        raise ValueError("root realization arrays must match their exact planned geometry")
    return tuple(value)


def _matrix(value: Any, rows: int, columns: int) -> Matrix:
    return tuple(_array(row, columns) for row in _array(value, rows))


def _base(value: Any, plan: JointRootPlan) -> JointRootBase:
    fields = _record(value, JointRootBase, "joint-root-base-v1")
    if type(fields["width"]) is not int or (
        fields["family"],
        fields["width"],
        fields["covariance_base"],
    ) != (plan.family, plan.width, plan.covariance_base):
        raise ValueError("root base differs from its owned plan")
    if plan.family == "covariance":
        assert plan.weights is not None
        kind = UniformWeightVector if plan.weights.family == "uniform" else PolynomialWeightVector
        version = (
            "uniform-weight-vector-v1"
            if kind is UniformWeightVector
            else "polynomial-weight-vector-v1"
        )
        weights = _record(fields["weights"], kind, version)
        weights["draws"] = _array(weights["draws"], plan.width)
        if kind is PolynomialWeightVector:
            weights["permutation"] = _array(weights["permutation"], plan.width)
        fields["weights"] = weight_vector_from_dict(weights)
        for name in ("gaussian_factor", "factor"):
            fields[name] = _matrix(fields[name], plan.width, plan.width)
    return JointRootBase(**fields)


def joint_root_realization_from_dict(
    data: dict[str, Any],
    *,
    decisions: JointSamplingRecord,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> JointRootRealization:
    """Check full raw payload and planned aggregate before any numerical reconstruction."""
    if type(decisions) is not JointSamplingRecord or type(limits) is not CausalLimits:
        raise ValueError("root reconstruction requires frozen decisions and caller-owned limits")
    admit_function_metadata(data, limits)
    try:
        return _decode(data, decisions, limits)
    except (KeyError, TypeError) as error:
        raise ValueError("root realization contains missing or malformed fields") from error


def _decode(
    data: dict[str, Any], decisions: JointSamplingRecord, limits: CausalLimits
) -> JointRootRealization:
    fields = _record(data, JointRootRealization, "joint-root-realization-v1")
    plan = joint_root_plan_from_dict(fields["plan"], decisions=decisions, limits=limits)
    admit_root_realization(plan, decisions, limits)
    function = fields["function"]
    if type(function) is not dict:
        raise ValueError("root function must be a realization object")
    function = cast(dict[str, Any], function)
    function_plan = joint_function_plan_from_dict(
        function["plan"], decisions=decisions, limits=limits
    )
    if function_plan != plan.function:
        raise ValueError("root function differs from its owned plan")
    sample = _record(fields["calibration"], JointRootBaseSample, "joint-root-base-sample-v1")
    sample["base"] = _base(sample["base"], plan)
    for name in ("draws", "values"):
        sample[name] = _matrix(sample[name], plan.calibration_rows, plan.width)
    sample["radial_uniform"] = _array(
        sample["radial_uniform"], plan.calibration_rows if plan.family == "ball" else 0
    )
    fields["plan"] = plan
    fields["calibration"] = JointRootBaseSample(**sample)
    fields["outputs"] = _matrix(fields["outputs"], plan.calibration_rows, plan.width)
    fields["function"] = joint_function_realization_from_dict(
        function, decisions=decisions, limits=limits
    )
    result = JointRootRealization(**fields)
    result.validate(decisions)
    return result
