"""Bounded JSON decoding for authored frozen-function DAGs."""

from dataclasses import fields as dataclass_fields
from typing import Any, cast

from r0.modules.priors.lib.calibrated_activations import (
    CalibratedActivation,
    CalibratedStandardization,
    activation_from_dict,
)
from r0.modules.priors.lib.calibrated_transforms import CalibratedTransform
from r0.modules.priors.lib.causal_activations import CausalActivation
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.exogenous import ExogenousDistribution, NoiseResponse
from r0.modules.priors.lib.frozen_function_decoding import decode_admitted_function
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.functional_causal import FunctionalCausalNode, FunctionalCausalRecipe
from r0.modules.priors.lib.inputs import InputDistribution
from r0.modules.priors.lib.joint_root_base import JointRootBase
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.node_pipeline import (
    CausalNodePipeline,
    NumericalConverter,
)
from r0.modules.priors.lib.weight_vectors import (
    PolynomialWeightVector,
    UniformWeightVector,
    weight_vector_from_dict,
)

_DEFAULT_LIMITS = CausalLimits()


def _record(value: Any, kind: type[Any]) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("functional graph records must be objects")
    raw = cast(dict[str, Any], value)
    if set(raw) - {f.name for f in dataclass_fields(kind)}:
        raise ValueError("functional graph record contains unknown fields")
    return dict(raw)


def _array(value: Any) -> tuple[Any, ...]:
    if type(value) not in (list, tuple):
        raise ValueError("functional graph arrays must be lists or tuples")
    return tuple(value)


def _activation(value: Any) -> CausalActivation:
    if type(value) is not dict:
        raise ValueError("functional activation must be an object")
    raw = cast(dict[str, Any], value)
    kinds: dict[str, type[CausalActivation]] = {
        "independent-standard-activation-v1": CalibratedActivation,
        "independent-standardization-v1": CalibratedStandardization,
        "independent-transform-v1": CalibratedTransform,
    }
    version = raw.get("calibration_version")
    if version is not None and (type(version) is not str or version not in kinds):
        raise ValueError("unsupported functional activation calibration version")
    fields = _record(raw, kinds[version] if version is not None else CausalActivation)
    fields["parameters"] = tuple(_array(pair) for pair in _array(fields.get("parameters", ())))
    for key in (
        "input_mean",
        "input_inverse_std",
        "shift",
        "output_mean",
        "output_inverse_std",
        "shift_rows",
        "mean",
        "inverse_std",
        "minimum",
        "maximum",
    ):
        if key in fields:
            fields[key] = _array(fields[key])
    return activation_from_dict(fields)


def _pipeline(value: Any) -> CausalNodePipeline:
    fields = _record(value, CausalNodePipeline)
    fields["stages"] = tuple(_activation(a) for a in _array(fields.get("stages", ())))
    fields["channel_importance"] = _array(fields.get("channel_importance", ()))
    if fields.get("l2") is not None:
        fields["l2"] = _activation(fields["l2"])
    converters: list[NumericalConverter] = []
    for raw in _array(fields.get("converters", ())):
        converter = _record(raw, NumericalConverter)
        converter["transform"] = _activation(converter.get("transform", {}))
        converters.append(NumericalConverter(**converter))
    fields["converters"] = tuple(converters)
    return CausalNodePipeline(**fields)


def _root(value: Any) -> JointRootBase:
    fields = _record(value, JointRootBase)
    for key in ("factor", "gaussian_factor"):
        if fields.get(key) is not None:
            fields[key] = tuple(_array(row) for row in _array(fields[key]))
    if fields.get("weights") is not None:
        raw = fields["weights"]
        if type(raw) is not dict:
            raise ValueError("root weights must be an object")
        raw = cast(dict[str, Any], raw)
        kind = (
            PolynomialWeightVector
            if raw.get("version") == "polynomial-weight-vector-v1"
            else UniformWeightVector
        )
        weights = _record(raw, kind)
        for key in ("draws", "permutation"):
            if key in weights:
                weights[key] = _array(weights[key])
        fields["weights"] = weight_vector_from_dict(weights)
    return JointRootBase(**fields)


def functional_recipe_from_dict(
    data: dict[str, Any], *, limits: CausalLimits = _DEFAULT_LIMITS
) -> FunctionalCausalRecipe:
    """Caller limits bound raw decoding; payload limits can only tighten those caps."""
    if type(limits) is not CausalLimits:
        raise ValueError("functional decoding requires caller-owned limits")
    admit_function_metadata(data, limits)
    try:
        fields = _record(data, FunctionalCausalRecipe)
        declared = CausalLimits(**_record(fields.get("limits", {}), CausalLimits))
        effective = CausalLimits(
            **{
                f.name: min(getattr(limits, f.name), getattr(declared, f.name))
                for f in dataclass_fields(CausalLimits)
            }
        )
        admit_function_metadata(data, effective)
        task = _record(fields["task"], LinearRecipe)
        if "class_offsets" in task:
            task["class_offsets"] = _array(task["class_offsets"])
        fields["task"] = LinearRecipe(**task)
        raw_nodes = _array(fields["nodes"])
        if len(raw_nodes) > effective.nodes:
            raise ValueError("functional graph node limit exceeded")
        nodes: list[FunctionalCausalNode] = []
        for raw in raw_nodes:
            node = _record(raw, FunctionalCausalNode)
            node["parents"] = _array(node["parents"])
            node["function"] = decode_admitted_function(node["function"])
            if node.get("root") is not None:
                node["root"] = _root(node["root"])
            node["noise_scale"] = _array(node.get("noise_scale", ()))
            exogenous = _record(node.get("exogenous", {}), ExogenousDistribution)
            exogenous["inputs"] = InputDistribution(
                **_record(exogenous.get("inputs", {}), InputDistribution)
            )
            exogenous["factor"] = tuple(_array(row) for row in _array(exogenous.get("factor", ())))
            node["exogenous"] = ExogenousDistribution(**exogenous)
            node["noise_response"] = NoiseResponse(
                **_record(node.get("noise_response", {}), NoiseResponse)
            )
            if node.get("pipeline") is not None:
                node["pipeline"] = _pipeline(node["pipeline"])
            nodes.append(FunctionalCausalNode(**node))
        fields["nodes"] = tuple(nodes)
        for key in ("observed", "target"):
            fields[key] = tuple(_array(ref) for ref in _array(fields[key]))
        fields["limits"] = effective
        return FunctionalCausalRecipe(**fields)
    except (TypeError, KeyError) as error:
        raise ValueError("malformed functional graph record") from error
