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
from r0.modules.priors.lib.functional_causal import (
    CalibratedFunctionalCausalRecipe,
    FunctionalCausalNode,
    FunctionalCausalRecipe,
)
from r0.modules.priors.lib.inputs import InputDistribution
from r0.modules.priors.lib.joint_root_base import JointRootBase
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.node_pipeline import (
    CausalNodePipeline,
    NumericalConverter,
)
from r0.modules.priors.lib.prototype_root_base import (
    PrototypeRootBase,
    prototype_root_base_from_dict,
)
from r0.modules.priors.lib.source_root_base import SourceRootBase, source_root_base_from_dict
from r0.modules.priors.lib.weight_vectors import (
    PolynomialWeightVector,
    UniformWeightVector,
    weight_vector_from_dict,
)

_DEFAULT_LIMITS = CausalLimits()


def functional_record(value: Any, kind: type[Any]) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("functional graph records must be objects")
    raw = cast(dict[str, Any], value)
    if set(raw) - {f.name for f in dataclass_fields(kind)}:
        raise ValueError("functional graph record contains unknown fields")
    return dict(raw)


def functional_array(value: Any) -> tuple[Any, ...]:
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
    fields = functional_record(raw, kinds[version] if version is not None else CausalActivation)
    fields["parameters"] = tuple(
        functional_array(pair) for pair in functional_array(fields.get("parameters", ()))
    )
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
            fields[key] = functional_array(fields[key])
    return activation_from_dict(fields)


def functional_pipeline(value: Any) -> CausalNodePipeline:
    fields = functional_record(value, CausalNodePipeline)
    fields["stages"] = tuple(_activation(a) for a in functional_array(fields.get("stages", ())))
    fields["channel_importance"] = functional_array(fields.get("channel_importance", ()))
    if fields.get("l2") is not None:
        fields["l2"] = _activation(fields["l2"])
    converters: list[NumericalConverter] = []
    for raw in functional_array(fields.get("converters", ())):
        converter = functional_record(raw, NumericalConverter)
        converter["transform"] = _activation(converter.get("transform", {}))
        converters.append(NumericalConverter(**converter))
    fields["converters"] = tuple(converters)
    return CausalNodePipeline(**fields)


def functional_root(
    value: Any, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> JointRootBase | SourceRootBase | PrototypeRootBase:
    if type(value) is dict:
        raw_root = cast(dict[str, Any], value)
        if raw_root.get("version") == "prototype-root-base-v1":
            return prototype_root_base_from_dict(raw_root, limits=limits)
        if raw_root.get("version") == "sampled-source-root-base-v1":
            from r0.modules.priors.lib.sampled_source_root_decoding import (
                sampled_source_root_base_from_dict,
            )

            return sampled_source_root_base_from_dict(raw_root, limits=limits)
        if raw_root.get("version") == "source-root-base-v1":
            return source_root_base_from_dict(raw_root)
    fields = functional_record(value, JointRootBase)
    for key in ("factor", "gaussian_factor"):
        if fields.get(key) is not None:
            fields[key] = tuple(functional_array(row) for row in functional_array(fields[key]))
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
        weights = functional_record(raw, kind)
        for key in ("draws", "permutation"):
            if key in weights:
                weights[key] = functional_array(weights[key])
        fields["weights"] = weight_vector_from_dict(weights)
    return JointRootBase(**fields)


def functional_recipe_from_dict(
    data: dict[str, Any], *, limits: CausalLimits = _DEFAULT_LIMITS
) -> FunctionalCausalRecipe:
    """Caller limits bound raw decoding; payload limits can only tighten those caps."""
    if type(limits) is not CausalLimits:
        raise ValueError("functional decoding requires caller-owned limits")
    admit_function_metadata(data, limits)
    if type(data) is not dict:
        raise ValueError("functional graph records must be objects")
    if data.get("version") == "standalone-network-innovations-v1":
        from r0.modules.priors.lib.network_innovation_decoding import network_innovation_from_dict

        return network_innovation_from_dict(data, limits=limits)
    from r0.modules.priors.lib.fitted_fit_decoding import decode_node_fit
    from r0.modules.priors.lib.fitted_graph import FittedFunctionalCausalRecipe
    from r0.modules.priors.lib.fitted_parameter_decoding import sampled_fitted_plan_from_dict
    from r0.modules.priors.lib.fitted_population_decoding import population_node_fit_from_dict
    from r0.modules.priors.lib.fitted_population_graph import (
        PopulationFittedGraph,
        SupportPopulationFunctionalRecipe,
    )
    from r0.modules.priors.lib.population_parameter_decoding import (
        sampled_population_plan_from_dict,
    )
    from r0.modules.priors.lib.sampled_fitted_graph import SampledFittedFunctionalCausalRecipe
    from r0.modules.priors.lib.sampled_population_graph import SampledPopulationFittedGraph
    from r0.modules.priors.lib.xgboost_fit_decoding import xgboost_fit_from_dict
    from r0.modules.priors.lib.xgboost_fit_graph import XGBoostFittedGraph, XGBoostNodeFit

    try:
        kind = (
            XGBoostFittedGraph
            if data.get("version") == "xgboost-fitted-functional-causal-v1"
            else SampledPopulationFittedGraph
            if data.get("version") == "sampled-population-fitted-functional-causal-v1"
            else SupportPopulationFunctionalRecipe
            if data.get("version") == "support-population-functional-causal-v1"
            else PopulationFittedGraph
            if data.get("version") == "population-fitted-functional-causal-v1"
            else SampledFittedFunctionalCausalRecipe
            if data.get("version") == "sampled-fitted-functional-causal-v1"
            else FittedFunctionalCausalRecipe
            if data.get("version") == "fitted-functional-causal-v1"
            else CalibratedFunctionalCausalRecipe
            if data.get("version") == "functional-causal-v2"
            else FunctionalCausalRecipe
        )
        fields = functional_record(data, kind)
        declared = CausalLimits(**functional_record(fields.get("limits", {}), CausalLimits))
        effective = CausalLimits(
            **{
                f.name: min(getattr(limits, f.name), getattr(declared, f.name))
                for f in dataclass_fields(CausalLimits)
            }
        )
        admit_function_metadata(data, effective)
        task = functional_record(fields["task"], LinearRecipe)
        if "class_offsets" in task:
            task["class_offsets"] = functional_array(task["class_offsets"])
        fields["task"] = LinearRecipe(**task)
        raw_nodes = functional_array(fields["nodes"])
        if len(raw_nodes) > effective.nodes:
            raise ValueError("functional graph node limit exceeded")
        node_limits = effective
        if kind is XGBoostFittedGraph:
            from r0.modules.priors.lib.xgboost_graph_admission import raw_fit_reservation

            node_limits = raw_fit_reservation(fields.get("fits", ()), effective, len(raw_nodes))
        nodes: list[FunctionalCausalNode] = []
        for raw in raw_nodes:
            nodes.append(functional_node_from_dict(raw, limits=node_limits))
        fields["nodes"] = tuple(nodes)
        for key in ("observed", "target"):
            fields[key] = tuple(functional_array(ref) for ref in functional_array(fields[key]))
        fields["limits"] = effective
        if kind is XGBoostFittedGraph:
            # Validate complete ordinary graph against the reserved remainder before fit replay.
            FunctionalCausalRecipe(
                fields["task"], tuple(nodes), fields["observed"], fields["target"], node_limits
            )
            raw_fits = functional_array(fields.get("fits", ()))
            fits: list[XGBoostNodeFit] = []
            for value in raw_fits:
                entry = functional_record(value, XGBoostNodeFit)
                if set(entry) != {f.name for f in dataclass_fields(XGBoostNodeFit)}:
                    raise ValueError("XGBoost node fit requires complete fields")
                entry["realization"] = xgboost_fit_from_dict(entry["realization"], limits=effective)
                fits.append(XGBoostNodeFit(**entry))
            fields["fits"] = tuple(fits)
        if issubclass(kind, FittedFunctionalCausalRecipe):
            fields["fits"] = tuple(
                decode_node_fit(f) for f in functional_array(fields.get("fits", ()))
            )
        if kind is SampledFittedFunctionalCausalRecipe:
            fields["parameter_samples"] = tuple(
                (pair[0], sampled_fitted_plan_from_dict(pair[1], limits=effective))
                for pair in functional_array(fields.get("parameter_samples", ()))
                if _parameter_pair(pair)
            )
        if issubclass(kind, PopulationFittedGraph):
            fields["fits"] = tuple(
                population_node_fit_from_dict(f, limits=effective)
                for f in functional_array(fields.get("fits", ()))
            )
        if kind is SampledPopulationFittedGraph:
            fields["parameter_samples"] = tuple(
                (pair[0], sampled_population_plan_from_dict(pair[1], limits=effective))
                for pair in functional_array(fields.get("parameter_samples", ()))
                if _parameter_pair(pair)
            )
        return kind(**fields)
    except (TypeError, KeyError) as error:
        raise ValueError("malformed functional graph record") from error


def functional_node_from_dict(
    raw: Any, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> FunctionalCausalNode:
    """Decode an equation after the enclosing record has passed metadata admission."""
    node = functional_record(raw, FunctionalCausalNode)
    node["parents"] = functional_array(node["parents"])
    node["function"] = decode_admitted_function(node["function"], limits=limits)
    if node.get("root") is not None:
        node["root"] = functional_root(node["root"], limits=limits)
    node["noise_scale"] = functional_array(node.get("noise_scale", ()))
    exogenous = functional_record(node.get("exogenous", {}), ExogenousDistribution)
    exogenous["inputs"] = InputDistribution(
        **functional_record(exogenous.get("inputs", {}), InputDistribution)
    )
    exogenous["factor"] = tuple(
        functional_array(row) for row in functional_array(exogenous.get("factor", ()))
    )
    node["exogenous"] = ExogenousDistribution(**exogenous)
    node["noise_response"] = NoiseResponse(
        **functional_record(node.get("noise_response", {}), NoiseResponse)
    )
    if node.get("pipeline") is not None:
        node["pipeline"] = functional_pipeline(node["pipeline"])
    return FunctionalCausalNode(**node)


def _parameter_pair(pair: Any) -> bool:
    entry = functional_array(pair)
    if len(entry) != 2 or type(entry[0]) is not int:
        raise ValueError("sampled fitted parameter bindings require node/sample pairs")
    return True
