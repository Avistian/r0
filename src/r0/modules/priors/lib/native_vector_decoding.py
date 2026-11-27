"""Bounded authored and sampled heterogeneous vector archive decoding."""

from dataclasses import fields
from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.functional_causal_decoding import (
    functional_array,
    functional_node_from_dict,
    functional_record,
)
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.joint_root_plans import JointRootPrior
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.native_causal import FrequencyVariableDomain, VariableDomain
from r0.modules.priors.lib.native_graph_decoding import native_domain_prior_from_dict
from r0.modules.priors.lib.native_vector_records import NativeVectorNode, NativeVectorRecipe
from r0.modules.priors.lib.native_vector_request import NativeVectorGraphRequest
from r0.modules.priors.lib.native_vector_sampling import sample_native_vector_graph
from r0.modules.priors.lib.parent_composition_plans import ParentCompositionPrior
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.sampled_graph_decoding import (
    construction_matches,
    decode_joint_function_prior,
)
from r0.modules.priors.lib.sampled_native_vector_graph import SampledNativeVectorGraph


def _limits(data: object, caller: CausalLimits) -> CausalLimits:
    declared = CausalLimits(**functional_record(data, CausalLimits))
    if any(getattr(declared, f.name) > getattr(caller, f.name) for f in fields(caller)):
        raise ValueError("native vector declared limits exceed caller limits")
    return declared


def _domain(data: object) -> VariableDomain:
    if type(data) is not dict:
        raise ValueError("native vector domain requires an object")
    record = cast(dict[str, Any], data)
    kind = (
        FrequencyVariableDomain
        if record.get("version") == "native-frequency-domain-v2"
        else VariableDomain
    )
    args = functional_record(record, kind)
    for name in ("vocabulary", "thresholds", "frequencies"):
        if name in args:
            args[name] = functional_array(args[name])
    return kind(**args)


def native_vector_recipe_from_dict(data: object, *, limits: CausalLimits) -> NativeVectorRecipe:
    admit_function_metadata(data, limits)
    try:
        args = functional_record(data, NativeVectorRecipe)
        declared = _limits(args.get("limits", {}), limits)
        admit_function_metadata(data, declared)
        task = functional_record(args["task"], LinearRecipe)
        if "class_offsets" in task:
            task["class_offsets"] = functional_array(task["class_offsets"])
        nodes: list[NativeVectorNode] = []
        raw_nodes = functional_array(args["nodes"])
        if not 1 <= len(raw_nodes) <= declared.nodes:
            raise ValueError("native vector node limit exceeded before decoding")
        for raw in raw_nodes:
            n = functional_record(raw, NativeVectorNode)
            n["equation"] = functional_node_from_dict(n["equation"])
            raw_domains = functional_array(n["domains"])
            if not 1 <= len(raw_domains) <= declared.node_channels:
                raise ValueError("native vector variable count limit exceeded before decoding")
            n["domains"] = tuple(_domain(d) for d in raw_domains)
            nodes.append(NativeVectorNode(**n))
        args.update(
            task=LinearRecipe(**task),
            nodes=tuple(nodes),
            limits=declared,
            observed=tuple(functional_array(r) for r in functional_array(args["observed"])),
            target=functional_array(args["target"]),
        )
        return NativeVectorRecipe(**args)
    except (KeyError, TypeError, RecursionError, OverflowError) as error:
        raise ValueError("malformed native vector recipe") from error


def native_vector_request_from_dict(
    data: object, *, limits: CausalLimits
) -> NativeVectorGraphRequest:
    admit_function_metadata(data, limits)
    args = functional_record(data, NativeVectorGraphRequest)
    args["limits"] = _limits(args.get("limits", {}), limits)
    args["raw_limits"] = RawDrawLimits(
        **functional_record(args.get("raw_limits", {}), RawDrawLimits)
    )
    args["domains"] = tuple(
        tuple(native_domain_prior_from_dict(d) for d in functional_array(ds))
        for ds in functional_array(args["domains"])
    )
    for key in ("parents", "observed"):
        args[key] = tuple(functional_array(row) for row in functional_array(args[key]))
    args["target"] = functional_array(args["target"])
    args["function_prior"] = decode_joint_function_prior(args["function_prior"])
    root = functional_record(args["root_prior"], JointRootPrior)
    root["families"] = functional_array(root["families"])
    root["function"] = decode_joint_function_prior(root["function"])
    args["root_prior"] = JointRootPrior(**root)
    if args.get("composition") is not None:
        args["composition"] = ParentCompositionPrior(
            **functional_record(args["composition"], ParentCompositionPrior)
        )
    return NativeVectorGraphRequest(**args)


def native_vector_graph_from_dict(
    data: object, *, limits: CausalLimits
) -> SampledNativeVectorGraph:
    admit_function_metadata(data, limits)
    try:
        record = functional_record(data, SampledNativeVectorGraph)
        request = native_vector_request_from_dict(record["request"], limits=limits)
        initial = functional_array(record["initial"])
        if len(initial) != 4:
            raise ValueError("native vector construction requires four initial streams")
        streams = tuple(
            GeneratorSnapshot(**functional_record(s, GeneratorSnapshot)).restore(
                byte_limit=request.raw_limits.state_bytes
            )
            for s in initial
        )
        graph = sample_native_vector_graph(
            request,
            named_rng=streams[0],
            raw_rng=streams[1],
            calibration_rng=streams[2],
            fit_rng=streams[3],
        )
        if not construction_matches(graph, data):
            raise ValueError("native vector archive differs from complete replay")
        return graph
    except (KeyError, TypeError, RecursionError, OverflowError) as error:
        raise ValueError("malformed sampled native vector graph") from error
