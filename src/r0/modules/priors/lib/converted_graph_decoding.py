"""Bounded complete replay decoding of independently fitted categorical pipelines."""

from dataclasses import fields

from r0.modules.priors.lib.categorical_converter_plans import CategoricalConverterPrior
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.converted_graph_request import ConvertedGraphRequest
from r0.modules.priors.lib.converted_graph_sampling import sample_converted_graph
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.functional_causal_decoding import functional_array, functional_record
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.geometry_topology import decode_geometry_topology
from r0.modules.priors.lib.joint_pipeline_decoding import pipeline_prior_from_dict
from r0.modules.priors.lib.joint_root_plans import JointRootPrior
from r0.modules.priors.lib.joint_width_plans import JointWidthPrior
from r0.modules.priors.lib.parent_composition_plans import ParentCompositionPrior
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.sampled_converted_graph import SampledConvertedGraph
from r0.modules.priors.lib.sampled_graph_decoding import (
    construction_matches,
    decode_joint_function_prior,
)


def root_prior(data: object) -> JointRootPrior:
    args = functional_record(data, JointRootPrior)
    if "families" in args:
        args["families"] = functional_array(args["families"])
    if "function" in args:
        args["function"] = decode_joint_function_prior(args["function"])
    return JointRootPrior(**args)


def converted_request_from_dict(data: object, *, limits: CausalLimits) -> ConvertedGraphRequest:
    admit_function_metadata(data, limits)
    try:
        args = functional_record(data, ConvertedGraphRequest)
        declared = CausalLimits(**functional_record(args.get("limits", {}), CausalLimits))
        if any(getattr(declared, f.name) > getattr(limits, f.name) for f in fields(limits)):
            raise ValueError("converted declared limits exceed caller limits")
        args["limits"] = declared
        args["topology"] = decode_geometry_topology(args["topology"])
        args["raw_limits"] = RawDrawLimits(
            **functional_record(args.get("raw_limits", {}), RawDrawLimits)
        )
        args["vocabularies"] = tuple(
            functional_array(v) for v in functional_array(args["vocabularies"])
        )
        args["target_vocabulary"] = functional_array(args.get("target_vocabulary", ()))
        args["width_prior"] = JointWidthPrior(
            **functional_record(args.get("width_prior", {}), JointWidthPrior)
        )
        if "function_prior" in args:
            args["function_prior"] = decode_joint_function_prior(args["function_prior"])
        if "root_prior" in args:
            args["root_prior"] = root_prior(args["root_prior"])
        if "pipeline_prior" in args:
            args["pipeline_prior"] = pipeline_prior_from_dict(args["pipeline_prior"])
        if "converter_prior" in args:
            prior = functional_record(args["converter_prior"], CategoricalConverterPrior)
            if "modes" in prior:
                prior["modes"] = functional_array(prior["modes"])
            if "function" in prior:
                prior["function"] = decode_joint_function_prior(prior["function"])
            if "embeddings" in prior:
                prior["embeddings"] = root_prior(prior["embeddings"])
            args["converter_prior"] = CategoricalConverterPrior(**prior)
        if args.get("composition") is not None:
            args["composition"] = ParentCompositionPrior(
                **functional_record(args["composition"], ParentCompositionPrior)
            )
        return ConvertedGraphRequest(**args)
    except (KeyError, TypeError, RecursionError, OverflowError) as error:
        raise ValueError("malformed converted graph request") from error


def converted_graph_from_dict(data: object, *, limits: CausalLimits) -> SampledConvertedGraph:
    admit_function_metadata(data, limits)
    try:
        record = functional_record(data, SampledConvertedGraph)
        request = converted_request_from_dict(record["request"], limits=limits)
        initial = functional_array(record["initial"])
        if len(initial) != 4:
            raise ValueError("converted archive requires four initial streams")
        streams = tuple(
            GeneratorSnapshot(**functional_record(s, GeneratorSnapshot)).restore(
                byte_limit=request.raw_limits.state_bytes
            )
            for s in initial
        )
        graph = sample_converted_graph(
            request,
            named_rng=streams[0],
            raw_rng=streams[1],
            calibration_rng=streams[2],
            fit_rng=streams[3],
        )
        if not construction_matches(graph, data):
            raise ValueError("converted archive differs from complete construction replay")
        return graph
    except (KeyError, TypeError, RecursionError, OverflowError) as error:
        raise ValueError("malformed sampled converted graph") from error
