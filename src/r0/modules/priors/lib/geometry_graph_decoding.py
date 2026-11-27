"""Reopen geometry archives with caller limits and complete construction replay."""

from dataclasses import fields
from typing import Any

from r0.modules.priors.lib.cauchy_graph import CauchyGraphPrior
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.functional_causal_decoding import functional_array, functional_record
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.geometry_graph_request import GeometryGraphRequest
from r0.modules.priors.lib.geometry_graph_sampling import sample_geometry_graph
from r0.modules.priors.lib.joint_root_plans import JointRootPrior
from r0.modules.priors.lib.joint_width_plans import JointWidthPrior
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.sampled_geometry_graph import SampledGeometryGraph
from r0.modules.priors.lib.sampled_graph_decoding import (
    construction_matches,
    decode_joint_function_prior,
)

_DEFAULT_LIMITS = CausalLimits()


def decode_geometry_graph_request(data: Any, limits: CausalLimits) -> GeometryGraphRequest:
    raw = functional_record(data, GeometryGraphRequest)
    declared = CausalLimits(**functional_record(raw["limits"], CausalLimits))
    if any(getattr(declared, f.name) > getattr(limits, f.name) for f in fields(CausalLimits)):
        raise ValueError("geometry graph declared limits exceed caller limits")
    raw["limits"] = declared
    raw["raw_limits"] = RawDrawLimits(**functional_record(raw["raw_limits"], RawDrawLimits))
    raw["topology"] = CauchyGraphPrior(**functional_record(raw["topology"], CauchyGraphPrior))
    raw["width_prior"] = JointWidthPrior(**functional_record(raw["width_prior"], JointWidthPrior))
    task = functional_record(raw["task"], LinearRecipe)
    if "class_offsets" in task:
        task["class_offsets"] = functional_array(task["class_offsets"])
    raw["task"] = LinearRecipe(**task)
    raw["function_prior"] = decode_joint_function_prior(raw["function_prior"])
    root = functional_record(raw["root_prior"], JointRootPrior)
    root["families"] = functional_array(root["families"])
    root["function"] = decode_joint_function_prior(root["function"])
    raw["root_prior"] = JointRootPrior(**root)
    return GeometryGraphRequest(**raw)


def geometry_graph_from_dict(
    data: object, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> SampledGeometryGraph:
    if type(limits) is not CausalLimits:
        raise ValueError("geometry decoding requires caller-owned limits")
    admit_function_metadata(data, limits)
    try:
        record = functional_record(data, SampledGeometryGraph)
        if record.get("version") != "sampled-geometry-graph-v1":
            raise ValueError("unsupported sampled geometry graph version")
        request = decode_geometry_graph_request(record["request"], limits)
        admit_function_metadata(data, request.limits)
        initial = functional_array(record["initial"])
        if len(initial) != 4:
            raise ValueError("geometry archive requires four initial snapshots")
        snapshots = tuple(
            GeneratorSnapshot(**functional_record(s, GeneratorSnapshot)) for s in initial
        )
        streams = tuple(s.restore(byte_limit=request.raw_limits.state_bytes) for s in snapshots)
        result = sample_geometry_graph(
            request,
            named_rng=streams[0],
            raw_rng=streams[1],
            calibration_rng=streams[2],
            fit_rng=streams[3],
        )
        if not construction_matches(result, data):
            raise ValueError("geometry graph archive differs from complete replay")
        return result
    except (TypeError, KeyError, OverflowError, RecursionError) as exc:
        raise ValueError("invalid geometry graph archive") from exc
