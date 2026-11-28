"""Bounded reopening compares every archive field against complete construction replay."""

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.functional_causal_decoding import functional_array, functional_record
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.parent_composition_plans import ParentCompositionPrior
from r0.modules.priors.lib.parent_graph_sampling import sample_parent_graph
from r0.modules.priors.lib.sampled_graph_decoding import (
    construction_matches,
    decode_joint_graph_request,
)
from r0.modules.priors.lib.sampled_parent_graph import SampledParentGraph

_DEFAULT_LIMITS = CausalLimits()


def parent_graph_from_dict(
    data: object, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> SampledParentGraph:
    if type(limits) is not CausalLimits:
        raise ValueError("parent graph decoding requires caller-owned limits")
    admit_function_metadata(data, limits)
    try:
        record = functional_record(data, SampledParentGraph)
        if record.get("version") != "sampled-parent-graph-v1":
            raise ValueError("unsupported parent graph archive")
        request = decode_joint_graph_request(record["request"], limits)
        admit_function_metadata(data, request.limits)
        composition = ParentCompositionPrior(
            **functional_record(record["composition"], ParentCompositionPrior)
        )
        initial = functional_array(record["initial"])
        if len(initial) != 4:
            raise ValueError("parent graph requires four initial stream snapshots")
        snapshots = tuple(
            GeneratorSnapshot(**functional_record(s, GeneratorSnapshot)) for s in initial
        )
        streams = tuple(s.restore(byte_limit=request.raw_limits.state_bytes) for s in snapshots)
        result = sample_parent_graph(
            request,
            composition=composition,
            named_rng=streams[0],
            raw_rng=streams[1],
            calibration_rng=streams[2],
            fit_rng=streams[3],
        )
        if not construction_matches(result, data):
            raise ValueError("parent graph archive differs from complete replay")
        return result
    except (TypeError, KeyError, OverflowError, RecursionError) as exc:
        raise ValueError("invalid parent graph archive") from exc
