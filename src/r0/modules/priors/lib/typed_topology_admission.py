"""Finite geometry and adapter support reservation before topology or mechanism draws."""

from dataclasses import replace
from typing import TYPE_CHECKING

from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.joint_hyperprior_request import JointHyperpriorGraphRequest
from r0.modules.priors.lib.joint_hyperprior_support import function_support
from r0.modules.priors.lib.joint_pipeline_plans import JointInnovationPipelinePrior

if TYPE_CHECKING:
    from r0.modules.priors.lib.typed_topology_request import TypedTopologyRequest


def maximum_typed_width(request: "TypedTopologyRequest") -> int:
    latent = request.width_prior.high - 1
    if request.native is not None:
        k = request.native.maximum_cardinality()
        return latent * k if request.family == "native-vector" else max(latent, k)
    if request.converted is not None:
        return latent + sum(
            max(1, len(v))
            for v in (*request.converted.vocabularies, request.converted.target_vocabulary)
        )
    outputs = request.task.classes if request.task.task_type == "multiclass" else 1
    return latent + request.task.features + outputs


def admit_typed_topology(request: "TypedTopologyRequest") -> dict[str, int]:
    n, w, rows = (
        request.topology.maximum_nodes,
        maximum_typed_width(request),
        request.calibration_rows,
    )
    # A redirection forest has at most one predecessor: unequal rectangular
    # matrices fit inside the W x W support used by the existing catalog planner.
    proxy = JointHyperpriorGraphRequest(
        request.task,
        request.topology,
        w,
        rows,
        request.root_prior,
        request.function_prior,
        mode=request.mode,
        categorical_policy=request.categorical_policy,
        global_policy=request.global_policy,
        limits=request.limits,
        raw_limits=request.raw_limits,
    )
    base = proxy.admit()
    metadata, storage = function_record_cost(request, request.limits)
    # Geometry: <=four observation draws, one width decision/node, one native
    # permutation. Native domains have <=six named scalar/catalog requests per
    # physical channel plus category frequencies, thresholds and fitting values.
    numbers = 64 * n * w + 8 * rows * n * w + 8 * metadata
    draws = 16 * n * w + 8
    calls = 16 * n * w + 8
    if request.pipeline is not None:
        # Each node gets its own fitted pipeline population (a full prefix).
        # Numerical converters retain at most two fitted scalar constants per
        # physical channel. Weight exponent/noise/importance controls add bounded
        # scalar records; v2 factors add W squared coefficients per node.
        numbers += 8 * rows * n * n * w + 32 * n * w * w
        draws += 8 * n * n * w + 16 * n * w
        calls += 32 * n * w
        if isinstance(request.pipeline, JointInnovationPipelinePrior):
            # Truncation is an explicit bounded rejection law, up to128 rounds,
            # each at most twice requested rows (minimum256), never a graph retry.
            numbers += 256 * max(rows, 256) * n * n * w
            draws += 256 * n * n * w
    if request.converted is not None:
        for vocabulary in (*request.converted.vocabularies, request.converted.target_vocabulary):
            if not vocabulary:
                continue
            prior = request.converted.converter
            # Category transforms and embeddings are separately owned functions.
            # Both supports are retained even though individual modes use a subset.
            for support in (
                function_support(prior.function, w, rows),
                function_support(prior.embeddings.function, w, rows),
            ):
                numbers += support.numbers + 8 * rows * w + 4 * w * w
                draws += support.draws + 2 * w + 8
                calls += support.named_calls + support.named_values + support.choices + 16
            numbers += 8 * rows * w + 16 * w * w
            draws += 4 * w + 16
            calls += 16 * w
    # Record schemas duplicate named request/preference/draw/owner metadata in
    # node registries and envelope replay: same64-leaf allowance as joint v1.
    named = 64 * calls + 512 * n
    extra_parameters = 8 * n * named + 8 * numbers
    raw_bytes = 128 * numbers + 16384 * draws
    extra_storage = (
        8 * storage
        + 8 * n * (128 * named + 8192 * calls)
        + 128 * extra_parameters
        + raw_bytes
        + 32 * max(rows, request.task.rows, 256) * n * w
    )
    result = dict(base)
    result["parameters"] += extra_parameters
    result["estimated_working_bytes"] += extra_storage
    result["raw_numbers"] += numbers
    result["raw_draws"] += draws
    result["raw_working_bytes"] += raw_bytes
    result["maximum_physical_width"] = w
    if (
        result["parameters"] > request.limits.parameters
        or result["estimated_working_bytes"] > request.limits.working_bytes
        or result["raw_numbers"] > request.raw_limits.numbers
        or result["raw_draws"] > request.raw_limits.draws
        or result["raw_working_bytes"] > request.raw_limits.working_bytes
    ):
        raise ValueError("typed topology complete geometry/adapter support exceeds limits")
    return result


def typed_topology_child_limits(request: "TypedTopologyRequest"):
    from r0.modules.priors.lib.topology_hyperprior_admission import admit_topology_hyperprior

    top = admit_topology_hyperprior(request.topology, request.limits, request.raw_limits)
    numbers, storage = function_record_cost(request, request.limits)
    # Outer request/topology/geometry survive alongside the complete child.
    return replace(
        request.limits,
        parameters=request.limits.parameters - top["parameters"] - 4 * numbers,
        working_bytes=request.limits.working_bytes - top["estimated_working_bytes"] - 4 * storage,
    )
