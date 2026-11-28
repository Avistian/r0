"""Reserve complete finite mechanism support before any of five owned RNG streams."""

from dataclasses import replace
from typing import TYPE_CHECKING

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.joint_hyperprior_support import MechanismSupport, function_support
from r0.modules.priors.lib.preference_limits import PreferenceLimits
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.topology_hyperprior_admission import admit_topology_hyperprior

if TYPE_CHECKING:
    from r0.modules.priors.lib.joint_hyperprior_request import JointHyperpriorGraphRequest


def joint_hyperprior_budget(request: "JointHyperpriorGraphRequest") -> dict[str, int]:
    limits, caps = request.limits, request.raw_limits
    top = admit_topology_hyperprior(request.topology, limits, caps)
    n, w, rows = request.topology.maximum_nodes, request.width, request.calibration_rows
    if w > limits.node_channels or n * w > limits.total_channels:
        raise ValueError("joint hyperprior worst channel geometry exceeds limits")
    root = function_support(request.root_prior.function, w, rows)
    # Base-family and covariance choices/weights are separate from the root function.
    root = root.add(
        MechanismSupport(
            8 * rows * w + 4 * w * w + 8 * w + 32, 2 * w + 8, 4, 4, 6, 48 * w, rows, categorical=2
        )
    )
    edge = (
        function_support(request.function_prior, w, rows)
        if n > request.topology.components
        else MechanismSupport()
    )
    c = request.topology.components
    calls = c * root.named_calls + (n - c) * edge.named_calls
    values = c * root.named_values + (n - c) * edge.named_values
    choices = c * root.choices + (n - c) * edge.choices
    categorical = c * root.categorical + (n - c) * edge.categorical
    # Numeric request/preference/model/draw/event copies. Selection probability
    # vectors have at most 16 retained numbers per choice; numerical leaf and
    # registry bookkeeping have at most 64 per scalar request/value.
    named = 64 * (calls + values) + 16 * choices + 512 * n
    PreferenceLimits().admit(
        draws=calls + 2 * categorical, values=values + 2 * categorical, parameters=named
    )
    from r0.modules.priors.lib.joint_operation_registry import JointRegistryLimits

    JointRegistryLimits().admit(4 * calls + 8 * n, 512 * (4 * calls + 8 * n))
    if max(root.channels, edge.channels, root.floor, edge.floor) >= 2**63:
        raise ValueError("joint hyperprior finite support overflow")
    if max(w, root.width, edge.width) > limits.node_channels:
        raise ValueError("joint hyperprior hidden channel support exceeds limits")
    # Each fitted nonroot draws one complete preceding prefix. Mixture noise uses
    # two draws per channel. Existing explicit truncation has at most128 rounds,
    # each requesting at most twice the population (minimum256); reserve all.
    bounded = request.noise.exogenous.inputs.bound is not None
    factor, fit_rows = (256, max(rows, 256)) if bounded else (2, rows)
    prefix_numbers = factor * fit_rows * w * n * n + 2 * rows * w * n * n
    prefix_draws = factor * w * n * n + 2 * n * n
    raw_numbers = c * root.numbers + (n - c) * edge.numbers + prefix_numbers + top["numbers"]
    raw_draws = c * root.draws + (n - c) * edge.draws + prefix_draws + top["draws"]
    raw_bytes = 128 * raw_numbers + 16384 * raw_draws + 32 * caps.state_bytes
    if raw_numbers > caps.numbers or raw_draws > caps.draws or raw_bytes > caps.working_bytes:
        raise ValueError("joint hyperprior full-support raw reservation exceeds limits")
    metadata, storage = function_record_cost(request, limits)
    function_numbers = c * root.numbers + (n - c) * edge.numbers
    # Existing child admission repeats its complete shared registry per node.
    # Eight copies dominate those four child reservations plus retained envelope,
    # decoder, replay and construction; names/paths get a separate text bound.
    parameters = (
        top["parameters"] + 8 * n * named + 8 * function_numbers + 4 * raw_numbers + 8 * metadata
    )
    metadata_bytes = 8 * n * (128 * named + 8192 * calls) + 8 * storage
    emission_rows = max(request.task.rows, rows, root.floor, edge.floor, 256)
    pipeline = request.pipeline.working_extra_channels if request.pipeline else 0
    work = (
        top["estimated_working_bytes"]
        + metadata_bytes
        + 128 * parameters
        + raw_bytes
        + 32 * emission_rows * (48 * n * w + root.channels + edge.channels + pipeline + 8)
        + 512 * (n * w) ** 2
        + 64 * caps.state_bytes
    )
    if parameters > limits.parameters or work > limits.working_bytes:
        raise ValueError("joint hyperprior full-support construction/emission budget exceeded")
    return dict(
        maximum_nodes=n,
        parameters=parameters,
        estimated_working_bytes=work,
        raw_numbers=raw_numbers,
        raw_draws=raw_draws,
        raw_working_bytes=raw_bytes,
        named_calls=calls + 2 * categorical,
        named_parameters=named,
    )


def joint_hyperprior_child_limits(request: "JointHyperpriorGraphRequest") -> CausalLimits:
    top = admit_topology_hyperprior(request.topology, request.limits, request.raw_limits)
    return replace(
        request.limits,
        parameters=request.limits.parameters - top["parameters"],
        working_bytes=request.limits.working_bytes - top["estimated_working_bytes"],
    )


def joint_hyperprior_child_raw(request: "JointHyperpriorGraphRequest") -> RawDrawLimits:
    top = admit_topology_hyperprior(request.topology, request.limits, request.raw_limits)
    # The topology sampler's complete reservation is retained alongside mechanisms.
    reserved = 128 * top["numbers"] + 16384 * top["draws"] + 4 * request.raw_limits.state_bytes
    return replace(
        request.raw_limits,
        numbers=request.raw_limits.numbers - top["numbers"],
        draws=request.raw_limits.draws - top["draws"],
        working_bytes=request.raw_limits.working_bytes - reserved,
    )
