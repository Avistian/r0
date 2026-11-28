"""Guarded feature-axis convolution construction with complete sampled receipts."""

from dataclasses import asdict
from typing import Any, cast

import numpy as np

from r0.modules import priors as p


def _row_workspace(request: p.FeatureConv1DRequest) -> int:
    """Conservative workspace over every admitted geometry/activation catalog."""
    length, channels, maximum = request.input_length, request.input_channels, 0
    for prior in request.layers:
        outgoing, next_channels = 0, 0
        for geometry in prior.geometries:
            # The request has already validated every channel/geometry combination.
            positions, left, right = geometry.shape(length, channels)
            maximum = max(
                maximum,
                8 * channels * (length + left + right + geometry.kernel_size * positions)
                + 32 * (length * channels + positions * geometry.output_channels)
                + max(a.working_extra_channels for a in prior.activations),
            )
            outgoing = max(outgoing, positions)
            next_channels = max(next_channels, geometry.output_channels)
        length, channels = outgoing, next_channels
    return maximum


def _output_widths(request: p.FeatureConv1DRequest, limits: p.CausalLimits) -> set[int]:
    states = {(request.input_length, request.input_channels)}
    for prior in request.layers:
        if len(states) * len(prior.geometries) > limits.parameters:
            raise ValueError("convolution graph geometry catalog exceeds enumeration cap")
        states = {
            (geometry.shape(length, channels)[0], geometry.output_channels)
            for length, channels in states
            for geometry in prior.geometries
        }
    return {length * channels for length, channels in states}


def construct_convolution(fields: dict[str, Any], *, seed: int, limits: p.CausalLimits) -> p.Recipe:
    if set(fields) != {"skeleton", "nodes"}:
        raise ValueError("feature convolution requires exactly skeleton and node selections")
    raw_skeleton = fields["skeleton"]
    if type(raw_skeleton) is not dict:
        raise ValueError("convolution skeleton must be a complete graph object")

    def authored(value: Any) -> None:
        if isinstance(value, dict):
            obj = cast(dict[str, Any], value)
            if str(obj.get("version", "")).startswith("sampled-"):
                raise ValueError("convolution skeleton must not replay sampled construction")
            for child in obj.values():
                authored(child)
        elif isinstance(value, (tuple, list)):
            for child in cast(tuple[Any, ...] | list[Any], value):
                authored(child)

    authored(raw_skeleton)
    raw = {**cast(dict[str, Any], raw_skeleton), "limits": asdict(limits)}
    skeleton = p.recipe_from_dict(raw, limits=limits)
    if type(skeleton) not in (
        p.FunctionalCausalRecipe,
        p.NativeCausalRecipe,
        p.NativeVectorRecipe,
        p.ConvertedCausalRecipe,
    ):
        raise ValueError("convolution skeleton requires an authored functional or typed graph")
    skeleton = cast(
        p.FunctionalCausalRecipe
        | p.NativeCausalRecipe
        | p.NativeVectorRecipe
        | p.ConvertedCausalRecipe,
        skeleton,
    )
    selections = fields["nodes"]
    if (
        type(selections) is not tuple
        or not 1 <= len(cast(tuple[Any, ...], selections)) <= limits.nodes
    ):
        raise ValueError("convolution node selections must be nonempty and bounded")
    prepared: list[tuple[int, p.FeatureConv1DRequest | p.FrozenFeatureConv1D]] = []
    # Full graph admission precedes source construction. The multiplicative reserve
    # covers retained request/raw/snapshot metadata in strict enclosing graph replay.
    report = skeleton.admit()
    work, parameters = report["estimated_working_bytes"], report["parameters"]
    raw = asdict(skeleton)
    for entry in cast(tuple[Any, ...], selections):
        if type(entry) is not dict:
            raise ValueError("convolution selection must be an object")
        item = cast(dict[str, Any], entry)
        if set(item) not in ({"node", "request"}, {"node", "function"}):
            raise ValueError("select exactly one authored function or sampled request")
        index = item["node"]
        if type(index) is not int or not 0 <= index < len(skeleton.nodes):
            raise ValueError("convolution selection requires an existing integer node")
        if prepared and index <= prepared[-1][0]:
            raise ValueError("convolution node selections must be unique and ordered")
        node = skeleton.nodes[index]
        equation = node if isinstance(node, p.FunctionalCausalNode) else node.equation
        if "request" in item:
            if type(item["request"]) is not dict:
                raise ValueError("convolution request must be a complete object")
            request = p.feature_conv1d_request_from_dict(
                {**item["request"], "limits": asdict(limits)}, limits=limits
            )
            if (
                request.input_length * request.input_channels != equation.function.input_width
                or _output_widths(request, limits) != {equation.function.output_width}
            ):
                raise ValueError("every sampled geometry must preserve the selected graph widths")
            source = request.admit(limits=limits)
            work += 16 * source["estimated_working_bytes"] + skeleton.task.rows * _row_workspace(
                request
            )
            parameters += 16 * source["parameters"]
            prepared.append((index, request))
        else:
            if (
                type(item["function"]) is not dict
                or cast(dict[str, Any], item["function"]).get("version")
                != "frozen-feature-conv1d-r0-v1"
            ):
                raise ValueError(
                    "authored selection requires frozen kernels; use request to sample"
                )
            function = p.feature_conv1d_from_dict(item["function"], limits=limits)
            if type(function) is not p.FrozenFeatureConv1D:
                raise ValueError(
                    "authored selection requires frozen kernels; use request to sample"
                )
            if (function.input_width, function.output_width) != (
                equation.function.input_width,
                equation.function.output_width,
            ):
                raise ValueError("authored convolution must preserve selected graph widths")
            source = p.admit_feature_conv1d(function, skeleton.task.rows, limits=limits)
            work += 16 * source["estimated_working_bytes"]
            parameters += 16 * source["parameters"]
            prepared.append((index, function))
    if work > limits.working_bytes or parameters > limits.parameters:
        raise ValueError("aggregate convolution graph construction exceeds exploration limits")
    for index, source in prepared:
        if isinstance(source, p.FeatureConv1DRequest):
            streams = np.random.SeedSequence([seed, index, 0x434F4E56]).spawn(2)
            realized: p.FrozenFeatureConv1D | p.SampledFeatureConv1D = p.sample_feature_conv1d(
                source,
                named_rng=np.random.default_rng(streams[0]),
                raw_rng=np.random.default_rng(streams[1]),
            )
        else:
            realized = source
        node = raw["nodes"][index]
        equation = node if "function" in node else node["equation"]
        equation["function"] = asdict(realized)
    return p.recipe_from_dict(raw, limits=limits)
