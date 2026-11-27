"""Conservative positive absolute-moment certificates for recorded stochastic DAGs."""

import math

from r0.modules.priors.lib.calibrated_activations import CalibratedActivation
from r0.modules.priors.lib.causal_activations import CausalActivation
from r0.modules.priors.lib.causal_nodes import (
    ActivatedCausalNode,
    CausalNode,
    RootInputCausalNode,
    StochasticCausalNode,
)
from r0.modules.priors.lib.node_pipeline import CausalNodePipeline

_BOUNDED = {
    "kumaraswamy",
    "tanh",
    "sigmoid",
    "relu6",
    "hardtanh",
    "sign",
    "sign_positive",
    "gaussian_bump",
    "interval_abs_le_1",
    "interval_0_1",
    "sin",
    "step",
    "heaviside",
    "heaviside_positive",
    "modulo",
    "softmax",
    "tied_max",
    "argmax",
    "rank",
    "argsort",
    "reciprocal_abs_power",
}


def _activation_orders(orders: list[float], activation: CausalActivation) -> list[float]:
    if not activation.uses_input(len(orders)) or activation.kind in _BOUNDED:
        return [math.inf] * len(orders)
    kind, parameters = activation.kind, dict(activation.parameters)
    if kind in ("exp", "log"):
        # Polynomial moments alone do not certify exponential growth or log near zero.
        return [0.0] * len(orders)
    if kind in ("log_abs", "log_clip_abs"):
        return [math.inf if order > 0 else 0.0 for order in orders]
    growth = {"square": 2.0, "sqrt_abs": 0.5}.get(kind, 1.0)
    if kind in ("power", "integer_power", "relu_power", "signed_power"):
        growth = parameters.get("exponent", 2.0)
        if growth < 0:
            return [0.0] * len(orders)
    return [order / growth for order in orders]


def _transform(
    orders: list[float], constants: list[bool], activation: CausalActivation
) -> tuple[list[float], list[bool]]:
    output = _activation_orders(orders, activation)
    if not activation.uses_input(len(orders)):
        constants = [True] * len(orders)
    elif activation.mixes_channels:
        constants = [all(constants)] * len(orders)
    return [
        math.inf if fixed else order for fixed, order in zip(constants, output, strict=True)
    ], constants


def _pipeline_transform(
    orders: list[float], constants: list[bool], activation: CausalActivation
) -> tuple[list[float], list[bool]]:
    needs_second = isinstance(activation, CalibratedActivation) or activation.kind == "l2_normalize"
    if needs_second and any(order <= 2 for order in orders):
        raise ValueError("pipeline normalization requires certified finite input second moments")
    output, fixed = _transform(orders, constants, activation)
    if isinstance(activation, CalibratedActivation) and any(order <= 2 for order in output):
        raise ValueError("pipeline standardization requires certified finite output second moments")
    return output, fixed


def causal_moment_orders(
    nodes: tuple[CausalNode, ...],
    *,
    pipelines: tuple[CausalNodePipeline, ...] | None = None,
    readout_orders: list[list[float]] | None = None,
) -> list[list[float]]:
    """Orders below each returned bound are finite; zero means not certified.

    Sums require each summand's moments. Noise-response products use independence
    of a node's exogenous draw from its ancestors. This is a sufficient certificate,
    not an exact symbolic integrability solver and not sample-estimated moments.
    """
    transform = _pipeline_transform if pipelines is not None else _transform
    moments: list[list[float]] = []
    constants: list[list[bool]] = []
    for index, node in enumerate(nodes):
        captured: dict[int, float] = {}
        incoming = [order for p in node.parents for order in moments[p]]
        fixed = [known for p in node.parents for known in constants[p]]
        if isinstance(node, ActivatedCausalNode):
            incoming, fixed = transform(incoming, fixed, node.input_activation)
        affine_fixed = [
            all(known or node.weights[i][j] == 0 for i, known in enumerate(fixed))
            for j in range(len(node.bias))
        ]
        affine = [
            math.inf
            if affine_fixed[j]
            else min(
                (order for i, order in enumerate(incoming) if node.weights[i][j] != 0),
                default=math.inf,
            )
            for j in range(len(node.bias))
        ]
        if isinstance(node, RootInputCausalNode):
            affine_fixed = [not sources for sources in node.root_sources()]
            affine = [
                math.inf if fixed else node.root_exogenous.moment_order for fixed in affine_fixed
            ]
        noise = [math.inf] * len(node.bias)
        noise_fixed = [scale == 0 for scale in node.noise_scale]
        if isinstance(node, StochasticCausalNode):
            spec, response = node.exogenous, node.noise_response
            base = spec.moment_order
            for j, sources in enumerate(spec.channel_sources(len(node.bias))):
                noise_fixed[j] = not node.has_noise_scale(j) or not sources
                if not noise_fixed[j]:
                    noise[j] = min(base, affine[j] / response.power) if response.strength else base
        if isinstance(node, ActivatedCausalNode) and node.noise_position == "after_output":
            output, output_fixed = transform(affine, affine_fixed, node.output_activation)
            if pipelines is not None:
                pipeline = pipelines[index]
                for stage in pipeline.stages:
                    output, output_fixed = transform(output, output_fixed, stage)
                if pipeline.channel_importance:
                    output_fixed = [
                        fixed or factor == 0
                        for fixed, factor in zip(
                            output_fixed, pipeline.channel_importance, strict=True
                        )
                    ]
                    output = [
                        math.inf if fixed else order
                        for fixed, order in zip(output_fixed, output, strict=True)
                    ]
                if pipeline.l2 is not None:
                    output, output_fixed = transform(output, output_fixed, pipeline.l2)
                for converter in pipelines[index].converters:
                    channel = converter.channel
                    warped, fixed_warped = _pipeline_transform(
                        [output[channel]], [output_fixed[channel]], converter.transform
                    )
                    captured[channel] = output[channel] if converter.captures_raw else warped[0]
                    if converter.changes_state:
                        output[channel], output_fixed[channel] = warped[0], fixed_warped[0]
            moments.append([min(a, n) for a, n in zip(output, noise, strict=True)])
            constants.append([a and n for a, n in zip(output_fixed, noise_fixed, strict=True)])
        else:
            output = [min(a, n) for a, n in zip(affine, noise, strict=True)]
            output_fixed = [a and n for a, n in zip(affine_fixed, noise_fixed, strict=True)]
            if isinstance(node, ActivatedCausalNode):
                output, output_fixed = transform(output, output_fixed, node.output_activation)
            moments.append(output)
            constants.append(output_fixed)
        if readout_orders is not None:
            readout = list(moments[-1])
            for channel, order in captured.items():
                readout[channel] = order
            readout_orders.append(readout)
    return moments
