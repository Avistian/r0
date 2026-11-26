"""Conservative positive absolute-moment certificates for recorded stochastic DAGs."""

import math

from r0.modules.priors.lib.causal_activations import CausalActivation
from r0.modules.priors.lib.causal_nodes import ActivatedCausalNode, CausalNode, StochasticCausalNode

_BOUNDED = {
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


def causal_moment_orders(nodes: tuple[CausalNode, ...]) -> list[list[float]]:
    """Orders below each returned bound are finite; zero means not certified.

    Sums require each summand's moments. Noise-response products use independence
    of a node's exogenous draw from its ancestors. This is a sufficient certificate,
    not an exact symbolic integrability solver and not sample-estimated moments.
    """
    moments: list[list[float]] = []
    constants: list[list[bool]] = []
    for node in nodes:
        incoming = [order for p in node.parents for order in moments[p]]
        fixed = [known for p in node.parents for known in constants[p]]
        if isinstance(node, ActivatedCausalNode):
            incoming, fixed = _transform(incoming, fixed, node.input_activation)
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
        noise = [math.inf] * len(node.bias)
        noise_fixed = [scale == 0 for scale in node.noise_scale]
        if isinstance(node, StochasticCausalNode):
            spec, response = node.exogenous, node.noise_response
            base = (
                spec.inputs.df
                if spec.inputs.distribution == "student_t"
                and spec.inputs.bound is None
                and spec.geometry == "independent"
                else math.inf
            )
            for j, sources in enumerate(spec.channel_sources(len(node.bias))):
                noise_fixed[j] = not node.has_noise_scale(j) or not sources
                if not noise_fixed[j]:
                    noise[j] = min(base, affine[j] / response.power) if response.strength else base
        if isinstance(node, ActivatedCausalNode) and node.noise_position == "after_output":
            output, output_fixed = _transform(affine, affine_fixed, node.output_activation)
            moments.append([min(a, n) for a, n in zip(output, noise, strict=True)])
            constants.append([a and n for a, n in zip(output_fixed, noise_fixed, strict=True)])
        else:
            output = [min(a, n) for a, n in zip(affine, noise, strict=True)]
            output_fixed = [a and n for a, n in zip(affine_fixed, noise_fixed, strict=True)]
            if isinstance(node, ActivatedCausalNode):
                output, output_fixed = _transform(output, output_fixed, node.output_activation)
            moments.append(output)
            constants.append(output_fixed)
    return moments
