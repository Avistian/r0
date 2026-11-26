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


def causal_moment_orders(nodes: tuple[CausalNode, ...]) -> list[list[float]]:
    """Orders below each returned bound are finite; zero means not certified.

    Sums require each summand's moments. Noise-response products use independence
    of a node's exogenous draw from its ancestors. This is a sufficient certificate,
    not an exact symbolic integrability solver and not sample-estimated moments.
    """
    moments: list[list[float]] = []
    for node in nodes:
        incoming = [order for p in node.parents for order in moments[p]]
        if isinstance(node, ActivatedCausalNode):
            incoming = _activation_orders(incoming, node.input_activation)
        affine = [
            min(
                (order for i, order in enumerate(incoming) if node.weights[i][j] != 0),
                default=math.inf,
            )
            for j in range(len(node.bias))
        ]
        noise = [math.inf] * len(node.bias)
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
                zero_affine = node.bias[j] == 0 and all(row[j] == 0 for row in node.weights)
                zero_response = response.offset == 0 and (response.strength == 0 or zero_affine)
                if node.noise_scale[j] and sources and not zero_response:
                    noise[j] = min(base, affine[j] / response.power) if response.strength else base
        if isinstance(node, ActivatedCausalNode) and node.noise_position == "after_output":
            output = _activation_orders(affine, node.output_activation)
            moments.append([min(a, n) for a, n in zip(output, noise, strict=True)])
        else:
            output = [min(a, n) for a, n in zip(affine, noise, strict=True)]
            moments.append(
                _activation_orders(output, node.output_activation)
                if isinstance(node, ActivatedCausalNode)
                else output
            )
    return moments
