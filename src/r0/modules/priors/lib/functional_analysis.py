"""Conservative channel ancestry and population-moment certificates for frozen DAGs."""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from r0.modules.priors.lib.calibrated_activations import (
    CalibratedActivation,
    CalibratedStandardization,
)
from r0.modules.priors.lib.causal_activations import CausalActivation
from r0.modules.priors.lib.causal_moments import BOUNDED_ACTIVATIONS, activation_moment_orders
from r0.modules.priors.lib.frozen_function_composites import (
    FrozenDiscretizationFunction,
    FrozenEMFunction,
    FrozenMLPFunction,
    FrozenProductFunction,
)
from r0.modules.priors.lib.frozen_function_evaluation import FrozenFunction
from r0.modules.priors.lib.frozen_function_primitives import (
    FrozenLinearFunction,
    FrozenQuadraticFunction,
    FrozenRFFFunction,
    FrozenTreeFunction,
)

if TYPE_CHECKING:
    from r0.modules.priors.lib.functional_causal import FunctionalCausalNode


@dataclass(frozen=True, slots=True)
class ChannelCertificate:
    sources: frozenset[tuple[int, int]] = frozenset()
    order: float = math.inf
    bounded: bool = True


def _sum(channels: list[ChannelCertificate]) -> ChannelCertificate:
    return ChannelCertificate(
        frozenset[tuple[int, int]]().union(*(c.sources for c in channels)),
        min((c.order for c in channels), default=math.inf),
        all(c.bounded for c in channels),
    )


def _product(left: ChannelCertificate, right: ChannelCertificate) -> ChannelCertificate:
    # Children share their inputs. Holder's inequality needs no independence claim.
    order = (
        right.order
        if left.bounded
        else left.order
        if right.bounded
        else 0.0
        if min(left.order, right.order) == 0
        else math.inf
        if left.order == right.order == math.inf
        else 1 / (1 / left.order + 1 / right.order)
    )
    return ChannelCertificate(left.sources | right.sources, order, left.bounded and right.bounded)


def _activation(
    channels: list[ChannelCertificate], activation: CausalActivation
) -> list[ChannelCertificate]:
    normalized = isinstance(activation, CalibratedActivation | CalibratedStandardization)
    if (normalized or activation.kind == "l2_normalize") and any(c.order <= 2 for c in channels):
        raise ValueError("functional normalization requires certified finite input second moments")
    result: list[ChannelCertificate]
    if not activation.uses_input(len(channels)):
        result = [ChannelCertificate() for _ in channels]
    else:
        inputs = [_sum(channels)] * len(channels) if activation.mixes_channels else channels
        orders = activation_moment_orders([c.order for c in inputs], activation)
        exponent = dict(activation.parameters).get("exponent", 2)
        singular = activation.kind == "log" or (
            activation.kind in ("power", "integer_power") and exponent < 0
        )
        result = []
        for channel, order in zip(inputs, orders, strict=True):
            bounded = (
                not channel.sources
                or activation.kind in BOUNDED_ACTIVATIONS
                or channel.bounded
                and not singular
            )
            result.append(
                ChannelCertificate(channel.sources, math.inf if bounded else order, bounded)
            )
    if isinstance(activation, CalibratedActivation) and any(c.order <= 2 for c in result):
        raise ValueError("functional normalization requires certified finite output second moments")
    return result


def function_certificates(
    function: FrozenFunction, inputs: list[ChannelCertificate]
) -> list[ChannelCertificate]:
    if type(function) is FrozenLinearFunction:
        return [
            _sum([c for c, weight in zip(inputs, row, strict=True) if weight != 0])
            for row in function.matrix
        ]
    if type(function) is FrozenQuadraticFunction:
        augmented = [inputs[i] for i in function.indices] + [ChannelCertificate()]
        return [
            _sum(
                [
                    _product(augmented[i], augmented[j])
                    for i, row in enumerate(matrix)
                    for j, weight in enumerate(row)
                    if weight != 0
                ]
            )
            for matrix in function.matrices
        ]
    if type(function) is FrozenProductFunction:
        return [
            _product(a, b)
            for a, b in zip(
                function_certificates(function.left, inputs),
                function_certificates(function.right, inputs),
                strict=True,
            )
        ]
    if type(function) is FrozenMLPFunction:
        values = inputs
        offset = 0
        if function.start_activation:
            values = _activation(values, function.activations[0])
            offset = 1
        for i, layer in enumerate(function.linears):
            values = function_certificates(layer, values)
            if i < len(function.linears) - 1 or function.end_activation:
                values = _activation(values, function.activations[offset])
                offset += 1
        return values
    if type(function) is FrozenTreeFunction:
        sources = _sum([inputs[i] for row in function.features for i in row]).sources
        return [
            ChannelCertificate(
                frozenset()
                if all(len({row[j] for row in table}) == 1 for table in function.leaves)
                else sources
            )
            for j in range(function.output_width)
        ]
    if type(function) is FrozenRFFFunction:
        return [
            ChannelCertificate(
                _sum(
                    [
                        inputs[i]
                        for k, row in enumerate(function.coefficients)
                        if row[j] != 0
                        for i, frequencies in enumerate(function.frequencies)
                        if frequencies[k] != 0
                    ]
                ).sources
            )
            for j in range(function.output_width)
        ]
    assert isinstance(function, FrozenDiscretizationFunction | FrozenEMFunction)
    sources = _sum(inputs).sources
    return [
        ChannelCertificate(sources if any(weight != 0 for weight in row) else frozenset())
        for row in function.linear.matrix
    ]


def functional_certificates(
    nodes: tuple["FunctionalCausalNode", ...],
) -> tuple[list[list[ChannelCertificate]], list[list[ChannelCertificate]]]:
    """Never infer population moments or delete dependencies from a finite sample."""
    states: list[list[ChannelCertificate]] = []
    readouts: list[list[ChannelCertificate]] = []
    for index, node in enumerate(nodes):
        state, readout = functional_node_certificates(
            node, index, [c for p in node.parents for c in states[p]]
        )
        states.append(state)
        readouts.append(readout)
    return states, readouts


def functional_node_certificates(
    node: "FunctionalCausalNode", index: int, incoming: list[ChannelCertificate]
) -> tuple[list[ChannelCertificate], list[ChannelCertificate]]:
    """Analyze one equation with caller-supplied parent state certificates."""
    width = node.function.output_width
    if node.root is not None:
        base = node.root
        bounded = base.family in ("uniform", "ball") or base.covariance_base == "uniform"
        incoming = []
        for j in range(base.width):
            sources = (
                {k for k, weight in enumerate(base.factor[j]) if weight != 0}
                if base.factor is not None
                else set(range(base.width))
                if base.family == "ball"
                else {j}
            )
            incoming.append(
                ChannelCertificate(
                    frozenset((index, width + k) for k in sources),
                    math.inf,
                    bounded or not sources,
                )
            )
    output = function_certificates(node.function, incoming)
    before = output
    captured: dict[int, ChannelCertificate] = {}
    if node.pipeline:
        pipeline = node.pipeline
        for stage in pipeline.stages:
            before = _activation(before, stage)
        if pipeline.channel_importance:
            before = [
                c if factor != 0 else ChannelCertificate()
                for c, factor in zip(before, pipeline.channel_importance, strict=True)
            ]
        if pipeline.l2:
            before = _activation(before, pipeline.l2)
        before = list(before)
        for converter in pipeline.converters:
            channel = converter.channel
            warped = _activation([before[channel]], converter.transform)[0]
            captured[channel] = before[channel] if converter.captures_raw else warped
            if converter.changes_state:
                before[channel] = warped
    noise: list[ChannelCertificate] = []
    response = node.noise_response
    for j, sources in enumerate(node.exogenous.channel_sources(width)):
        if (
            not node.noise_scale
            or node.noise_scale[j] == 0
            or not sources
            or (response.offset == response.strength == 0)
        ):
            noise.append(ChannelCertificate())
            continue
        law = node.exogenous.inputs
        bounded = (
            law.bound is not None
            or law.distribution == "uniform"
            or node.exogenous.geometry == "ball"
        )
        dependencies = frozenset((index, k) for k in sources)
        order = node.exogenous.moment_order
        if response.strength:
            dependencies |= output[j].sources
            order = min(order, output[j].order / response.power)
            bounded = bounded and output[j].bounded
        noise.append(ChannelCertificate(dependencies, order, bounded))
    state = [_sum([value, innovation]) for value, innovation in zip(before, noise, strict=True)]
    return state, [captured.get(j, c) for j, c in enumerate(state)]
