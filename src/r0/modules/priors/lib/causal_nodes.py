"""Frozen causal node equations and their shared saved-record decoder."""

import math
from dataclasses import dataclass
from typing import Any

from r0.modules.priors.lib.causal_activations import CausalActivation
from r0.modules.priors.lib.exogenous import ExogenousDistribution, NoiseResponse
from r0.modules.priors.lib.inputs import InputDistribution


@dataclass(frozen=True, slots=True)
class CausalNode:
    parents: tuple[int, ...] = ()
    weights: tuple[tuple[float, ...], ...] = ()
    bias: tuple[float, ...] = (0.0,)
    noise_scale: tuple[float, ...] = (1.0,)

    @property
    def activation_parameter_count(self) -> int:
        return 0

    @property
    def structural_parameter_count(self) -> int:
        return self.activation_parameter_count

    @property
    def working_extra_channels(self) -> int:
        return 0

    @property
    def working_row_floor(self) -> int:
        return 0

    def __post_init__(self) -> None:
        if (
            type(self.parents) is not tuple
            or any(type(p) is not int or p < 0 for p in self.parents)
            or len(set(self.parents)) != len(self.parents)
        ):
            raise ValueError("causal parents must be unique nonnegative node indices")
        if (
            type(self.bias) is not tuple
            or not self.bias
            or type(self.noise_scale) is not tuple
            or len(self.noise_scale) != len(self.bias)
            or type(self.weights) is not tuple
            or any(type(row) is not tuple or len(row) != len(self.bias) for row in self.weights)
        ):
            raise ValueError("causal bias, noise and weight output widths must agree")
        values = (*self.bias, *self.noise_scale, *(v for row in self.weights for v in row))
        if any(type(v) not in (float, int) or not math.isfinite(v) for v in values):
            raise ValueError("causal parameters must be finite numbers")
        if any(v < 0 for v in self.noise_scale):
            raise ValueError("structural noise scales must be nonnegative")


@dataclass(frozen=True, slots=True)
class ActivatedCausalNode(CausalNode):
    input_activation: CausalActivation = CausalActivation()
    output_activation: CausalActivation = CausalActivation()
    noise_position: str = "before_output"

    @property
    def activation_parameter_count(self) -> int:
        return len(self.input_activation.parameters) + len(self.output_activation.parameters)

    def __post_init__(self) -> None:
        CausalNode.__post_init__(self)
        if any(
            type(a) is not CausalActivation
            for a in (
                self.input_activation,
                self.output_activation,
            )
        ):
            raise ValueError("node activations must use CausalActivation")
        if self.noise_position not in ("before_output", "after_output"):
            raise ValueError("noise position must be before_output or after_output")
        if not self.parents and self.input_activation != CausalActivation():
            raise ValueError("root nodes have no parent channels to activate")


@dataclass(frozen=True, slots=True)
class StochasticCausalNode(ActivatedCausalNode):
    exogenous: ExogenousDistribution = ExogenousDistribution()
    noise_response: NoiseResponse = NoiseResponse()

    @property
    def structural_parameter_count(self) -> int:
        return self.activation_parameter_count + self.exogenous.parameter_count + 3

    @property
    def working_extra_channels(self) -> int:
        return 8 * len(self.bias)

    @property
    def working_row_floor(self) -> int:
        return 256 if self.exogenous.inputs.bound is not None else 0

    def has_noise_scale(self, channel: int) -> bool:
        response = self.noise_response
        zero_affine = self.bias[channel] == 0 and all(row[channel] == 0 for row in self.weights)
        zero_response = response.offset == 0 and (response.strength == 0 or zero_affine)
        return self.noise_scale[channel] != 0 and not zero_response

    def __post_init__(self) -> None:
        ActivatedCausalNode.__post_init__(self)
        if type(self.exogenous) is not ExogenousDistribution:
            raise ValueError("stochastic nodes require ExogenousDistribution")
        self.exogenous.validate_width(len(self.bias))
        if type(self.noise_response) is not NoiseResponse:
            raise ValueError("stochastic nodes require NoiseResponse")


def causal_node_from_dict(data: dict[str, Any]) -> CausalNode:
    """One decoder shared by recipe loading and original-row inspection."""
    fields = dict(data)
    for key in ("parents", "bias", "noise_scale"):
        if key in fields:
            fields[key] = tuple(fields[key])
    if "weights" in fields:
        fields["weights"] = tuple(tuple(row) for row in fields["weights"])
    activated = any(
        key in fields
        for key in (
            "input_activation",
            "output_activation",
            "noise_position",
        )
    )
    for key in ("input_activation", "output_activation"):
        if key in fields:
            activation = dict(fields[key])
            activation["parameters"] = tuple(
                tuple(pair) for pair in activation.get("parameters", ())
            )
            fields[key] = CausalActivation(**activation)
    if "exogenous" in fields or "noise_response" in fields:
        spec = dict(fields.get("exogenous", {}))
        spec["inputs"] = InputDistribution(**spec.get("inputs", {}))
        spec["factor"] = tuple(tuple(row) for row in spec.get("factor", ()))
        fields["exogenous"] = ExogenousDistribution(**spec)
        fields["noise_response"] = NoiseResponse(**fields.get("noise_response", {}))
        return StochasticCausalNode(**fields)
    return ActivatedCausalNode(**fields) if activated else CausalNode(**fields)
