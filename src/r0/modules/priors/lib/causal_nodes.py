"""Frozen causal node equations and their shared saved-record decoder."""

import math
from dataclasses import dataclass
from typing import Any

from r0.modules.priors.lib.calibrated_activations import (
    CalibratedActivation,
    CalibratedStandardization,
    activation_from_dict,
)
from r0.modules.priors.lib.calibrated_transforms import CalibratedTransform
from r0.modules.priors.lib.causal_activations import CausalActivation
from r0.modules.priors.lib.exogenous import ExogenousDistribution, NoiseResponse
from r0.modules.priors.lib.inputs import InputDistribution
from r0.modules.priors.lib.weight_constructions import (
    WEIGHT_CONSTRUCTION_TYPES,
    WeightConstruction,
    weight_construction_from_dict,
)
from r0.modules.priors.lib.weight_matrices import FactorizedWeights


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
        return self.input_activation.parameter_count + self.output_activation.parameter_count

    @property
    def working_extra_channels(self) -> int:
        return max(
            (a.working_extra_channels for a in (self.input_activation, self.output_activation)),
            default=0,
        )

    def __post_init__(self) -> None:
        CausalNode.__post_init__(self)
        if any(
            type(a)
            not in (
                CausalActivation,
                CalibratedActivation,
                CalibratedTransform,
                CalibratedStandardization,
            )
            for a in (
                self.input_activation,
                self.output_activation,
            )
        ):
            raise ValueError("node activations must use CausalActivation")
        self.input_activation.validate_width(len(self.weights))
        self.output_activation.validate_width(len(self.bias))
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
        return max(
            self.working_channels_for_width(len(self.bias)),
            super(StochasticCausalNode, self).working_extra_channels,
        )

    @staticmethod
    def working_channels_for_width(width: int) -> int:
        return 8 * width

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


@dataclass(frozen=True, slots=True)
class RootInputCausalNode(StochasticCausalNode):
    """Independent root input before activation and separate structural innovation."""

    root_exogenous: ExogenousDistribution = ExogenousDistribution()
    root_scale: tuple[float, ...] = (1.0,)
    root_version: str = "root-input-v1"
    noise_position: str = "after_output"

    @property
    def structural_parameter_count(self) -> int:
        return (
            super(RootInputCausalNode, self).structural_parameter_count
            + self.root_exogenous.parameter_count
            + len(self.root_scale)
        )

    @property
    def working_extra_channels(self) -> int:
        return super(RootInputCausalNode, self).working_extra_channels + 8 * len(self.bias)

    @property
    def working_row_floor(self) -> int:
        return max(
            super(RootInputCausalNode, self).working_row_floor,
            256 if self.root_exogenous.inputs.bound is not None else 0,
        )

    def root_sources(self) -> tuple[set[int], ...]:
        return tuple(
            sources if scale != 0 else set[int]()
            for sources, scale in zip(
                self.root_exogenous.channel_sources(len(self.bias)), self.root_scale, strict=True
            )
        )

    def has_noise_scale(self, channel: int) -> bool:
        response = self.noise_response
        root_varies = self.root_scale[channel] != 0 and (
            not self.root_exogenous.factor
            or any(row[channel] != 0 for row in self.root_exogenous.factor)
        )
        zero_affine = self.bias[channel] == 0 and not root_varies
        zero_response = response.offset == 0 and (response.strength == 0 or zero_affine)
        return self.noise_scale[channel] != 0 and not zero_response

    def __post_init__(self) -> None:
        StochasticCausalNode.__post_init__(self)
        if self.parents or self.weights or self.root_version != "root-input-v1":
            raise ValueError("root inputs require a parentless root-input-v1 node")
        if type(self.root_exogenous) is not ExogenousDistribution:
            raise ValueError("root input requires a frozen exogenous law")
        self.root_exogenous.validate_width(len(self.bias))
        if (
            type(self.root_scale) is not tuple
            or len(self.root_scale) != len(self.bias)
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in self.root_scale)
        ):
            raise ValueError("root scales must be finite and match the node width")


@dataclass(frozen=True, slots=True)
class MatrixCausalNode(StochasticCausalNode):
    weight_construction: WeightConstruction = FactorizedWeights()

    @property
    def structural_parameter_count(self) -> int:
        return (
            super(MatrixCausalNode, self).structural_parameter_count
            + self.weight_construction.parameter_count
        )

    def __post_init__(self) -> None:
        StochasticCausalNode.__post_init__(self)
        if type(self.weight_construction) not in WEIGHT_CONSTRUCTION_TYPES:
            raise ValueError("matrix nodes require a frozen weight construction")
        if not self.parents or self.weight_construction.shape != (
            len(self.weights),
            len(self.bias),
        ):
            raise ValueError("weight construction shape must match the node projection")


def causal_node_from_dict(data: dict[str, Any]) -> CausalNode:
    """One decoder shared by recipe loading and original-row inspection."""
    fields = dict(data)
    for key in ("parents", "bias", "noise_scale", "root_scale"):
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
            fields[key] = activation_from_dict(fields[key])
    if "weight_construction" in fields:
        fields["weight_construction"] = weight_construction_from_dict(fields["weight_construction"])
    if any(
        key in fields
        for key in (
            "exogenous",
            "noise_response",
            "weight_construction",
            "root_exogenous",
            "root_scale",
            "root_version",
        )
    ):
        spec = dict(fields.get("exogenous", {}))
        spec["inputs"] = InputDistribution(**spec.get("inputs", {}))
        spec["factor"] = tuple(tuple(row) for row in spec.get("factor", ()))
        fields["exogenous"] = ExogenousDistribution(**spec)
        fields["noise_response"] = NoiseResponse(**fields.get("noise_response", {}))
        if any(key in fields for key in ("root_exogenous", "root_scale", "root_version")):
            root = dict(fields.get("root_exogenous", {}))
            root["inputs"] = InputDistribution(**root.get("inputs", {}))
            root["factor"] = tuple(tuple(row) for row in root.get("factor", ()))
            fields["root_exogenous"] = ExogenousDistribution(**root)
            return RootInputCausalNode(**fields)
        return (
            MatrixCausalNode(**fields)
            if "weight_construction" in fields
            else StochasticCausalNode(**fields)
        )
    return ActivatedCausalNode(**fields) if activated else CausalNode(**fields)
