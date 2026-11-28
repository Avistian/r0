"""Frozen functions with fitted centers or ordered child mechanisms."""

from dataclasses import dataclass

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.calibrated_activations import CalibratedActivation
from r0.modules.priors.lib.frozen_function_primitives import (
    FrozenLinearFunction,
    FrozenQuadraticFunction,
    FrozenRFFFunction,
    FrozenTreeFunction,
)
from r0.modules.priors.lib.weight_geometry import Matrix, matrix_shape


@dataclass(frozen=True, slots=True)
class FrozenDiscretizationFunction:
    centers: Matrix
    linear: FrozenLinearFunction
    p: float = 2.0
    version: str = "frozen-discretization-function-v1"

    def __post_init__(self) -> None:
        _, width = matrix_shape(self.centers)
        if (
            self.version != "frozen-discretization-function-v1"
            or type(self.linear) is not FrozenLinearFunction
            or self.linear.input_width != width
        ):
            raise ValueError("discretization requires aligned centers and a frozen linear map")
        if not finite_number(self.p) or not 0.5 <= self.p <= 4:
            raise ValueError("discretization distance power must lie in [0.5,4]")

    @property
    def input_width(self) -> int:
        return self.linear.input_width

    @property
    def output_width(self) -> int:
        return self.linear.output_width

    @property
    def parameter_count(self) -> int:
        return len(self.centers) * self.input_width + self.linear.parameter_count + 1


@dataclass(frozen=True, slots=True)
class FrozenEMFunction:
    centers: Matrix
    scales: tuple[float, ...]
    linear: FrozenLinearFunction
    p: float = 2.0
    q: float = 1.0
    version: str = "frozen-em-function-v1"

    def __post_init__(self) -> None:
        count, _ = matrix_shape(self.centers)
        if (
            self.version != "frozen-em-function-v1"
            or type(self.linear) is not FrozenLinearFunction
            or self.linear.input_width != count
            or not 2 <= count <= max(16, 2 * self.linear.output_width)
        ):
            raise ValueError(
                "EM components must match the frozen output map and source count bounds"
            )
        if (
            type(self.scales) is not tuple
            or len(self.scales) != count
            or any(not finite_number(s) or s <= 0 for s in self.scales)
        ):
            raise ValueError("EM spreads must be finite positive values for every center")
        if (
            not finite_number(self.p)
            or not 1 <= self.p <= 4
            or not finite_number(self.q)
            or not 1 <= self.q <= 2
        ):
            raise ValueError("EM distance and exponent must lie in [1,4] and [1,2]")

    @property
    def input_width(self) -> int:
        return len(self.centers[0])

    @property
    def output_width(self) -> int:
        return self.linear.output_width

    @property
    def parameter_count(self) -> int:
        return len(self.centers) * (self.input_width + 1) + self.linear.parameter_count + 2


type CheapFrozenFunction = (
    FrozenLinearFunction
    | FrozenQuadraticFunction
    | FrozenTreeFunction
    | FrozenDiscretizationFunction
    | FrozenRFFFunction
)


@dataclass(frozen=True, slots=True)
class FrozenProductFunction:
    left: CheapFrozenFunction
    right: CheapFrozenFunction
    version: str = "frozen-product-function-v1"

    def __post_init__(self) -> None:
        allowed = (
            FrozenLinearFunction,
            FrozenQuadraticFunction,
            FrozenTreeFunction,
            FrozenDiscretizationFunction,
            FrozenRFFFunction,
        )
        if (
            self.version != "frozen-product-function-v1"
            or type(self.left) not in allowed
            or type(self.right) not in allowed
        ):
            raise ValueError("product requires exactly two nonrecursive cheap function children")
        if (self.left.input_width, self.left.output_width) != (
            self.right.input_width,
            self.right.output_width,
        ):
            raise ValueError("product children must have matching input and output geometry")

    @property
    def input_width(self) -> int:
        return self.left.input_width

    @property
    def output_width(self) -> int:
        return self.left.output_width

    @property
    def parameter_count(self) -> int:
        return self.left.parameter_count + self.right.parameter_count


@dataclass(frozen=True, slots=True)
class FrozenMLPFunction:
    linears: tuple[FrozenLinearFunction, ...]
    activations: tuple[CalibratedActivation, ...] = ()
    start_activation: bool = False
    end_activation: bool = False
    version: str = "frozen-mlp-function-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "frozen-mlp-function-v1"
            or type(self.linears) is not tuple
            or not 1 <= len(self.linears) <= 3
            or any(type(layer) is not FrozenLinearFunction for layer in self.linears)
            or type(self.start_activation) is not bool
            or type(self.end_activation) is not bool
        ):
            raise ValueError(
                "MLP requires one to three frozen linears and Boolean activation flags"
            )
        if any(
            left.output_width != right.input_width
            for left, right in zip(self.linears[:-1], self.linears[1:], strict=True)
        ):
            raise ValueError("MLP adjacent linear dimensions must match")
        hidden = self.linears[0].output_width
        if len(self.linears) > 1 and (
            not 1 <= hidden <= 127
            or any(layer.output_width != hidden for layer in self.linears[:-1])
        ):
            raise ValueError("MLP hidden widths must share the source width in [1,127]")
        widths = [self.input_width] if self.start_activation else []
        widths += [layer.output_width for layer in self.linears[:-1]]
        if self.end_activation:
            widths.append(self.output_width)
        if (
            type(self.activations) is not tuple
            or len(self.activations) != len(widths)
            or any(type(a) is not CalibratedActivation for a in self.activations)
        ):
            raise ValueError("MLP must retain every fitted standard activation in order")
        for activation, width in zip(self.activations, widths, strict=True):
            activation.validate_width(width)

    @property
    def input_width(self) -> int:
        return self.linears[0].input_width

    @property
    def output_width(self) -> int:
        return self.linears[-1].output_width

    @property
    def parameter_count(self) -> int:
        return (
            2
            + sum(layer.parameter_count for layer in self.linears)
            + sum(a.parameter_count for a in self.activations)
        )
