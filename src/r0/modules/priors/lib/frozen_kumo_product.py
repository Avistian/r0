"""Kumo's two same-input, non-product children, separate from the I2 cheap product."""

from dataclasses import dataclass

from r0.modules.priors.lib.frozen_function_composites import (
    FrozenDiscretizationFunction,
    FrozenEMFunction,
    FrozenMLPFunction,
)
from r0.modules.priors.lib.frozen_function_primitives import (
    FrozenLinearFunction,
    FrozenQuadraticFunction,
    FrozenRFFFunction,
    FrozenTreeFunction,
)

type KumoProductChild = (
    FrozenLinearFunction
    | FrozenQuadraticFunction
    | FrozenRFFFunction
    | FrozenTreeFunction
    | FrozenDiscretizationFunction
    | FrozenEMFunction
    | FrozenMLPFunction
)

KUMO_CHILD_TYPES = (
    FrozenLinearFunction,
    FrozenQuadraticFunction,
    FrozenRFFFunction,
    FrozenTreeFunction,
    FrozenDiscretizationFunction,
    FrozenEMFunction,
    FrozenMLPFunction,
)


@dataclass(frozen=True, slots=True)
class FrozenKumoProductFunction:
    left: KumoProductChild
    right: KumoProductChild
    version: str = "frozen-kumo-product-function-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "frozen-kumo-product-function-v1"
            or type(self.left) not in KUMO_CHILD_TYPES
            or type(self.right) not in KUMO_CHILD_TYPES
        ):
            raise ValueError("Kumo product requires exactly two nonrecursive non-product children")
        if (self.left.input_width, self.left.output_width) != (
            self.right.input_width,
            self.right.output_width,
        ):
            raise ValueError("Kumo product children must share input and output geometry")

    @property
    def input_width(self) -> int:
        return self.left.input_width

    @property
    def output_width(self) -> int:
        return self.left.output_width

    @property
    def parameter_count(self) -> int:
        return self.left.parameter_count + self.right.parameter_count
