"""Frozen exogenous laws shared by causal roots and internal innovations."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from r0.modules.priors.lib.inputs import InputDistribution, sample_inputs


@dataclass(frozen=True, slots=True)
class ExogenousDistribution:
    inputs: InputDistribution = InputDistribution()
    version: str = "exogenous-v1"

    def __post_init__(self) -> None:
        if self.version != "exogenous-v1" or type(self.inputs) is not InputDistribution:
            raise ValueError("exogenous laws require inputs-v1 and exogenous-v1 descriptors")

    @property
    def parameter_count(self) -> int:
        return 4 + int(self.inputs.bound is not None)


def sample_exogenous(
    spec: ExogenousDistribution,
    rows: int,
    width: int,
    rng: Callable[..., np.random.Generator],
    node: int,
) -> NDArray[np.float64]:
    def stream(channel: str, column: int, purpose: str) -> np.random.Generator:
        return (
            rng("innovation", node, column)
            if purpose == "values"
            else rng("innovation", node, column, purpose)
        )

    return sample_inputs(spec.inputs, rows, width, stream, "exogenous")
