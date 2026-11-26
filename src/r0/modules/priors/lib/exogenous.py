"""Frozen exogenous laws shared by causal roots and internal innovations."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from r0.modules.priors.lib.inputs import InputDistribution, sample_inputs


@dataclass(frozen=True, slots=True)
class ExogenousDistribution:
    inputs: InputDistribution = InputDistribution()
    version: str = "exogenous-v1"
    factor: tuple[tuple[float, ...], ...] = ()

    def __post_init__(self) -> None:
        if self.version != "exogenous-v1" or type(self.inputs) is not InputDistribution:
            raise ValueError("exogenous laws require inputs-v1 and exogenous-v1 descriptors")

        if type(self.factor) is not tuple or any(
            type(row) is not tuple
            or len(row) != len(self.factor)
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in row)
            for row in self.factor
        ):
            raise ValueError("exogenous factor must be a finite square tuple matrix")

    def validate_width(self, width: int) -> None:
        if self.factor and len(self.factor) != width:
            raise ValueError("exogenous factor must match the node width")

    def channel_sources(self, width: int) -> tuple[set[int], ...]:
        return tuple(
            {i for i, row in enumerate(self.factor) if row[j] != 0} if self.factor else {j}
            for j in range(width)
        )

    @property
    def parameter_count(self) -> int:
        return 4 + int(self.inputs.bound is not None) + len(self.factor) ** 2


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

    values = sample_inputs(spec.inputs, rows, width, stream, "exogenous")
    if spec.factor:
        with np.errstate(over="ignore", invalid="ignore"):
            values = values @ np.asarray(spec.factor, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite transformed exogenous draw; task was not retried")
    return values
