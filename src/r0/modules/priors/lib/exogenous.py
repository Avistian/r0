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
    geometry: str = "independent"

    def __post_init__(self) -> None:
        if self.version != "exogenous-v1" or type(self.inputs) is not InputDistribution:
            raise ValueError("exogenous laws require inputs-v1 and exogenous-v1 descriptors")

        if self.geometry not in ("independent", "ball"):
            raise ValueError("unsupported exogenous geometry")
        if self.geometry == "ball" and self.inputs != InputDistribution():
            raise ValueError("ball geometry requires its standard Gaussian direction law")
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
        sources = tuple(
            {i for i, row in enumerate(self.factor) if row[j] != 0} if self.factor else {j}
            for j in range(width)
        )
        return tuple(set(range(width)) if s and self.geometry == "ball" else s for s in sources)

    @property
    def moment_order(self) -> float:
        """Positive absolute moments strictly below this bound are finite."""
        return (
            self.inputs.df
            if self.inputs.distribution == "student_t"
            and self.inputs.bound is None
            and self.geometry == "independent"
            else math.inf
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
    if spec.geometry == "ball":
        lengths = np.sqrt(np.square(values).sum(axis=1, keepdims=True))
        if np.any(lengths == 0) or not np.isfinite(lengths).all():
            raise ValueError("invalid ball direction; task was not retried")
        radii = rng("innovation", node, "radius").random((rows, 1)) ** (1 / width)
        values = values / lengths * radii
    if spec.factor:
        with np.errstate(over="ignore", invalid="ignore"):
            values = values @ np.asarray(spec.factor, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite transformed exogenous draw; task was not retried")
    return values


@dataclass(frozen=True, slots=True)
class NoiseResponse:
    offset: float = 1.0
    strength: float = 0.0
    power: float = 1.0

    def __post_init__(self) -> None:
        if any(
            type(v) not in (int, float) or not math.isfinite(v)
            for v in (self.offset, self.strength, self.power)
        ):
            raise ValueError("noise-response parameters must be finite numbers")
        if self.offset < 0 or self.strength < 0 or self.power <= 0:
            raise ValueError(
                "noise response requires nonnegative offset/strength and positive power"
            )
