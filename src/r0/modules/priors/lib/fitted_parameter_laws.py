"""Explicit scalar parameter laws; source profiles never imply a fitting lifecycle."""

import math
from dataclasses import dataclass
from typing import Any, cast

from r0.modules.priors.lib.activation_sampling import finite_number


@dataclass(frozen=True, slots=True)
class FittedParameterLaw:
    kind: str = "fixed"
    values: tuple[int | float | str, ...] = (1,)
    probabilities: tuple[float, ...] = ()
    low: int | float = 1
    high: int | float = 4
    rate: float = 0.5
    offset: int = 1
    cap: int = 4

    def __post_init__(self) -> None:
        if self.kind not in (
            "fixed",
            "categorical",
            "integer-uniform",
            "uniform",
            "log-uniform",
            "capped-exponential",
        ):
            raise ValueError("unknown fitted parameter law")
        if type(self.values) is not tuple or type(self.probabilities) is not tuple:
            raise ValueError("parameter law options must be immutable")
        if (
            not finite_number(self.low)
            or not finite_number(self.high)
            or not finite_number(self.rate)
            or self.rate <= 0
            or not math.isfinite(1 / self.rate)
            or type(self.offset) is not int
            or type(self.cap) is not int
            or not 1 <= self.offset <= self.cap <= 1024
            or any(type(v) is not str and not finite_number(cast(float, v)) for v in self.values)
            or any(not finite_number(v) or v < 0 for v in self.probabilities)
            or bool(self.probabilities)
            and self.kind != "categorical"
        ):
            raise ValueError(
                "parameter fields must be finite; exponential rate/bounds must be valid"
            )
        if self.kind in ("fixed", "categorical"):
            if not 1 <= len(self.values) <= 128 or any(
                type(v) is not str and not finite_number(cast(float, v)) for v in self.values
            ):
                raise ValueError("parameter options must be bounded finite scalars or names")
            if self.kind == "fixed" and len(self.values) != 1:
                raise ValueError("fixed parameter law requires exactly one value")
            if self.probabilities and (
                self.kind != "categorical"
                or len(self.probabilities) != len(self.values)
                or any(not finite_number(v) or v < 0 for v in self.probabilities)
                or not math.isclose(math.fsum(self.probabilities), 1.0, rel_tol=1e-10)
            ):
                raise ValueError("categorical probabilities must be normalized and aligned")
        elif self.kind == "capped-exponential":
            if (
                not finite_number(self.rate)
                or self.rate <= 0
                or not math.isfinite(1 / self.rate)
                or type(self.offset) is not int
                or type(self.cap) is not int
                or not 1 <= self.offset <= self.cap <= 1024
            ):
                raise ValueError(
                    "capped exponential needs positive finite scale and integer bounds"
                )
        elif (
            not finite_number(self.low)
            or not finite_number(self.high)
            or self.low > self.high
            or self.kind == "integer-uniform"
            and (type(self.low) is not int or type(self.high) is not int)
            or self.kind == "log-uniform"
            and self.low <= 0
        ):
            raise ValueError("invalid fitted parameter interval")

    @property
    def extrema(self) -> tuple[Any, ...]:
        if self.kind in ("fixed", "categorical"):
            return self.values
        if self.kind == "capped-exponential":
            return self.offset, self.cap
        return self.low, self.high

    @property
    def stochastic(self) -> bool:
        return self.kind != "fixed"


PINNED_DEPTH = FittedParameterLaw(kind="capped-exponential", offset=2)
PINNED_COUNT = FittedParameterLaw(kind="capped-exponential", offset=1)
PINNED_SOURCE = "soda-inria/tabicl@9530bd65a7b806f6ad4f9f4335af745577e9d3cc:_tree_scm.py"
