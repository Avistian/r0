"""Strict frozen coordinate laws, separate from source selection and normalization."""

import math
from dataclasses import dataclass, fields
from typing import Any, cast

from r0.modules.priors.lib.activation_sampling import finite_number


@dataclass(frozen=True, slots=True)
class SourceRootCoordinate:
    family: str
    location: float = 0.0
    scale: float = 1.0
    probabilities: tuple[float, ...] = ()
    exponent: float = 2.0
    clip: int | None = None
    version: str = "source-root-coordinate-v1"

    def __post_init__(self) -> None:
        if self.version != "source-root-coordinate-v1" or self.family not in (
            "gaussian",
            "unit_uniform",
            "symmetric_uniform",
            "categorical",
            "zipf",
        ):
            raise ValueError("unsupported source root coordinate law or version")
        if not finite_number(self.location) or not finite_number(self.scale) or self.scale < 0:
            raise ValueError("root Gaussian location and scale must be finite, scale nonnegative")
        if self.family != "gaussian" and (self.location != 0 or self.scale != 1):
            raise ValueError("unused root Gaussian controls")
        if not finite_number(self.exponent) or self.exponent <= 1:
            raise ValueError("root Zipf exponent must be finite and greater than one")
        if self.clip is not None and (type(self.clip) is not int or not 1 <= self.clip < 2**63):
            raise ValueError("root Zipf clip must be a positive int64 value")
        if self.family != "zipf" and (self.exponent != 2 or self.clip is not None):
            raise ValueError("unused root Zipf controls")
        if type(self.probabilities) is not tuple:
            raise ValueError("root probabilities must be a frozen tuple")
        if self.family == "categorical":
            if (
                not 1 <= len(self.probabilities) <= 1024
                or any(not finite_number(x) or x < 0 for x in self.probabilities)
                or not math.isclose(math.fsum(self.probabilities), 1, rel_tol=1e-12, abs_tol=1e-12)
            ):
                raise ValueError("root category probabilities must sum to one")
        elif self.probabilities:
            raise ValueError("unused root category probabilities")

    @property
    def moment_order(self) -> float:
        return self.exponent - 1 if self.family == "zipf" and self.clip is None else math.inf


@dataclass(frozen=True, slots=True)
class SourceRootLaw:
    coordinates: tuple[SourceRootCoordinate, ...]
    version: str = "source-root-law-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "source-root-law-v1"
            or type(self.coordinates) is not tuple
            or not 1 <= len(self.coordinates) <= 1024
            or any(type(x) is not SourceRootCoordinate for x in self.coordinates)
        ):
            raise ValueError("source root law requires 1..1024 frozen coordinate laws")

    @property
    def width(self) -> int:
        return len(self.coordinates)

    @property
    def moment_order(self) -> float:
        return min(x.moment_order for x in self.coordinates)


def source_root_from_dict(data: dict[str, Any]) -> SourceRootLaw:
    if type(data) is not dict or set(data) != {"coordinates", "version"}:
        raise ValueError("source root law fields differ")
    values = data["coordinates"]
    if type(values) not in (tuple, list) or not 1 <= len(values) <= 1024:
        raise ValueError("invalid source root coordinate list")
    coordinates: list[SourceRootCoordinate] = []
    expected = {field.name for field in fields(SourceRootCoordinate)}
    for item in cast(list[Any] | tuple[Any, ...], values):
        if type(item) is not dict:
            raise ValueError("source root coordinate fields differ")
        raw = cast(dict[str, Any], item)
        if set(raw) != expected:
            raise ValueError("source root coordinate fields differ")
        probabilities = raw["probabilities"]
        if type(probabilities) not in (tuple, list) or len(probabilities) > 1024:
            raise ValueError("invalid source root probabilities")
        record: dict[str, Any] = {**raw, "probabilities": tuple(probabilities)}
        coordinates.append(SourceRootCoordinate(**record))
    return SourceRootLaw(tuple(coordinates), data["version"])
