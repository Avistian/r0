"""Frozen named numerical draws with checked scope and replayable realizations."""

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Any

import numpy as np

from r0.modules.priors.lib.activation_sampling import finite_number


@dataclass(frozen=True, slots=True)
class NumericalParameter:
    name: str
    low: float
    high: float
    shape: tuple[int, ...] = ()
    mode: str = "meta"

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("numerical parameter name must be nonempty")
        if not finite_number(self.low) or not finite_number(self.high) or self.low > self.high:
            raise ValueError("numerical bounds must be ordered finite numbers")
        if self.mode not in ("meta", "global"):
            raise ValueError("unsupported numerical sampling mode")
        if (
            type(self.shape) is not tuple
            or len(self.shape) > 8
            or any(type(n) is not int or n < 1 for n in self.shape)
            or math.prod(self.shape) > 100_000
        ):
            raise ValueError("numerical shape requires at most eight positive bounded dimensions")


def _values(request: NumericalParameter, units: tuple[float, ...]) -> tuple[float, ...]:
    with localcontext() as context:
        context.prec = 2 * (309 + 1074) + 10
        low, high = Decimal.from_float(float(request.low)), Decimal.from_float(float(request.high))
        return tuple(float(low * (1 - Decimal(u)) + high * Decimal(u)) for u in units)


@dataclass(frozen=True, slots=True)
class NumericalPreference:
    request: NumericalParameter
    units: tuple[float, ...]

    def __post_init__(self) -> None:
        if type(self.request) is not NumericalParameter:
            raise ValueError("numerical preference requires a frozen request")
        if (
            type(self.units) is not tuple
            or len(self.units) != math.prod(self.request.shape)
            or any(not finite_number(u) or not 0 <= u < 1 for u in self.units)
        ):
            raise ValueError("global numerical units must match the shape and lie in [0,1)")


@dataclass(frozen=True, slots=True)
class NumericalDraw:
    request: NumericalParameter
    preference: int
    units: tuple[float, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        NumericalPreference(self.request, self.units)
        if type(self.preference) is not int or self.preference < 0:
            raise ValueError("numerical preference index must be nonnegative")
        if (
            type(self.values) is not tuple
            or any(not finite_number(v) for v in self.values)
            or self.values != _values(self.request, self.units)
        ):
            raise ValueError("numerical realization does not match retained units and request")


@dataclass(frozen=True, slots=True)
class ParameterPreferences:
    preferences: tuple[NumericalPreference, ...] = ()
    draws: tuple[NumericalDraw, ...] = ()
    default_mode: str = "global"
    version: str = "parameter-preferences-v1"

    def __post_init__(self) -> None:
        if self.version != "parameter-preferences-v1" or self.default_mode != "global":
            raise ValueError("unsupported parameter preferences version or default mode")
        if type(self.preferences) is not tuple or any(
            type(p) is not NumericalPreference for p in self.preferences
        ):
            raise ValueError("parameter preferences must be frozen numerical preferences")
        if (
            type(self.draws) is not tuple
            or len(self.draws) > 10_000
            or any(type(d) is not NumericalDraw for d in self.draws)
        ):
            raise ValueError("parameter preferences require at most 10000 frozen draws")
        names = [p.request.name for p in self.preferences]
        if len(set(names)) != len(names):
            raise ValueError("global numerical preference names must be unique")
        used: set[int] = set()
        for draw in self.draws:
            if draw.preference >= len(self.preferences):
                raise ValueError("numerical draw references an absent preference")
            preference = self.preferences[draw.preference]
            if draw.request != preference.request or draw.units != preference.units:
                raise ValueError("global numerical realization violates the cached contract")
            used.add(draw.preference)
        if used != set(range(len(self.preferences))):
            raise ValueError("numerical preferences must each have a retained realization")


class ParameterSampler:
    """One explicitly seeded sampling scope; freeze it for replay and inspection."""

    def __init__(self, rng: np.random.Generator, *, default_mode: str = "global") -> None:
        if type(rng) is not np.random.Generator or default_mode != "global":
            raise ValueError("parameter sampler requires an injected Generator and global mode")
        self._rng = rng
        self._preferences: list[NumericalPreference] = []
        self._draws: list[NumericalDraw] = []
        self._names: dict[str, int] = {}

    def numerical(self, request: NumericalParameter) -> NumericalDraw:
        if type(request) is not NumericalParameter:
            raise ValueError("numerical sampling requires a frozen request")
        if len(self._draws) >= 10_000:
            raise ValueError("parameter preference draw limit exceeded")
        index = self._names.get(request.name)
        if index is None:
            index = len(self._preferences)
            units = tuple(float(u) for u in self._rng.random(math.prod(request.shape)))
            preference = NumericalPreference(request, units)
        else:
            preference = self._preferences[index]
            if request != preference.request:
                raise ValueError("global numerical contract changed for a cached name")
        draw = NumericalDraw(request, index, preference.units, _values(request, preference.units))
        if index == len(self._preferences):
            self._preferences.append(preference)
            self._names[request.name] = index
        self._draws.append(draw)
        return draw

    def freeze(self) -> ParameterPreferences:
        return ParameterPreferences(tuple(self._preferences), tuple(self._draws))


def parameter_preferences_from_dict(data: dict[str, Any]) -> ParameterPreferences:
    def request(raw: dict[str, Any]) -> NumericalParameter:
        fields = dict(raw)
        fields["shape"] = tuple(fields.get("shape", ()))
        return NumericalParameter(**fields)

    fields = dict(data)
    fields["preferences"] = tuple(
        NumericalPreference(request(p["request"]), tuple(p["units"]))
        for p in fields.get("preferences", ())
    )
    fields["draws"] = tuple(
        NumericalDraw(request(d["request"]), d["preference"], tuple(d["units"]), tuple(d["values"]))
        for d in fields.get("draws", ())
    )
    return ParameterPreferences(**fields)
