"""Frozen named numerical draws with checked scope and replayable realizations."""

import math
from dataclasses import dataclass, replace
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
        if self.mode not in ("meta", "local", "global"):
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
    units: tuple[float, ...] = ()
    mode: str = "global"
    mean: float | None = None
    concentration: float | None = None

    def __post_init__(self) -> None:
        if type(self.request) is not NumericalParameter:
            raise ValueError("numerical preference requires a frozen request")
        if self.mode not in ("meta", "local", "global"):
            raise ValueError("unsupported numerical preference mode")
        if self.mode == "global":
            _validate_units(self.request, self.units, uniform=True)
        elif self.units != ():
            raise ValueError("local and meta preferences do not cache realizations")
        if self.mode == "meta":
            if (
                self.mean is None
                or not finite_number(self.mean)
                or not 0 < self.mean < 1
                or self.concentration is None
                or not finite_number(self.concentration)
                or not 0.1 <= self.concentration <= 10000
            ):
                raise ValueError("meta numerical preference requires valid Beta hyperparameters")
        elif self.mean is not None or self.concentration is not None:
            raise ValueError("Beta hyperparameters apply only to meta preferences")


def _validate_units(
    request: NumericalParameter, units: tuple[float, ...], *, uniform: bool = False
) -> None:
    if type(request) is not NumericalParameter:
        raise ValueError("numerical draw requires a frozen request")
    if (
        type(units) is not tuple
        or len(units) != math.prod(request.shape)
        or any(not finite_number(u) or not 0 <= u <= 1 or (uniform and u == 1) for u in units)
    ):
        raise ValueError("numerical units must match the shape and lie in the unit interval")


@dataclass(frozen=True, slots=True)
class NumericalDraw:
    request: NumericalParameter
    preference: int
    units: tuple[float, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        _validate_units(self.request, self.units)
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
    default_mode: str = "meta"
    version: str = "parameter-preferences-v1"

    def __post_init__(self) -> None:
        if self.version != "parameter-preferences-v1" or self.default_mode not in (
            "meta",
            "local",
            "global",
        ):
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
        names = [(p.request.name, p.mode) for p in self.preferences]
        if len(set(names)) != len(names):
            raise ValueError("numerical preference names must be unique within each mode")
        used: set[int] = set()
        for draw in self.draws:
            if draw.preference >= len(self.preferences):
                raise ValueError("numerical draw references an absent preference")
            preference = self.preferences[draw.preference]
            mode = self.default_mode if draw.request.mode == "meta" else draw.request.mode
            if draw.request.name != preference.request.name or mode != preference.mode:
                raise ValueError("numerical realization references a different preference scope")
            _validate_units(draw.request, draw.units, uniform=mode != "meta")
            if mode == "global" and (
                replace(draw.request, mode=mode) != replace(preference.request, mode=mode)
                or draw.units != preference.units
            ):
                raise ValueError("global numerical realization violates the cached contract")
            used.add(draw.preference)
        if used != set(range(len(self.preferences))):
            raise ValueError("numerical preferences must each have a retained realization")


class ParameterSampler:
    """One explicitly seeded sampling scope; freeze it for replay and inspection."""

    def __init__(self, rng: np.random.Generator, *, default_mode: str = "meta") -> None:
        if type(rng) is not np.random.Generator or default_mode not in ("meta", "local", "global"):
            raise ValueError("parameter sampler requires an injected Generator and supported mode")
        self._rng = rng
        self._default_mode = default_mode
        self._preferences: list[NumericalPreference] = []
        self._draws: list[NumericalDraw] = []
        self._names: dict[tuple[str, str], int] = {}

    def numerical(self, request: NumericalParameter) -> NumericalDraw:
        if type(request) is not NumericalParameter:
            raise ValueError("numerical sampling requires a frozen request")
        if len(self._draws) >= 10_000:
            raise ValueError("parameter preference draw limit exceeded")
        mode = self._default_mode if request.mode == "meta" else request.mode
        key = (request.name, mode)
        index = self._names.get(key)
        if index is None:
            index = len(self._preferences)
            if mode == "meta":
                mean = float(self._rng.random())
                concentration = math.exp(
                    math.log(0.1) + float(self._rng.random()) * math.log(100000)
                )
                preference = NumericalPreference(
                    request, mode=mode, mean=mean, concentration=concentration
                )
            elif mode == "global":
                units = tuple(float(u) for u in self._rng.random(math.prod(request.shape)))
                preference = NumericalPreference(request, units, mode)
            else:
                preference = NumericalPreference(request, mode=mode)
        else:
            preference = self._preferences[index]
            if mode == "global" and (
                replace(request, mode=mode) != replace(preference.request, mode=mode)
            ):
                raise ValueError("global numerical contract changed for a cached name")
        if mode == "global":
            units = preference.units
        elif mode == "meta":
            assert preference.mean is not None
            assert preference.concentration is not None
            units = tuple(
                float(u)
                for u in self._rng.beta(
                    preference.mean * preference.concentration,
                    (1 - preference.mean) * preference.concentration,
                    size=math.prod(request.shape),
                )
            )
        else:
            units = tuple(float(u) for u in self._rng.random(math.prod(request.shape)))
        draw = NumericalDraw(request, index, units, _values(request, units))
        if index == len(self._preferences):
            self._preferences.append(preference)
            self._names[key] = index
        self._draws.append(draw)
        return draw

    def freeze(self) -> ParameterPreferences:
        return ParameterPreferences(
            tuple(self._preferences), tuple(self._draws), self._default_mode
        )


def parameter_preferences_from_dict(data: dict[str, Any]) -> ParameterPreferences:
    def request(raw: dict[str, Any]) -> NumericalParameter:
        fields = dict(raw)
        fields["shape"] = tuple(fields.get("shape", ()))
        return NumericalParameter(**fields)

    def preference(raw: dict[str, Any]) -> NumericalPreference:
        values = dict(raw)
        values["request"] = request(values["request"])
        values["units"] = tuple(values["units"])
        return NumericalPreference(**values)

    fields = dict(data)
    fields["preferences"] = tuple(preference(p) for p in fields.get("preferences", ()))
    fields["draws"] = tuple(
        NumericalDraw(request(d["request"]), d["preference"], tuple(d["units"]), tuple(d["values"]))
        for d in fields.get("draws", ())
    )
    return ParameterPreferences(**fields)
