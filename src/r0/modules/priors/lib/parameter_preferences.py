"""Frozen named numerical draws with checked scope and replayable realizations."""

import copy
import math
from dataclasses import dataclass, replace
from decimal import Decimal, localcontext
from typing import Any

import numpy as np

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.preference_limits import PreferenceLimits

_DEFAULT_LIMITS = PreferenceLimits()


@dataclass(frozen=True, slots=True)
class NumericalParameter:
    name: str
    low: float
    high: float
    shape: tuple[int, ...] = ()
    mode: str = "meta"
    scale: str = "linear"
    integer: str = "none"
    boundary_mass: bool = False

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("numerical parameter name must be nonempty")
        if not finite_number(self.low) or not finite_number(self.high) or self.low > self.high:
            raise ValueError("numerical bounds must be ordered finite numbers")
        if self.mode not in ("meta", "local", "global"):
            raise ValueError("unsupported numerical sampling mode")
        if self.scale not in ("linear", "log") or self.integer not in ("none", "round", "floor"):
            raise ValueError("unsupported numerical scale or integer conversion")
        if self.scale == "log" and self.low <= 0:
            raise ValueError("log numerical bounds must be positive")
        if type(self.boundary_mass) is not bool:
            raise ValueError("boundary mass must be boolean")
        if self.integer == "floor" and (
            type(self.low) is not int or type(self.high) is not int or self.low >= self.high
        ):
            raise ValueError("floor sampling requires distinct ordered integer bounds")
        if (
            type(self.shape) is not tuple
            or len(self.shape) > 8
            or any(type(n) is not int or n < 1 for n in self.shape)
            or math.prod(self.shape) > 100_000
        ):
            raise ValueError("numerical shape requires at most eight positive bounded dimensions")

    @property
    def parameter_count(self) -> int:
        return 2 + len(self.shape)


def _values(request: NumericalParameter, units: tuple[float, ...]) -> tuple[float, ...]:
    values: list[float] = []
    with localcontext() as context:
        context.prec = 80 if request.scale == "log" else 2 * (309 + 1074) + 10
        low, high = Decimal.from_float(float(request.low)), Decimal.from_float(float(request.high))
        for unit in units:
            u = min(1.0, max(0.0, 1.2 * unit - 0.1)) if request.boundary_mass else unit
            if u == 0 or low == high:
                value = float(low)
            elif u == 1:
                value = float(high)
            elif request.scale == "log":
                value = float((low.ln() * (1 - Decimal(u)) + high.ln() * Decimal(u)).exp())
            else:
                value = float(low * (1 - Decimal(u)) + high * Decimal(u))
            if request.integer == "round":
                value = round(value)
            elif request.integer == "floor":
                value = max(int(request.low), min(int(request.high) - 1, math.floor(value)))
            values.append(value)
    return tuple(values)


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

    @property
    def parameter_count(self) -> int:
        return self.request.parameter_count + len(self.units) + (2 if self.mode == "meta" else 0)


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
            or (self.request.integer != "none" and any(type(v) is not int for v in self.values))
            or self.values != _values(self.request, self.units)
        ):
            raise ValueError("numerical realization does not match retained units and request")

    @property
    def parameter_count(self) -> int:
        return self.request.parameter_count + 1 + len(self.units) + len(self.values)


@dataclass(frozen=True, slots=True)
class ParameterPreferences:
    preferences: tuple[NumericalPreference, ...] = ()
    draws: tuple[NumericalDraw, ...] = ()
    default_mode: str = "meta"
    version: str = "parameter-preferences-v1"
    global_policy: str = "strict"
    limits: PreferenceLimits = _DEFAULT_LIMITS

    def __post_init__(self) -> None:
        if self.version != "parameter-preferences-v1" or self.default_mode not in (
            "meta",
            "local",
            "global",
        ):
            raise ValueError("unsupported parameter preferences version or default mode")
        if self.global_policy not in ("strict", "quantile"):
            raise ValueError("unsupported global numerical reuse policy")
        if type(self.preferences) is not tuple or any(
            type(p) is not NumericalPreference for p in self.preferences
        ):
            raise ValueError("parameter preferences must be frozen numerical preferences")
        if type(self.draws) is not tuple or any(type(d) is not NumericalDraw for d in self.draws):
            raise ValueError("parameter preferences require frozen draws")
        if type(self.limits) is not PreferenceLimits:
            raise ValueError("parameter preferences require frozen limits")
        self.limits.admit(
            draws=len(self.draws),
            values=sum(len(d.values) for d in self.draws),
            parameters=self.parameter_count,
        )
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
                not _global_compatible(draw.request, preference.request, self.global_policy)
                or draw.units != preference.units
            ):
                raise ValueError("global numerical realization violates the cached contract")
            if draw.preference not in used and (
                draw.preference != len(used) or draw.request != preference.request
            ):
                raise ValueError("numerical preference first request or order is inconsistent")
            used.add(draw.preference)
        if used != set(range(len(self.preferences))):
            raise ValueError("numerical preferences must each have a retained realization")

    @property
    def parameter_count(self) -> int:
        return sum(p.parameter_count for p in self.preferences) + sum(
            d.parameter_count for d in self.draws
        )


def _global_compatible(
    request: NumericalParameter, previous: NumericalParameter, policy: str
) -> bool:
    if policy == "quantile":
        return request.name == previous.name and request.shape == previous.shape
    return replace(request, mode="global") == replace(previous, mode="global")


class ParameterSampler:
    """One explicitly seeded sampling scope; freeze it for replay and inspection."""

    def __init__(
        self,
        rng: np.random.Generator,
        *,
        default_mode: str = "meta",
        global_policy: str = "strict",
        limits: PreferenceLimits = _DEFAULT_LIMITS,
    ) -> None:
        if type(rng) is not np.random.Generator or default_mode not in ("meta", "local", "global"):
            raise ValueError("parameter sampler requires an injected Generator and supported mode")
        if global_policy not in ("strict", "quantile"):
            raise ValueError("unsupported global numerical reuse policy")
        if type(limits) is not PreferenceLimits:
            raise ValueError("parameter sampler requires frozen limits")
        self._limits = limits
        self._value_count = 0
        self._parameter_count = 0
        self._global_policy = global_policy
        self._rng = rng
        self._default_mode = default_mode
        self._preferences: list[NumericalPreference] = []
        self._draws: list[NumericalDraw] = []
        self._names: dict[tuple[str, str], int] = {}

    @property
    def admission(self) -> dict[str, int]:
        return dict(
            draws=len(self._draws), values=self._value_count, parameters=self._parameter_count
        )

    def fork(
        self, rng: np.random.Generator, *, limits: PreferenceLimits | None = None
    ) -> "ParameterSampler":
        """Copy scope bookkeeping onto an injected stream, sharing frozen records."""
        if type(rng) is not np.random.Generator:
            raise ValueError("sampling fork requires an injected Generator")
        fork = copy.copy(self)
        fork._rng = rng
        if limits is not None:
            if type(limits) is not PreferenceLimits:
                raise ValueError("sampling fork requires frozen limits")
            limits.admit(**self.admission)
            fork._limits = limits
        fork._preferences = self._preferences.copy()
        fork._draws = self._draws.copy()
        fork._names = self._names.copy()
        return fork

    def numerical(self, request: NumericalParameter) -> NumericalDraw:
        if type(request) is not NumericalParameter:
            raise ValueError("numerical sampling requires a frozen request")
        mode = self._default_mode if request.mode == "meta" else request.mode
        key = (request.name, mode)
        index = self._names.get(key)
        width = math.prod(request.shape)
        additional = request.parameter_count + 1 + 2 * width
        if index is None:
            additional += request.parameter_count + (width if mode == "global" else 0)
            additional += 2 if mode == "meta" else 0
        self._limits.admit(
            draws=len(self._draws) + 1,
            values=self._value_count + width,
            parameters=self._parameter_count + additional,
        )
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
            if mode == "global" and not _global_compatible(
                request, preference.request, self._global_policy
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
        self._value_count += width
        self._parameter_count += additional
        return draw

    def freeze(self) -> ParameterPreferences:
        return ParameterPreferences(
            tuple(self._preferences),
            tuple(self._draws),
            self._default_mode,
            global_policy=self._global_policy,
            limits=self._limits,
        )


def raw_numerical_admission(data: dict[str, Any]) -> dict[str, int]:
    preferences, draws = data.get("preferences", ()), data.get("draws", ())
    parameters = sum(
        2
        + len(p["request"].get("shape", ()))
        + len(p.get("units", ()))
        + (2 if p.get("mode", "global") == "meta" else 0)
        for p in preferences
    ) + sum(
        3 + len(d["request"].get("shape", ())) + len(d["units"]) + len(d["values"]) for d in draws
    )
    return dict(
        draws=len(draws), values=sum(len(d["values"]) for d in draws), parameters=parameters
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
    limits = PreferenceLimits(**fields.get("limits", {}))
    raw_preferences, raw_draws = fields.get("preferences", ()), fields.get("draws", ())
    if len(raw_preferences) > len(raw_draws):
        raise ValueError("numerical preferences must each have a retained realization")
    # Admit the raw dimensions before allocating decoded tuples or recomputing values.
    limits.admit(**raw_numerical_admission(fields))
    fields["limits"] = limits
    fields["preferences"] = tuple(preference(p) for p in fields.get("preferences", ()))
    draws: list[NumericalDraw] = []
    for raw in raw_draws:
        values = dict(raw)
        values["request"] = request(values["request"])
        values["units"] = tuple(values["units"])
        values["values"] = tuple(values["values"])
        draws.append(NumericalDraw(**values))
    fields["draws"] = tuple(draws)
    return ParameterPreferences(**fields)
