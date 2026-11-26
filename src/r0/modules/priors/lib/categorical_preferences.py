"""Frozen categorical sampling laws and retained inverse-CDF realizations."""

from dataclasses import dataclass
from typing import Any

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.preference_limits import PreferenceLimits


@dataclass(frozen=True, slots=True)
class CategoricalParameter:
    name: str
    choices: tuple[str, ...]
    mode: str = "meta"

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("categorical name must be nonempty")
        if self.mode not in ("meta", "global", "local"):
            raise ValueError("unsupported categorical mode")
        if (
            type(self.choices) is not tuple
            or not 1 <= len(self.choices) <= 256
            or any(type(v) is not str or not v for v in self.choices)
            or len(set(self.choices)) != len(self.choices)
        ):
            raise ValueError("categorical choices must be 1..256 distinct nonempty strings")


def _unit(value: float) -> None:
    if not finite_number(value) or not 0 <= value < 1:
        raise ValueError("categorical unit draw must be in [0,1)")


@dataclass(frozen=True, slots=True)
class CategoricalPreference:
    request: CategoricalParameter
    mode: str
    unit: float | None = None

    def __post_init__(self) -> None:
        if type(self.request) is not CategoricalParameter or self.mode not in ("global", "local"):
            raise ValueError("categorical preference requires a frozen request and supported mode")
        if self.mode == "global":
            if self.unit is None:
                raise ValueError("global categorical preference requires a cached unit")
            _unit(self.unit)
        elif self.unit is not None:
            raise ValueError("local categorical preference cannot cache a unit")

    @property
    def parameter_count(self) -> int:
        return int(self.unit is not None)

    @property
    def probabilities(self) -> tuple[float, ...]:
        return (1.0 / len(self.request.choices),) * len(self.request.choices)

    def choose(self, unit: float) -> int:
        _unit(unit)
        return min(len(self.request.choices) - 1, int(unit * len(self.request.choices)))


@dataclass(frozen=True, slots=True)
class CategoricalDraw:
    request: CategoricalParameter
    preference: int
    unit: float
    index: int
    value: str

    def __post_init__(self) -> None:
        if type(self.request) is not CategoricalParameter:
            raise ValueError("categorical draw requires a frozen request")
        _unit(self.unit)
        if type(self.preference) is not int or self.preference < 0:
            raise ValueError("categorical preference index must be nonnegative")
        if (
            type(self.index) is not int
            or not 0 <= self.index < len(self.request.choices)
            or type(self.value) is not str
            or self.value != self.request.choices[self.index]
        ):
            raise ValueError("categorical realization must match a declared choice")

    @property
    def parameter_count(self) -> int:
        return 3  # Reference, unit and selected index; names/labels are not numeric parameters.


@dataclass(frozen=True, slots=True)
class CategoricalPreferences:
    preferences: tuple[CategoricalPreference, ...] = ()
    draws: tuple[CategoricalDraw, ...] = ()
    default_mode: str = "global"
    limits: PreferenceLimits = PreferenceLimits()
    version: str = "categorical-preferences-v1"

    def __post_init__(self) -> None:
        if self.version != "categorical-preferences-v1" or self.default_mode not in (
            "global",
            "local",
        ):
            raise ValueError("unsupported categorical record version or mode")
        if type(self.preferences) is not tuple or any(
            type(p) is not CategoricalPreference for p in self.preferences
        ):
            raise ValueError("categorical preferences must be a frozen tuple")
        if type(self.draws) is not tuple or any(type(d) is not CategoricalDraw for d in self.draws):
            raise ValueError("categorical draws must be a frozen tuple")
        if type(self.limits) is not PreferenceLimits:
            raise ValueError("categorical preferences require frozen limits")
        self.limits.admit(
            draws=len(self.draws), values=len(self.draws), parameters=self.parameter_count
        )
        keys = [(p.request.name, p.mode) for p in self.preferences]
        if len(set(keys)) != len(keys):
            raise ValueError("categorical preference scopes must be unique")
        used: set[int] = set()
        for draw in self.draws:
            if draw.preference >= len(self.preferences):
                raise ValueError("categorical realization references an absent preference")
            preference = self.preferences[draw.preference]
            mode = self.default_mode if draw.request.mode == "meta" else draw.request.mode
            if (
                draw.request.name != preference.request.name
                or mode != preference.mode
                or draw.request.choices != preference.request.choices
            ):
                raise ValueError("categorical contract differs from the preference scope")
            if preference.choose(draw.unit) != draw.index or (
                mode == "global" and draw.unit != preference.unit
            ):
                raise ValueError("categorical realization violates its probability law or cache")
            if draw.preference not in used and (
                draw.preference != len(used) or draw.request != preference.request
            ):
                raise ValueError("categorical first request or preference order is inconsistent")
            used.add(draw.preference)
        if used != set(range(len(self.preferences))):
            raise ValueError("categorical preferences must each have a retained realization")

    @property
    def parameter_count(self) -> int:
        return sum(p.parameter_count for p in self.preferences) + sum(
            d.parameter_count for d in self.draws
        )


def categorical_preferences_from_dict(data: dict[str, Any]) -> CategoricalPreferences:
    def request(raw: dict[str, Any]) -> CategoricalParameter:
        fields = dict(raw)
        fields["choices"] = tuple(fields["choices"])
        return CategoricalParameter(**fields)

    fields = dict(data)
    limits = PreferenceLimits(**fields.get("limits", {}))
    raw_preferences, raw_draws = fields.get("preferences", ()), fields.get("draws", ())
    limits.admit(
        draws=len(raw_draws),
        values=len(raw_draws),
        parameters=3 * len(raw_draws) + sum(p.get("unit") is not None for p in raw_preferences),
    )
    if len(raw_preferences) > len(raw_draws):
        raise ValueError("categorical preferences must each have a retained realization")
    preferences: list[CategoricalPreference] = []
    for raw in raw_preferences:
        values = dict(raw)
        values["request"] = request(values["request"])
        preferences.append(CategoricalPreference(**values))
    draws: list[CategoricalDraw] = []
    for raw in raw_draws:
        values = dict(raw)
        values["request"] = request(values["request"])
        draws.append(CategoricalDraw(**values))
    fields.update(preferences=tuple(preferences), draws=tuple(draws), limits=limits)
    return CategoricalPreferences(**fields)
