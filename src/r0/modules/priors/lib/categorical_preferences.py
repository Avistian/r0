"""Frozen categorical sampling laws and retained inverse-CDF realizations."""

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, cast

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.parameter_preferences import (
    NumericalParameter,
    ParameterPreferences,
    parameter_preferences_from_dict,
)
from r0.modules.priors.lib.preference_limits import PreferenceLimits
from r0.modules.priors.lib.weight_vectors import PolynomialWeightVector, weight_vector_from_dict


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
    weights: PolynomialWeightVector | None = None
    hyperparameters: ParameterPreferences | None = None

    def __post_init__(self) -> None:
        if type(self.request) is not CategoricalParameter or self.mode not in (
            "global",
            "local",
            "meta",
        ):
            raise ValueError("categorical preference requires a frozen request and supported mode")
        if self.mode == "global":
            if self.unit is None:
                raise ValueError("global categorical preference requires a cached unit")
            _unit(self.unit)
        elif self.unit is not None:
            raise ValueError("local/meta categorical preference cannot cache a unit")
        if self.mode == "meta":
            if type(self.weights) is not PolynomialWeightVector or (
                len(self.weights.draws) != len(self.request.choices)
            ):
                raise ValueError(
                    "meta categorical preference requires a matching polynomial vector"
                )
            if type(self.hyperparameters) is not ParameterPreferences:
                raise ValueError("categorical polynomial hyperparameters must be retained")
            hyper = self.hyperparameters
            expected = categorical_weight_requests(len(self.request.choices))
            if (
                hyper.default_mode != "meta"
                or len(hyper.preferences) != 2
                or tuple(d.request for d in hyper.draws) != expected
                or tuple(p.mode for p in hyper.preferences) != ("meta", "meta")
                or self.weights.decay != hyper.draws[0].values[0]
                or self.weights.log_noise_scale != hyper.draws[1].values[0]
            ):
                raise ValueError("categorical polynomial hyperparameters disagree with the vector")
        elif self.weights is not None or self.hyperparameters is not None:
            raise ValueError("categorical weights and hyperparameters apply only to meta mode")

    @property
    def parameter_count(self) -> int:
        return (
            int(self.unit is not None)
            + (self.weights.parameter_count if self.weights is not None else 0)
            + (self.hyperparameters.parameter_count if self.hyperparameters is not None else 0)
        )

    @property
    def probabilities(self) -> tuple[float, ...]:
        if self.weights is None:
            return (1.0 / len(self.request.choices),) * len(self.request.choices)
        weights = self.weights.values()
        total = math.fsum(weights)
        probabilities = tuple(w / total for w in weights)
        if any(w > 0 and p == 0 for w, p in zip(weights, probabilities, strict=True)):
            raise ValueError(
                "categorical probabilities underflow in binary64 view; use log_probabilities"
            )
        return probabilities

    @property
    def log_probabilities(self) -> tuple[float, ...]:
        """Log view of the represented masses, retaining subnormal positive support."""
        if self.weights is None:
            return (-math.log(len(self.request.choices)),) * len(self.request.choices)
        weights = self.weights.values()
        normalizer = math.log(math.fsum(weights))
        return tuple(math.log(w) - normalizer if w else -math.inf for w in weights)

    def choose(self, unit: float) -> int:
        _unit(unit)
        if self.weights is None:
            return min(len(self.request.choices) - 1, int(unit * len(self.request.choices)))
        # Compare exact rational masses of the recorded binary64 weights: no
        # cumulative rounding can erase a small bin or select a zero-weight bin.
        masses = tuple(Fraction(w) for w in self.weights.values())
        threshold = Fraction(unit) * sum(masses)
        cumulative = Fraction(0)
        for index, mass in enumerate(masses):
            cumulative += mass
            if threshold < cumulative:
                return index
        raise ValueError("categorical inverse CDF has no positive support")


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
    default_mode: str = "meta"
    limits: PreferenceLimits = PreferenceLimits()
    version: str = "categorical-preferences-v1"
    policy: str = "shared"

    def __post_init__(self) -> None:
        if self.version != "categorical-preferences-v1" or self.default_mode not in (
            "global",
            "local",
            "meta",
        ):
            raise ValueError("unsupported categorical record version or mode")
        if self.policy not in ("shared", "refresh"):
            raise ValueError("unsupported categorical preference reuse policy")
        if type(self.preferences) is not tuple or any(
            type(p) is not CategoricalPreference for p in self.preferences
        ):
            raise ValueError("categorical preferences must be a frozen tuple")
        if type(self.draws) is not tuple or any(type(d) is not CategoricalDraw for d in self.draws):
            raise ValueError("categorical draws must be a frozen tuple")
        if type(self.limits) is not PreferenceLimits:
            raise ValueError("categorical preferences require frozen limits")
        self.limits.admit(
            draws=self.draw_count, values=self.value_count, parameters=self.parameter_count
        )
        keys = [
            (p.request.name, p.mode)
            for p in self.preferences
            if self.policy == "shared" or p.mode != "meta"
        ]
        if len(set(keys)) != len(keys):
            raise ValueError("categorical preference scopes must be unique")
        contracts: dict[tuple[str, str], tuple[str, ...]] = {}
        for preference in self.preferences:
            key = (preference.request.name, preference.mode)
            if key in contracts and contracts[key] != preference.request.choices:
                raise ValueError("categorical choice contract changed within a named scope")
            contracts[key] = preference.request.choices
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
            if self.policy == "refresh" and mode == "meta" and draw.preference in used:
                raise ValueError("refreshed categorical scope cannot reuse a probability vector")
            if draw.preference not in used and (
                draw.preference != len(used) or draw.request != preference.request
            ):
                raise ValueError("categorical first request or preference order is inconsistent")
            used.add(draw.preference)
        if used != set(range(len(self.preferences))):
            raise ValueError("categorical preferences must each have a retained realization")

    @property
    def draw_count(self) -> int:
        return len(self.draws) + sum(
            len(p.hyperparameters.draws) for p in self.preferences if p.hyperparameters is not None
        )

    @property
    def value_count(self) -> int:
        return len(self.draws) + sum(
            len(d.values)
            for p in self.preferences
            if p.hyperparameters is not None
            for d in p.hyperparameters.draws
        )

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
        draws=len(raw_draws)
        + sum(
            len(cast(dict[str, Any], p.get("hyperparameters") or {}).get("draws", ()))
            for p in raw_preferences
        ),
        values=len(raw_draws)
        + sum(
            len(d["values"])
            for p in raw_preferences
            for d in cast(dict[str, Any], p.get("hyperparameters") or {}).get("draws", ())
        ),
        parameters=raw_parameter_count(raw_preferences) + raw_parameter_count(raw_draws),
    )
    if len(raw_preferences) > len(raw_draws):
        raise ValueError("categorical preferences must each have a retained realization")
    preferences: list[CategoricalPreference] = []
    for raw in raw_preferences:
        values = dict(raw)
        values["request"] = request(values["request"])
        if values.get("weights") is not None:
            values["weights"] = weight_vector_from_dict(values["weights"])
        if values.get("hyperparameters") is not None:
            values["hyperparameters"] = parameter_preferences_from_dict(values["hyperparameters"])
        preferences.append(CategoricalPreference(**values))
    draws: list[CategoricalDraw] = []
    for raw in raw_draws:
        values = dict(raw)
        values["request"] = request(values["request"])
        draws.append(CategoricalDraw(**values))
    fields.update(preferences=tuple(preferences), draws=tuple(draws), limits=limits)
    return CategoricalPreferences(**fields)


def categorical_weight_requests(width: int) -> tuple[NumericalParameter, NumericalParameter]:
    return (
        NumericalParameter(
            "simple_random_weights_decay_rate",
            0.1 / math.log1p(width),
            6.0,
            shape=(1,),
            scale="log",
            boundary_mass=True,
        ),
        NumericalParameter(
            "simple_random_weights_std_scale", 0.0001, 10.0, shape=(1,), scale="log"
        ),
    )


def raw_parameter_count(value: Any) -> int:
    """Count raw retained numeric leaves before allocating decoded vectors."""
    if isinstance(value, dict):
        return sum(
            raw_parameter_count(v)
            for k, v in cast(dict[str, Any], value).items()
            if k not in ("limits", "exact_scores_limit")
        )
    if isinstance(value, (list, tuple)):
        return sum(raw_parameter_count(v) for v in cast(list[Any] | tuple[Any, ...], value))
    return int(type(value) in (int, float))
