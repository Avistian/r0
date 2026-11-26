"""Explicit-RNG named categorical sampling with retained scope and admission."""

import numpy as np

from r0.modules.priors.lib.categorical_preferences import (
    CategoricalDraw,
    CategoricalParameter,
    CategoricalPreference,
    CategoricalPreferences,
)
from r0.modules.priors.lib.preference_limits import PreferenceLimits

_DEFAULT_LIMITS = PreferenceLimits()


class CategoricalSampler:
    def __init__(
        self,
        rng: np.random.Generator,
        *,
        default_mode: str = "global",
        limits: PreferenceLimits = _DEFAULT_LIMITS,
    ) -> None:
        if type(rng) is not np.random.Generator or default_mode not in ("local", "global"):
            raise ValueError(
                "categorical sampler requires an injected Generator and supported mode"
            )
        if type(limits) is not PreferenceLimits:
            raise ValueError("categorical sampler requires frozen limits")
        self._rng = rng
        self._default_mode = default_mode
        self._limits = limits
        self._preferences: list[CategoricalPreference] = []
        self._draws: list[CategoricalDraw] = []
        self._keys: dict[tuple[str, str], int] = {}
        self._parameters = 0

    def categorical(self, request: CategoricalParameter) -> CategoricalDraw:
        if type(request) is not CategoricalParameter:
            raise ValueError("categorical sampling requires a frozen request")
        mode = self._default_mode if request.mode == "meta" else request.mode
        key = (request.name, mode)
        reference = self._keys.get(key)
        if (
            reference is not None
            and self._preferences[reference].request.choices != request.choices
        ):
            raise ValueError("categorical choice contract changed for a cached name")
        additional = 3 + int(reference is None and mode == "global")
        self._limits.admit(
            draws=len(self._draws) + 1,
            values=len(self._draws) + 1,
            parameters=self._parameters + additional,
        )
        if reference is None:
            reference = len(self._preferences)
            preference = CategoricalPreference(
                request, mode, float(self._rng.random()) if mode == "global" else None
            )
        else:
            preference = self._preferences[reference]
        unit = preference.unit if mode == "global" else float(self._rng.random())
        assert unit is not None
        index = preference.choose(unit)
        draw = CategoricalDraw(request, reference, unit, index, request.choices[index])
        if reference == len(self._preferences):
            self._keys[key] = reference
            self._preferences.append(preference)
        self._draws.append(draw)
        self._parameters += additional
        return draw

    def freeze(self) -> CategoricalPreferences:
        return CategoricalPreferences(
            tuple(self._preferences), tuple(self._draws), self._default_mode, self._limits
        )
