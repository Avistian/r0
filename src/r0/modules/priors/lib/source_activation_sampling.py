"""Retained hierarchical activation selection using explicit named preference scopes."""

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from r0.modules.priors.lib.categorical_preferences import (
    CategoricalPreferences,
    categorical_preferences_from_dict,
)
from r0.modules.priors.lib.categorical_sampling import CategoricalSampler
from r0.modules.priors.lib.causal_activations import CausalActivation
from r0.modules.priors.lib.parameter_preferences import (
    ParameterPreferences,
    ParameterSampler,
    parameter_preferences_from_dict,
)
from r0.modules.priors.lib.preference_limits import PreferenceLimits
from r0.modules.priors.lib.source_activation_law import (
    SourceActivationSelection,
    exponent_request,
    fixed_request,
    selector_request,
)

_DEFAULT_LIMITS = PreferenceLimits()


@dataclass(frozen=True, slots=True)
class SourceActivationSelections:
    selections: tuple[SourceActivationSelection, ...]
    numerical: ParameterPreferences
    categorical: CategoricalPreferences
    version: str = "source-activation-selections-v1"

    def __post_init__(self) -> None:
        if self.version != "source-activation-selections-v1":
            raise ValueError("unsupported source activation selection version")
        if (
            type(self.selections) is not tuple
            or any(type(s) is not SourceActivationSelection for s in self.selections)
            or type(self.numerical) is not ParameterPreferences
            or type(self.categorical) is not CategoricalPreferences
        ):
            raise ValueError("source activation selections require frozen records")
        if self.numerical.default_mode != self.categorical.default_mode or (
            self.numerical.global_policy != "strict"
        ):
            raise ValueError("source activation selection scopes must agree")
        categorical_index, numerical_index = 0, 0
        for selection in self.selections:
            request = selector_request(selection.selector)
            if selection.slot_draw != categorical_index or categorical_index >= len(
                self.categorical.draws
            ):
                raise ValueError("source selection slot references must cover ordered draws")
            slot = self.categorical.draws[categorical_index]
            categorical_index += 1
            if slot.request != request:
                raise ValueError("source selection slot request does not match its selector")
            if slot.value.startswith("fixed-"):
                if (
                    selection.primitive_draw != categorical_index
                    or selection.exponent_draw is not None
                    or categorical_index >= len(self.categorical.draws)
                ):
                    raise ValueError("source fixed selection requires its ordered primitive draw")
                primitive = self.categorical.draws[categorical_index]
                categorical_index += 1
                if primitive.request != fixed_request():
                    raise ValueError("source selection primitive catalog differs")
                expected = CausalActivation(primitive.value)
            else:
                if (
                    selection.primitive_draw is not None
                    or selection.exponent_draw != numerical_index
                    or numerical_index >= len(self.numerical.draws)
                ):
                    raise ValueError("source power selection requires its ordered exponent draw")
                exponent = self.numerical.draws[numerical_index]
                numerical_index += 1
                if exponent.request != exponent_request(slot.value):
                    raise ValueError("source selection exponent request differs")
                expected = CausalActivation(slot.value, (("exponent", exponent.values[0]),))
            if selection.activation != expected:
                raise ValueError("source activation selection differs from its retained draws")
        if categorical_index != len(self.categorical.draws) or numerical_index != len(
            self.numerical.draws
        ):
            raise ValueError("source activation selections must bind every retained draw")

    @property
    def parameter_count(self) -> int:
        return (
            self.numerical.parameter_count
            + self.categorical.parameter_count
            + sum(s.parameter_count for s in self.selections)
        )


class SourceActivationSampler:
    """Select primitives; standard selections still need independent wrapper calibration.

    Numerical and categorical records have separate explicit limits. The categorical
    limit includes its nested weight hyperpriors. Consumers admit the combined frozen
    parameter_count when integrating selections into a mechanism.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        *,
        mode: str = "meta",
        policy: str = "shared",
        numerical_limits: PreferenceLimits = _DEFAULT_LIMITS,
        categorical_limits: PreferenceLimits = _DEFAULT_LIMITS,
    ) -> None:
        self._rng = rng
        self._numerical = ParameterSampler(rng, default_mode=mode, limits=numerical_limits)
        self._categorical = CategoricalSampler(
            rng, default_mode=mode, policy=policy, limits=categorical_limits
        )
        self._selections: list[SourceActivationSelection] = []
        self._categorical_count = 0
        self._numerical_count = 0

    def sample(self, selector: str = "plain") -> CausalActivation:
        request = selector_request(selector)
        staged_rng = copy.deepcopy(self._rng)
        numerical = self._numerical.fork(staged_rng)
        categorical = self._categorical.fork(staged_rng)
        slot_index = self._categorical_count
        slot = categorical.categorical(request)
        categorical_count, numerical_count = slot_index + 1, self._numerical_count
        if slot.value.startswith("fixed-"):
            primitive_index = categorical_count
            primitive = categorical.categorical(fixed_request())
            categorical_count += 1
            activation = CausalActivation(primitive.value)
            selection = SourceActivationSelection(
                selector, activation, slot_index, primitive_draw=primitive_index
            )
        else:
            exponent_index = numerical_count
            exponent = numerical.numerical(exponent_request(slot.value))
            numerical_count += 1
            activation = CausalActivation(slot.value, (("exponent", exponent.values[0]),))
            selection = SourceActivationSelection(
                selector, activation, slot_index, exponent_draw=exponent_index
            )
        # A rejected child request commits neither the parent draw nor RNG advancement.
        self._rng.bit_generator.state = staged_rng.bit_generator.state
        self._numerical, self._categorical = numerical, categorical
        self._categorical_count, self._numerical_count = categorical_count, numerical_count
        self._selections.append(selection)
        return activation

    def freeze(self) -> SourceActivationSelections:
        return SourceActivationSelections(
            tuple(self._selections), self._numerical.freeze(), self._categorical.freeze()
        )


def source_activation_selections_from_dict(data: dict[str, Any]) -> SourceActivationSelections:
    fields = dict(data)
    # The bounded preference decoders run before allocating selection records.
    numerical = parameter_preferences_from_dict(fields.pop("numerical"))
    categorical = categorical_preferences_from_dict(fields.pop("categorical"))
    raw = fields.pop("selections")
    if len(raw) > len(categorical.draws):
        raise ValueError("source activation selection count exceeds slot draws")
    selections: list[SourceActivationSelection] = []
    for item in raw:
        values = dict(item)
        activation = dict(values.pop("activation"))
        activation["parameters"] = tuple(tuple(p) for p in activation.get("parameters", ()))
        selections.append(
            SourceActivationSelection(activation=CausalActivation(**activation), **values)
        )
    return SourceActivationSelections(tuple(selections), numerical, categorical, **fields)
