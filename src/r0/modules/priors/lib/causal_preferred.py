"""Causal recipes whose named sampling decisions are bound to matrix fields."""

from dataclasses import asdict, dataclass
from typing import Any

from r0.modules.priors.lib.categorical_preferences import CategoricalPreferences
from r0.modules.priors.lib.causal_sampled import SampledCausalRecipe
from r0.modules.priors.lib.matrix_preference_prior import CausalMatrixPrior
from r0.modules.priors.lib.matrix_preferences import validate_matrix_preferences
from r0.modules.priors.lib.parameter_preferences import ParameterPreferences
from r0.modules.priors.lib.preference_binding import PreferenceBinding


@dataclass(frozen=True, slots=True)
class PreferredCausalRecipe(SampledCausalRecipe):
    matrix_prior: CausalMatrixPrior = CausalMatrixPrior()
    numerical_preferences: ParameterPreferences = ParameterPreferences()
    categorical_preferences: CategoricalPreferences = CategoricalPreferences()
    preference_bindings: tuple[PreferenceBinding, ...] = ()

    def __post_init__(self) -> None:
        if self.construction != "activation-shared-matrices-v1":
            raise ValueError("preferred causal recipes require their shared matrix construction")
        validate_matrix_preferences(self)
        SampledCausalRecipe.__post_init__(self)

    @property
    def sampling_parameter_count(self) -> int:
        return (
            self.numerical_preferences.parameter_count
            + self.categorical_preferences.parameter_count
            + sum(b.parameter_count for b in self.preference_bindings)
        )

    @property
    def sampling_record(self) -> dict[str, Any]:
        return {
            **super(PreferredCausalRecipe, self).sampling_record,
            "matrix_prior": asdict(self.matrix_prior),
            "numerical_preferences": asdict(self.numerical_preferences),
            "categorical_preferences": asdict(self.categorical_preferences),
            "preference_bindings": tuple(asdict(b) for b in self.preference_bindings),
        }
