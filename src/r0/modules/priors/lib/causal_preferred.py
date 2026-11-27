"""Causal recipes whose named sampling decisions are bound to matrix fields."""

from dataclasses import asdict, dataclass
from typing import Any

from r0.modules.priors.lib.categorical_preferences import CategoricalPreferences
from r0.modules.priors.lib.causal_nodes import MatrixCausalNode
from r0.modules.priors.lib.causal_sampled import SampledCausalRecipe
from r0.modules.priors.lib.matrix_preference_prior import CausalMatrixPrior
from r0.modules.priors.lib.parameter_preferences import ParameterPreferences
from r0.modules.priors.lib.preference_binding import PreferenceBinding
from r0.modules.priors.lib.preferred_contract import Contract, matrix_contract
from r0.modules.priors.lib.transformed_weights import NormalizedWeights


@dataclass(frozen=True, slots=True)
class PreferredCausalRecipe(SampledCausalRecipe):
    matrix_prior: CausalMatrixPrior = CausalMatrixPrior()
    numerical_preferences: ParameterPreferences = ParameterPreferences()
    categorical_preferences: CategoricalPreferences = CategoricalPreferences()
    preference_bindings: tuple[PreferenceBinding, ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.matrix_prior) is not CausalMatrixPrior
            or type(self.numerical_preferences) is not ParameterPreferences
            or type(self.categorical_preferences) is not CategoricalPreferences
            or type(self.preference_bindings) is not tuple
            or any(type(b) is not PreferenceBinding for b in self.preference_bindings)
        ):
            raise ValueError(
                "preferred causal recipes require frozen priors, preferences and bindings"
            )
        if self.construction != "activation-shared-matrices-v1":
            raise ValueError("preferred causal recipes require their shared matrix construction")
        if (
            self.numerical_preferences.default_mode != self.matrix_prior.mode
            or self.categorical_preferences.default_mode != self.matrix_prior.mode
            or self.categorical_preferences.policy != self.matrix_prior.categorical_policy
            or self.numerical_preferences.global_policy
            != ("quantile" if self.matrix_prior.mode == "global" else "strict")
        ):
            raise ValueError("matrix preference policies disagree with their prior")
        SampledCausalRecipe.__post_init__(self)
        contracts: list[Contract] = []
        for index, node in enumerate(self.nodes):
            if not node.parents:
                continue
            if not isinstance(node, MatrixCausalNode) or not isinstance(
                node.weight_construction, NormalizedWeights
            ):
                raise ValueError("preferred matrix recipes require every projection construction")
            contracts.extend(
                matrix_contract(
                    node.weight_construction,
                    self.matrix_prior,
                    ("nodes", index, "weight_construction", "source"),
                )
            )
        expected = {
            (kind, path, operation): request for kind, path, operation, request in contracts
        }
        actual = {(b.kind, b.path, b.operation) for b in self.preference_bindings}
        if actual != set(expected) or len(actual) != len(self.preference_bindings):
            raise ValueError(
                "preference bindings must cover each controlled matrix field exactly once"
            )
        used: set[tuple[str, int]] = set()
        for binding in self.preference_bindings:
            key = (binding.kind, binding.path, binding.operation)
            if binding.kind == "numerical":
                if binding.draw >= len(self.numerical_preferences.draws):
                    raise ValueError("numerical binding references an absent draw")
                numeric = self.numerical_preferences.draws[binding.draw]
                request, value = numeric.request, numeric.values[0]
            else:
                if binding.draw >= len(self.categorical_preferences.draws):
                    raise ValueError("categorical binding references an absent draw")
                category = self.categorical_preferences.draws[binding.draw]
                request, value = category.request, category.value
            if request != expected[key] or binding.read(self) != value:
                raise ValueError("preference binding realization disagrees with its matrix field")
            used.add((binding.kind, binding.draw))
        all_draws = {("numerical", i) for i in range(len(self.numerical_preferences.draws))} | {
            ("categorical", i) for i in range(len(self.categorical_preferences.draws))
        }
        if used != all_draws or len(used) != len(self.preference_bindings):
            raise ValueError("preference bindings must use every top-level draw exactly once")

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
