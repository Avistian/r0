"""Shared validation of retained matrix decisions and their controlled fields."""

from typing import Protocol

from r0.modules.priors.lib.categorical_preferences import CategoricalPreferences
from r0.modules.priors.lib.causal_nodes import CausalNode, MatrixCausalNode
from r0.modules.priors.lib.matrix_preference_prior import CausalMatrixPrior
from r0.modules.priors.lib.parameter_preferences import ParameterPreferences
from r0.modules.priors.lib.preference_binding import PreferenceBinding
from r0.modules.priors.lib.preferred_contract import Contract, matrix_contract
from r0.modules.priors.lib.transformed_weights import NormalizedWeights


class MatrixPreferences(Protocol):
    @property
    def nodes(self) -> tuple[CausalNode, ...]: ...
    @property
    def matrix_prior(self) -> CausalMatrixPrior: ...
    @property
    def numerical_preferences(self) -> ParameterPreferences: ...
    @property
    def categorical_preferences(self) -> CategoricalPreferences: ...
    @property
    def preference_bindings(self) -> tuple[PreferenceBinding, ...]: ...


def validate_matrix_preferences(recipe: MatrixPreferences) -> None:
    if (
        type(recipe.matrix_prior) is not CausalMatrixPrior
        or type(recipe.numerical_preferences) is not ParameterPreferences
        or type(recipe.categorical_preferences) is not CategoricalPreferences
        or type(recipe.preference_bindings) is not tuple
        or any(type(b) is not PreferenceBinding for b in recipe.preference_bindings)
    ):
        raise ValueError("preferred causal recipes require frozen priors, preferences and bindings")
    if (
        recipe.numerical_preferences.default_mode != recipe.matrix_prior.mode
        or recipe.categorical_preferences.default_mode != recipe.matrix_prior.mode
        or recipe.categorical_preferences.policy != recipe.matrix_prior.categorical_policy
        or recipe.numerical_preferences.global_policy
        != ("quantile" if recipe.matrix_prior.mode == "global" else "strict")
    ):
        raise ValueError("matrix preference policies disagree with their prior")
    contracts: list[Contract] = []
    for index, node in enumerate(recipe.nodes):
        if not node.parents:
            continue
        if not isinstance(node, MatrixCausalNode) or not isinstance(
            node.weight_construction, NormalizedWeights
        ):
            raise ValueError("preferred matrix recipes require every projection construction")
        contracts.extend(
            matrix_contract(
                node.weight_construction,
                recipe.matrix_prior,
                ("nodes", index, "weight_construction", "source"),
            )
        )
    expected = {(kind, path, operation): request for kind, path, operation, request in contracts}
    actual = {(b.kind, b.path, b.operation) for b in recipe.preference_bindings}
    if actual != set(expected) or len(actual) != len(recipe.preference_bindings):
        raise ValueError("preference bindings must cover each controlled matrix field exactly once")
    used: set[tuple[str, int]] = set()
    for binding in recipe.preference_bindings:
        key = (binding.kind, binding.path, binding.operation)
        if binding.kind == "numerical":
            if binding.draw >= len(recipe.numerical_preferences.draws):
                raise ValueError("numerical binding references an absent draw")
            numeric = recipe.numerical_preferences.draws[binding.draw]
            request, value = numeric.request, numeric.values[0]
        else:
            if binding.draw >= len(recipe.categorical_preferences.draws):
                raise ValueError("categorical binding references an absent draw")
            category = recipe.categorical_preferences.draws[binding.draw]
            request, value = category.request, category.value
        if request != expected[key] or binding.read(recipe) != value:
            raise ValueError("preference binding realization disagrees with its matrix field")
        used.add((binding.kind, binding.draw))
    all_draws = {("numerical", i) for i in range(len(recipe.numerical_preferences.draws))} | {
        ("categorical", i) for i in range(len(recipe.categorical_preferences.draws))
    }
    if used != all_draws or len(used) != len(recipe.preference_bindings):
        raise ValueError("preference bindings must use every top-level draw exactly once")
