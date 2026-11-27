"""Plan shared numerical/categorical matrix decisions before materializing arrays."""

from collections.abc import Callable
from dataclasses import fields

import numpy as np

from r0.modules.priors.lib.categorical_preferences import CategoricalParameter
from r0.modules.priors.lib.categorical_sampling import CategoricalSampler
from r0.modules.priors.lib.causal_activations import CausalActivation
from r0.modules.priors.lib.causal_matrix_sampling import MatrixPlan, sample_matrix_plan
from r0.modules.priors.lib.causal_nodes import (
    ActivatedCausalNode,
    CausalNode,
    MatrixCausalNode,
    StochasticCausalNode,
)
from r0.modules.priors.lib.causal_preferred import PreferredCausalRecipe
from r0.modules.priors.lib.causal_sampled import SampledCausalRecipe
from r0.modules.priors.lib.matrix_preference_prior import (
    FAMILY_VERSIONS,
    CausalMatrixPrior,
    component_request,
    distance_request,
    vector_requests,
)
from r0.modules.priors.lib.parameter_preferences import NumericalParameter, ParameterSampler
from r0.modules.priors.lib.preference_binding import PreferenceBinding
from r0.modules.priors.lib.preference_limits import PreferenceLimits
from r0.modules.priors.lib.preferred_contract import Path

_DEFAULT_PRIOR = CausalMatrixPrior()
_DEFAULT_LIMITS = PreferenceLimits()


class MatrixPlanner:
    def __init__(
        self,
        prior: CausalMatrixPrior,
        rng: np.random.Generator,
        activation: Callable[[Path], CausalActivation],
        *,
        numerical_limits: PreferenceLimits = _DEFAULT_LIMITS,
        categorical_limits: PreferenceLimits = _DEFAULT_LIMITS,
    ) -> None:
        self.prior, self.rng, self.activation = prior, rng, activation
        self.numerical = ParameterSampler(
            rng,
            default_mode=prior.mode,
            global_policy="quantile" if prior.mode == "global" else "strict",
            limits=numerical_limits,
        )
        self.categorical = CategoricalSampler(
            rng, default_mode=prior.mode, policy=prior.categorical_policy, limits=categorical_limits
        )
        self.bindings: list[PreferenceBinding] = []
        self.numeric_index = 0
        self.category_index = 0

    def number(self, request: NumericalParameter, path: Path, operation: str = "value") -> float:
        draw = self.numerical.numerical(request)
        self.bindings.append(PreferenceBinding("numerical", self.numeric_index, path, operation))
        self.numeric_index += 1
        return draw.values[0]

    def category(self, request: CategoricalParameter, path: Path) -> str:
        draw = self.categorical.categorical(request)
        self.bindings.append(PreferenceBinding("categorical", self.category_index, path))
        self.category_index += 1
        return draw.value

    def vector(self, width: int, path: Path, *, polynomial: bool) -> tuple[float, float] | None:
        if not polynomial:
            return None
        decay, scale = vector_requests(width)
        return self.number(decay, path + ("decay",)), self.number(
            scale, path + ("log_noise_scale",)
        )

    def plan(self, rows: int, columns: int, path: Path, *, simple: bool = False) -> MatrixPlan:
        version = self.category(self.prior.family_request(simple=simple), path + ("version",))
        family = next(k for k, v in FAMILY_VERSIONS.items() if v == version)
        if family == "gaussian":
            return MatrixPlan(rows, columns, family)
        if family == "transformed":
            activation = self.activation(path + ("activation",))
            child = self.plan(rows, columns, path + ("child", "source"), simple=True)
            return MatrixPlan(rows, columns, family, activation=activation, child=child)
        polynomial = self.prior.vector_family == "polynomial"
        if family == "importance":
            vectors = tuple(
                self.vector(columns, path + ("importance", i), polynomial=polynomial)
                for i in range(rows)
            )
            return MatrixPlan(
                rows, columns, family, polynomial=polynomial, vector_parameters=vectors
            )
        if family == "factorized":
            vector = self.vector(min(rows, columns), path + ("diagonal",), polynomial=polynomial)
            return MatrixPlan(
                rows, columns, family, polynomial=polynomial, vector_parameters=(vector,)
            )
        point = path + ("point_construction",)
        law = self.category(self.prior.point_request(), point + ("law",))
        count = (
            int(self.number(component_request(), point + ("centers",), "length"))
            if law == "gaussian-mixture"
            else 1
        )
        base = (
            "gaussian"
            if law == "gaussian-mixture"
            else self.category(self.prior.base_request(), point + ("base_law",))
        )
        component_polynomial = polynomial and law == "gaussian-mixture"
        component = self.vector(
            count, point + ("component_weights",), polynomial=component_polynomial
        )
        factors = tuple(
            self.vector(3, point + ("factor_weights", i), polynomial=polynomial)
            for i in range(count)
        )
        scale = self.number(distance_request(), path + ("distance_scale",))
        return MatrixPlan(
            rows,
            columns,
            family,
            polynomial=polynomial,
            point_law=law,
            components=count,
            component_polynomial=component_polynomial,
            base_law=base,
            vector_parameters=(component, *factors),
            distance_scale=scale,
        )


def sample_preferred_matrices(
    recipe: SampledCausalRecipe,
    *,
    rng: np.random.Generator,
    prior: CausalMatrixPrior = _DEFAULT_PRIOR,
) -> PreferredCausalRecipe:
    """Replace base Gaussian projections with a fully recorded shared matrix prior."""
    if (
        type(recipe) is not SampledCausalRecipe
        or recipe.construction != "gaussian-activation-prior-v1"
    ):
        raise ValueError("preferred matrix sampling requires a base activation-prior recipe")
    if type(prior) is not CausalMatrixPrior or type(rng) is not np.random.Generator:
        raise ValueError("preferred matrix sampling requires a frozen prior and injected Generator")

    def activation(path: Path) -> CausalActivation:
        index = int(
            rng.choice(len(recipe.activation_probabilities), p=recipe.activation_probabilities)
        )
        return recipe.activation_prior.sample_choice(index, rng)

    planner = MatrixPlanner(prior, rng, activation)
    plans = tuple(
        planner.plan(len(n.weights), len(n.bias), ("nodes", i, "weight_construction", "source"))
        if n.parents
        else None
        for i, n in enumerate(recipe.nodes)
    )
    numerical, categorical, bindings = (
        planner.numerical.freeze(),
        planner.categorical.freeze(),
        tuple(planner.bindings),
    )
    metadata = (
        numerical.parameter_count
        + categorical.parameter_count
        + sum(b.parameter_count for b in bindings)
    )
    defaults = StochasticCausalNode().structural_parameter_count
    recipe.limits.admit_dimensions(
        tuple(len(n.bias) for n in recipe.nodes),
        tuple(n.parents for n in recipe.nodes),
        rows=recipe.task.rows,
        calibration_rows=recipe.calibration_rows if recipe.normalize else 0,
        features=recipe.task.features,
        outputs=len(recipe.target),
        extra_parameters=metadata
        + sum(
            n.structural_parameter_count + (defaults + p.parameter_count if p is not None else 0)
            for n, p in zip(recipe.nodes, plans, strict=True)
        ),
        extra_working_channels=max(
            (
                StochasticCausalNode.working_channels_for_width(len(n.bias))
                for n in recipe.nodes
                if n.parents
            ),
            default=0,
        ),
    )
    nodes: list[CausalNode] = []
    for node, plan in zip(recipe.nodes, plans, strict=True):
        if plan is None:
            nodes.append(node)
            continue
        assert isinstance(node, ActivatedCausalNode)
        construction = sample_matrix_plan(plan, rng)
        nodes.append(
            MatrixCausalNode(
                parents=node.parents,
                weights=construction.materialize(),
                bias=node.bias,
                noise_scale=node.noise_scale,
                input_activation=node.input_activation,
                output_activation=node.output_activation,
                noise_position=node.noise_position,
                weight_construction=construction,
            )
        )
    values = {f.name: getattr(recipe, f.name) for f in fields(SampledCausalRecipe)}
    values.update(
        nodes=tuple(nodes),
        construction="activation-shared-matrices-v1",
        matrix_prior=prior,
        numerical_preferences=numerical,
        categorical_preferences=categorical,
        preference_bindings=bindings,
    )
    return PreferredCausalRecipe(**values)
