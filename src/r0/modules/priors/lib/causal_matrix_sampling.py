"""Plan and admit recorded matrix-family draws before allocating their arrays."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from r0.modules.priors.lib.causal_activations import CausalActivation
from r0.modules.priors.lib.causal_nodes import (
    ActivatedCausalNode,
    CausalNode,
    MatrixCausalNode,
    StochasticCausalNode,
)
from r0.modules.priors.lib.causal_sampled import SampledCausalRecipe
from r0.modules.priors.lib.kernel_points import KernelPointCloud
from r0.modules.priors.lib.transformed_weights import (
    GaussianWeights,
    NormalizedWeights,
    SimpleWeights,
    TransformedWeights,
)
from r0.modules.priors.lib.weight_geometry import Matrix
from r0.modules.priors.lib.weight_matrices import (
    ImportanceFactorizedWeights,
    ImportanceWeights,
    LaplaceKernelWeights,
)
from r0.modules.priors.lib.weight_vector_sampling import sample_polynomial_weights
from r0.modules.priors.lib.weight_vectors import UniformWeightVector, WeightVector

_FAMILIES = ("gaussian", "importance", "factorized", "kernel", "transformed")
_POINT_DIMENSION = 3


@dataclass(frozen=True, slots=True)
class MatrixPlan:
    rows: int
    columns: int
    family: str
    polynomial: bool = False
    point_law: str = "gaussian-mixture"
    components: int = 1
    component_polynomial: bool = False
    base_law: str = "gaussian"
    activation: CausalActivation = CausalActivation()
    child: MatrixPlan | None = None

    @property
    def parameter_count(self) -> int:
        n, m = self.rows, self.columns
        if self.family == "gaussian":
            raw = n * m
        elif self.family == "importance":
            raw = n * m + n * _vector_parameters(m, self.polynomial) + 1
        elif self.family == "factorized":
            k = min(n, m)
            raw = n * k + k * m + _vector_parameters(k, self.polynomial)
        elif self.family == "kernel":
            d, k, points = _POINT_DIMENSION, self.components, n + m
            # Dense geometry + signs + distance scale, then the point-cloud record.
            raw = (
                points * d
                + n * m
                + 1
                + (
                    k * d
                    + k * d * d
                    + k * _vector_parameters(d, self.polynomial)
                    + _vector_parameters(k, self.component_polynomial)
                    + points
                    + points * d
                    + 2
                    + k
                )
            )
        else:
            assert self.family == "transformed"
            assert self.child is not None
            raw = self.child.parameter_count + len(self.activation.parameters) + n * m + 1
        # Every planned matrix has independent top/child jitter and row normalization.
        return raw + n * m + 2


def _vector_parameters(width: int, polynomial: bool) -> int:
    return 2 * width + 2 if polynomial else width


def _plan(
    n: int,
    m: int,
    recipe: SampledCausalRecipe,
    rng: np.random.Generator,
    *,
    simple: bool = False,
    gaussian_only: bool = False,
) -> MatrixPlan:
    family = (
        "gaussian" if gaussian_only else str(rng.choice(_FAMILIES[:4] if simple else _FAMILIES))
    )
    if family == "transformed":
        # Matrix stages independently draw parameters from the saved choice law;
        # node-level homogeneous parameters are not asserted for matrix stages.
        index = int(
            rng.choice(len(recipe.activation_probabilities), p=recipe.activation_probabilities)
        )
        activation = recipe.activation_prior.sample_choice(index, rng)
        return MatrixPlan(
            n, m, family, activation=activation, child=_plan(n, m, recipe, rng, simple=True)
        )
    polynomial = bool(rng.integers(2)) if family != "gaussian" else False
    if family != "kernel":
        return MatrixPlan(n, m, family, polynomial=polynomial)
    law = str(rng.choice(("gaussian-mixture", "random-covariance")))
    components = (
        min(15, int(math.exp(float(rng.random()) * math.log(16))))
        if law == "gaussian-mixture"
        else 1
    )
    return MatrixPlan(
        n,
        m,
        family,
        polynomial=polynomial,
        point_law=law,
        components=components,
        component_polynomial=bool(rng.integers(2)) if law == "gaussian-mixture" else False,
        base_law="gaussian"
        if law == "gaussian-mixture"
        else str(rng.choice(("gaussian", "uniform"))),
    )


def sample_matrix_profile(
    recipe: SampledCausalRecipe, rng: np.random.Generator, *, gaussian_only: bool = False
) -> SampledCausalRecipe:
    plans = tuple(
        _plan(len(n.weights), len(n.bias), recipe, rng, gaussian_only=gaussian_only)
        if n.parents
        else None
        for n in recipe.nodes
    )
    defaults = StochasticCausalNode().structural_parameter_count
    recipe.limits.admit_dimensions(
        tuple(len(n.bias) for n in recipe.nodes),
        tuple(n.parents for n in recipe.nodes),
        rows=recipe.task.rows,
        calibration_rows=recipe.calibration_rows if recipe.normalize else 0,
        features=recipe.task.features,
        outputs=len(recipe.target),
        extra_parameters=sum(
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
        record = _sample(plan, rng)
        nodes.append(
            MatrixCausalNode(
                parents=node.parents,
                weights=record.materialize(),
                bias=node.bias,
                noise_scale=node.noise_scale,
                input_activation=node.input_activation,
                output_activation=node.output_activation,
                noise_position=node.noise_position,
                weight_construction=record,
            )
        )
    construction = "activation-gaussian-matrices-v1" if gaussian_only else "activation-matrices-v1"
    return replace(recipe, nodes=tuple(nodes), construction=construction)


def _gaussian(rows: int, columns: int, rng: np.random.Generator) -> Matrix:
    return tuple(tuple(float(v) for v in row) for row in rng.normal(size=(rows, columns)))


def _weight(width: int, polynomial: bool, rng: np.random.Generator) -> WeightVector:
    return (
        sample_polynomial_weights(width, rng)
        if polynomial
        else UniformWeightVector(tuple(float(v) for v in rng.random(width)))
    )


def _sample(plan: MatrixPlan, rng: np.random.Generator) -> NormalizedWeights:
    n, m = plan.rows, plan.columns
    raw: SimpleWeights | TransformedWeights
    if plan.family == "gaussian":
        raw = GaussianWeights(_gaussian(n, m, rng))
    elif plan.family == "importance":
        raw = ImportanceWeights(
            gaussian=_gaussian(n, m, rng),
            importance=tuple(_weight(m, plan.polynomial, rng) for _ in range(n)),
        )
    elif plan.family == "factorized":
        k = min(n, m)
        raw = ImportanceFactorizedWeights(
            left=_gaussian(n, k, rng),
            right=_gaussian(k, m, rng),
            diagonal=_weight(k, plan.polynomial, rng),
        )
    elif plan.family == "kernel":
        cloud = _point_cloud(plan, rng)
        points = cloud.materialize()
        raw = LaplaceKernelWeights(
            left_points=points[:n],
            right_points=points[n:],
            signs=tuple(tuple(float(np.sign(v)) for v in row) for row in _gaussian(n, m, rng)),
            distance_scale=math.exp(float(rng.uniform(math.log(0.1), math.log(10)))),
            point_construction=cloud,
        )
    else:
        assert plan.family == "transformed"
        assert plan.child is not None
        raw = TransformedWeights(
            child=_sample(plan.child, rng), activation=plan.activation, jitter=_gaussian(n, m, rng)
        )
    return NormalizedWeights(source=raw, jitter=_gaussian(n, m, rng))


def _point_cloud(plan: MatrixPlan, rng: np.random.Generator) -> KernelPointCloud:
    k, d, count = plan.components, _POINT_DIMENSION, plan.rows + plan.columns
    component_weights = _weight(k, plan.component_polynomial, rng)
    weights = component_weights.values()
    largest = max(weights)
    if largest == 0:
        raise ValueError("kernel mixture component weights have zero mass; draw was not retried")
    scaled = tuple(v / largest for v in weights)
    probabilities = tuple(v / math.fsum(scaled) for v in scaled)
    assignments = tuple(int(i) for i in rng.choice(k, size=count, p=probabilities))
    centers = _gaussian(k, d, rng) if plan.point_law == "gaussian-mixture" else ((0.0,) * d,)
    factors = tuple(_gaussian(d, d, rng) for _ in range(k))
    factor_weights = tuple(_weight(d, plan.polynomial, rng) for _ in range(k))
    if plan.point_law == "gaussian-mixture":
        mean, coefficient = float(rng.normal()), float(rng.normal())
        scale_draws = tuple(float(z) for z in rng.normal(size=k))
    else:
        mean, coefficient, scale_draws = 0.0, 0.0, (0.0,)
    innovations = (
        _gaussian(count, d, rng)
        if plan.base_law == "gaussian"
        else tuple(tuple(float(v) for v in row) for row in rng.uniform(-1, 1, size=(count, d)))
    )
    return KernelPointCloud(
        centers=centers,
        factors=factors,
        factor_weights=factor_weights,
        component_weights=component_weights,
        assignments=assignments,
        innovations=innovations,
        scale_mean=mean,
        scale_coefficient=coefficient,
        scale_draws=scale_draws,
        law=plan.point_law,
        base_law=plan.base_law,
    )
