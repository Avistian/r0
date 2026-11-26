"""Admitted importance-weighted projections with independent local parameter draws."""

import math
from dataclasses import replace

import numpy as np

from r0.modules.priors.lib.causal_nodes import (
    ActivatedCausalNode,
    CausalNode,
    MatrixCausalNode,
    StochasticCausalNode,
)
from r0.modules.priors.lib.causal_sampled import SampledCausalRecipe
from r0.modules.priors.lib.weight_matrices import ImportanceWeights
from r0.modules.priors.lib.weight_vectors import PolynomialWeightVector, UniformWeightVector


def sample_importance_profile(
    recipe: SampledCausalRecipe, rng: np.random.Generator
) -> SampledCausalRecipe:
    polynomial = tuple(bool(rng.integers(2)) if n.parents else False for n in recipe.nodes)
    defaults = StochasticCausalNode().structural_parameter_count
    extra = sum(
        node.structural_parameter_count
        + (
            defaults
            + len(node.weights)
            * (len(node.bias) + (2 * len(node.bias) + 2 if poly else len(node.bias)))
            + 1
            if node.parents
            else 0
        )
        for node, poly in zip(recipe.nodes, polynomial, strict=True)
    )
    recipe.limits.admit_dimensions(
        tuple(len(n.bias) for n in recipe.nodes),
        tuple(n.parents for n in recipe.nodes),
        rows=recipe.task.rows,
        calibration_rows=recipe.calibration_rows if recipe.normalize else 0,
        features=recipe.task.features,
        outputs=len(recipe.target),
        extra_parameters=extra,
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
    for node, poly in zip(recipe.nodes, polynomial, strict=True):
        if not node.parents:
            nodes.append(node)
            continue
        assert isinstance(node, ActivatedCausalNode)
        incoming, outgoing = len(node.weights), len(node.bias)
        gaussian = tuple(
            tuple(float(v) for v in row) for row in rng.normal(size=(incoming, outgoing))
        )
        vectors = tuple(
            _polynomial(outgoing, rng)
            if poly
            else UniformWeightVector(tuple(float(v) for v in rng.random(outgoing)))
            for _ in range(incoming)
        )
        record = ImportanceWeights(gaussian=gaussian, importance=vectors)
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
    return replace(recipe, nodes=tuple(nodes), construction="activation-importance-v1")


def _polynomial(width: int, rng: np.random.Generator) -> PolynomialWeightVector:
    low = 0.1 / math.log(1 + width)
    u = min(1.0, max(0.0, 1.2 * float(rng.random()) - 0.1))
    decay = low if u == 0 else 6.0 if u == 1 else math.exp(math.log(low) + u * math.log(6 / low))
    scale = math.exp(math.log(1e-4) + float(rng.random()) * math.log(10 / 1e-4))
    return PolynomialWeightVector(
        draws=tuple(float(v) for v in rng.normal(size=width)),
        decay=decay,
        log_noise_scale=scale,
        permutation=tuple(int(i) for i in rng.permutation(width)),
    )
