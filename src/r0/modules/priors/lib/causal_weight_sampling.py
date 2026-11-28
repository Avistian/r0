"""Admitted Gaussian-factor projections for a versioned causal exploration profile."""

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
from r0.modules.priors.lib.weight_matrices import FactorizedWeights


def sample_factorized_profile(
    recipe: SampledCausalRecipe, rng: np.random.Generator
) -> SampledCausalRecipe:
    widths = tuple(len(node.bias) for node in recipe.nodes)
    ranks = tuple(
        int(rng.integers(1, min(len(node.weights), len(node.bias)) + 1)) if node.parents else 0
        for node in recipe.nodes
    )
    default_noise_parameters = StochasticCausalNode().structural_parameter_count
    extra = sum(
        node.structural_parameter_count
        + (
            default_noise_parameters + rank * (len(node.weights) + len(node.bias) + 1)
            if rank
            else 0
        )
        for node, rank in zip(recipe.nodes, ranks, strict=True)
    )
    # Admit all retained factor scalars before allocating any factor arrays.
    recipe.limits.admit_dimensions(
        widths,
        tuple(node.parents for node in recipe.nodes),
        rows=recipe.task.rows,
        calibration_rows=recipe.calibration_rows if recipe.normalize else 0,
        features=recipe.task.features,
        outputs=len(recipe.target),
        extra_parameters=extra,
        extra_working_channels=max(
            (
                StochasticCausalNode.working_channels_for_width(width)
                for width, rank in zip(widths, ranks, strict=True)
                if rank
            ),
            default=0,
        ),
    )
    nodes: list[CausalNode] = []
    for node, rank in zip(recipe.nodes, ranks, strict=True):
        if not rank:
            nodes.append(node)
            continue
        assert isinstance(node, ActivatedCausalNode)
        incoming, outgoing = len(node.weights), len(node.bias)
        record = FactorizedWeights(
            left=tuple(tuple(float(v) for v in row) for row in rng.normal(size=(incoming, rank))),
            right=tuple(tuple(float(v) for v in row) for row in rng.normal(size=(rank, outgoing))),
            diagonal=(1 / math.sqrt(incoming * rank),) * rank,
        )
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
    return replace(recipe, nodes=tuple(nodes), construction="activation-factorized-v1")
