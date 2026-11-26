"""An explicit r0 exploration law for exogenous distributions and noise responses."""

import math
from dataclasses import replace

import numpy as np

from r0.modules.priors.lib.causal_nodes import ActivatedCausalNode, StochasticCausalNode
from r0.modules.priors.lib.causal_sampled import SampledCausalRecipe
from r0.modules.priors.lib.exogenous import ExogenousDistribution, NoiseResponse
from r0.modules.priors.lib.inputs import InputDistribution


def sample_exogenous_profile(
    recipe: SampledCausalRecipe, rng: np.random.Generator
) -> SampledCausalRecipe:
    nodes: list[StochasticCausalNode] = []
    for node in recipe.nodes:
        assert isinstance(node, ActivatedCausalNode)
        width = len(node.bias)
        geometry = "ball" if not node.parents and rng.random() < 0.2 else "independent"
        inputs = InputDistribution()
        if geometry == "independent":
            family = str(rng.choice(("gaussian", "uniform", "student_t", "mixture")))
            inputs = InputDistribution(
                distribution=family,
                location=float(rng.uniform(-0.5, 0.5)),
                scale=float(math.exp(rng.uniform(math.log(0.25), math.log(2)))),
                df=float(rng.choice((0.75, 1, 1.5, 2, 3, 5, 10))) if family == "student_t" else 5,
                mixture_gap=float(math.exp(rng.uniform(math.log(0.25), math.log(4))))
                if family == "mixture"
                else 2,
                bound=float(rng.choice((2, 4, 8))) if rng.random() < 0.25 else None,
            )
        factor = ()
        if rng.random() < 0.5:
            matrix = rng.normal(size=(width, width)) / math.sqrt(width)
            factor = tuple(tuple(float(v) for v in row) for row in matrix)
        response = NoiseResponse(
            strength=float(rng.choice((0, 0.1, 0.5, 1))) if node.parents else 0,
            power=float(rng.choice((0.5, 1, 2))) if node.parents else 1,
        )
        nodes.append(
            StochasticCausalNode(
                parents=node.parents,
                weights=node.weights,
                bias=node.bias,
                noise_scale=node.noise_scale,
                input_activation=node.input_activation,
                output_activation=node.output_activation,
                noise_position=node.noise_position,
                exogenous=ExogenousDistribution(inputs=inputs, factor=factor, geometry=geometry),
                noise_response=response,
            )
        )
    return replace(recipe, nodes=tuple(nodes), construction="activation-exogenous-v1")
