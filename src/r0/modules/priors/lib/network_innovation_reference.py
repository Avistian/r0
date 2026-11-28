"""Explicit zero-innovation or fixed-innovation references, never conditional means."""

from collections.abc import Mapping

import torch

from r0.modules.priors.lib.frozen_function_evaluation import evaluate_admitted_function
from r0.modules.priors.lib.network_innovation_recipe import NetworkInnovationRecipe
from r0.modules.priors.lib.signal_noise import signal_noise_scale


def evaluate_network_innovation_reference(
    recipe: NetworkInnovationRecipe,
    inputs: torch.Tensor,
    *,
    innovations: Mapping[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Raw latent output on observed inputs, excluding root and target adapters.

    None chooses exactly zero standardized innovations (means remain in equations).
    A mapping supplies every noise-bearing compiled node's standardized draws,
    after the exogenous distribution/factor transformation. It is a fixed-draw
    response, not a probability, marginal mean, or draw from the configured law.
    """
    if (
        type(recipe) is not NetworkInnovationRecipe
        or type(inputs) is not torch.Tensor
        or inputs.ndim != 2
        or inputs.shape[1] != recipe.task.features
        or not inputs.is_floating_point()
    ):
        raise ValueError("network reference requires finite observed input rows and recipe")
    # Shape/dtype checks above are metadata-only. Even the finite check creates
    # an O(rows * channels) boolean tensor, so admit the full query first.
    recipe.admit(evaluation_rows=inputs.shape[0])
    if not torch.isfinite(inputs).all():
        raise ValueError("network reference requires finite observed input rows and recipe")
    assert recipe.plan is not None
    noise_nodes: set[int] = set()
    index = 1
    for stage in recipe.plan.stages:
        index += 1
        if stage is not None:
            noise_nodes.add(index)
            index += 1 + int(stage.placement == "pre-activation")
    if innovations is not None:
        if set(innovations) != noise_nodes or any(type(i) is not int for i in innovations):
            raise ValueError(
                "fixed innovation reference requires exactly the compiled noise node keys"
            )
        for i, value in innovations.items():
            if (
                type(value) is not torch.Tensor
                or value.shape != (len(inputs), recipe.nodes[i].function.output_width)
                or value.dtype != inputs.dtype
                or value.device != inputs.device
                or not torch.isfinite(value).all()
            ):
                raise ValueError(
                    "fixed innovations must align with reference rows/channels/device/dtype"
                )
    values = [inputs]
    for index, node in enumerate(recipe.nodes[1:], 1):
        argument = torch.cat([values[p] for p in node.parents], dim=1)
        output = evaluate_admitted_function(node.function, argument)
        if innovations is not None and index in noise_nodes:
            output = (
                output
                + signal_noise_scale(output, node.noise_scale, node.noise_response)
                * innovations[index]
            )
        if not torch.isfinite(output).all():
            raise ValueError("nonfinite stochastic network reference; not retried")
        values.append(output)
    return torch.stack([values[i][:, j] for i, j in recipe.target], dim=1)


def network_innovation_row(
    recipe: NetworkInnovationRecipe, *, seed: int, task_id: int, row: int
) -> dict[int, torch.Tensor]:
    """Regenerate every configured noise node's original standardized row draw.

    Returned tensors have shape (1, channels), including channels beyond trace
    previews. No statistics or architecture resampling occurs. The observed root
    is not redrawn; existing keyed innovation streams define these values.
    """
    from r0.modules.priors.lib.exogenous import sample_exogenous
    from r0.modules.priors.lib.functional_causal_evaluation import functional_rng

    if (
        type(recipe) is not NetworkInnovationRecipe
        or any(type(v) is not int or not 0 <= v < 2**63 for v in (seed, task_id))
        or type(row) is not int
        or not 0 <= row < recipe.task.rows
    ):
        raise ValueError("original innovation row requires exact bounded seed/task/row identities")
    recipe.admit()
    assert recipe.plan is not None
    result: dict[int, torch.Tensor] = {}
    node = 1
    for stage in recipe.plan.stages:
        node += 1
        if stage is not None:
            equation = recipe.nodes[node]
            values = sample_exogenous(
                equation.exogenous,
                recipe.task.rows,
                equation.function.output_width,
                lambda *keys: functional_rng(seed, task_id, *keys),
                node,
            )
            result[node] = torch.tensor(values[row : row + 1], dtype=torch.float64)
            node += 1 + int(stage.placement == "pre-activation")
    return result
