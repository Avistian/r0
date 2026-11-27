"""Explicit state/readout clamps with fixed latent roots and structural innovations."""

from typing import cast

import torch

from r0.modules.priors.lib.exogenous import sample_exogenous
from r0.modules.priors.lib.frozen_function_evaluation import evaluate_admitted_function
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_causal_evaluation import functional_rng
from r0.modules.priors.lib.node_pipeline import PipelineEvaluation
from r0.modules.priors.lib.signal_noise import signal_noise_scale
from r0.modules.priors.lib.source_root_base import functional_root_values


def evaluate_functional_reference(
    recipe: FunctionalCausalRecipe,
    inputs: torch.Tensor,
    *,
    seed: int,
    task_id: int,
    row: int,
    reference: str = "original-row",
) -> torch.Tensor:
    """Return raw target readouts under simultaneous observed-channel clamps.

    Every observed channel is set at both state and readout, after its frozen
    equation/pipeline. Hidden root draws stay at the selected original row.
    Structural draws are fixed at that row or explicitly zeroed. This is a causal
    reference, not a conditional mean, normalized task score or enclosing task probability.
    No target adapter, backend fitting or independent calibration is run.
    """
    if (
        not isinstance(cast(object, recipe), FunctionalCausalRecipe)
        or type(inputs) is not torch.Tensor
        or inputs.ndim != 2
        or inputs.dtype != torch.float64
        or inputs.device.type not in ("cpu", "cuda")
        or inputs.shape[1] != recipe.task.features
        or not 1 <= inputs.shape[0] <= 20000
        or any(type(v) is not int or not 0 <= v < 2**63 for v in (seed, task_id))
        or type(row) is not int
        or not 0 <= row < recipe.task.rows
        or reference not in ("zero", "original-row")
    ):
        raise ValueError("functional reference requires bounded inputs and exact row identities")
    if getattr(recipe, "population", None) == "support-only":
        raise ValueError("support-population references require partition-specific latent replay")
    # Include the retained original population and the entire query before finite masks,
    # root sampling, repeated latent rows or function evaluation allocations.
    recipe.admit(evaluation_rows=recipe.task.rows + len(inputs))
    if not torch.isfinite(inputs).all():
        raise ValueError("functional reference inputs must be finite")
    validate = getattr(recipe, "validate_custody", None)
    if validate is not None:
        validate()
    states: list[torch.Tensor] = []
    readouts: list[torch.Tensor] = []

    def rng(*keys: object):
        return functional_rng(seed, task_id, *keys)

    for index, node in enumerate(recipe.nodes):
        if node.root is not None:
            root = functional_root_values(
                node.root, recipe.task.rows, rng("root-base", index), limits=recipe.limits
            )
            argument = torch.tensor(root[row : row + 1], dtype=inputs.dtype, device=inputs.device)
            argument = argument.expand(len(inputs), -1)
        else:
            argument = torch.cat([states[parent] for parent in node.parents], dim=1)
        output = evaluate_admitted_function(node.function, argument)
        pipeline = (
            PipelineEvaluation(node.pipeline, index, None) if node.pipeline is not None else None
        )
        state = pipeline.before_noise(output) if pipeline else output
        if reference == "original-row":
            raw = sample_exogenous(
                node.exogenous, recipe.task.rows, node.function.output_width, rng, index
            )
            innovation = torch.tensor(raw[row : row + 1], dtype=inputs.dtype, device=inputs.device)
            state = (
                state
                + signal_noise_scale(
                    output,
                    node.noise_scale or (0.0,) * node.function.output_width,
                    node.noise_response,
                )
                * innovation
            )
        state, readout = pipeline.finish(state) if pipeline else (state, state)
        clamps = [
            (column, channel) for column, (i, channel) in enumerate(recipe.observed) if i == index
        ]
        if clamps:
            state, readout = state.clone(), readout.clone()
            for column, channel in clamps:
                state[:, channel] = inputs[:, column]
                readout[:, channel] = inputs[:, column]
        if not torch.isfinite(state).all() or not torch.isfinite(readout).all():
            raise ValueError("nonfinite functional reference; no retry or repair")
        states.append(state)
        readouts.append(readout)
    return torch.stack([readouts[i][:, j] for i, j in recipe.target], dim=1)
