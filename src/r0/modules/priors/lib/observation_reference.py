"""Causal references retain complete observation source and selection custody."""

import torch

from r0.modules.priors.lib.functional_reference import evaluate_functional_reference
from r0.modules.priors.lib.network_observation_recipe import NetworkObservationRecipe


def evaluate_observation_reference(
    recipe: NetworkObservationRecipe,
    inputs: torch.Tensor,
    *,
    seed: int,
    task_id: int,
    row: int,
    reference: str = "original-row",
) -> torch.Tensor:
    """Clamp all observed state/readout channels; return raw target readouts.

    Hidden roots use the original row population shape. Structural innovations
    use that row's original draws or zero. Source statistics, masks, importance
    and selection remain frozen. This is not conditioning or target adaptation.
    """
    if (
        type(recipe) is not NetworkObservationRecipe
        or type(inputs) is not torch.Tensor
        or inputs.ndim != 2
        or inputs.dtype != torch.float64
        or inputs.layout != torch.strided
        or inputs.device.type not in ("cpu", "cuda")
        or inputs.shape[1] != recipe.task.features
        or not 1 <= inputs.shape[0] <= 20000
        or any(type(v) is not int or not 0 <= v < 2**63 for v in (seed, task_id))
        or type(row) is not int
        or not 0 <= row < recipe.task.rows
        or reference not in ("zero", "original-row")
    ):
        raise ValueError("observation reference requires bounded inputs and exact original row")
    # Charge all nested construction custody and both populations before finite
    # masks, snapshot restoration, latent roots or repeated reference arrays.
    recipe.admit(evaluation_rows=recipe.task.rows + len(inputs))
    if not torch.isfinite(inputs).all():
        raise ValueError("observation reference inputs must be finite")
    recipe.validate()
    return evaluate_functional_reference(
        recipe.runtime, inputs, seed=seed, task_id=task_id, row=row, reference=reference
    )
