"""Shared regression/classification adapters for synthetic mechanisms."""

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from r0.modules.priors.lib.linear import LinearRecipe


def adapt_targets(
    recipe: "LinearRecipe",
    scores: torch.Tensor,
    *,
    noise_rng: np.random.Generator,
    label_rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor | None, tuple[str, ...]]:
    device = scores.device
    noise = torch.tensor(noise_rng.normal(size=recipe.rows), dtype=torch.float64, device=device)
    targets = scores[:, 0] + recipe.noise_ratio * noise
    probabilities: torch.Tensor | None = None
    classes: tuple[str, ...] = ()
    if recipe.task_type != "regression":
        probabilities = classification_probabilities(recipe, scores)
        uniforms = torch.tensor(
            label_rng.uniform(size=recipe.rows), dtype=torch.float64, device=device
        )
        targets = torch.sum(uniforms[:, None] >= probabilities.cumsum(dim=1)[:, :-1], dim=1)
        classes = tuple(f"class-{i}" for i in range(recipe.classes))
    if not torch.isfinite(targets).all():
        raise ValueError("nonfinite generated targets; task was not retried")
    return targets, probabilities, classes


def classification_probabilities(recipe: "LinearRecipe", scores: torch.Tensor) -> torch.Tensor:
    """Shared deterministic class adapter; no label or observation-noise draws."""
    device = scores.device
    outputs = recipe.classes if recipe.task_type == "multiclass" else 1
    offsets = torch.tensor(
        recipe.class_offsets or (0.0,) * outputs, dtype=torch.float64, device=device
    )
    logits = recipe.separation * scores + offsets
    if not torch.isfinite(logits).all():
        raise ValueError("nonfinite class scores; task was not retried")
    if recipe.task_type == "binary":
        positive = torch.sigmoid(logits[:, 0])
        probabilities = torch.stack((1 - positive, positive), dim=1)
    else:
        probabilities = torch.softmax(logits, dim=1)
    return probabilities
