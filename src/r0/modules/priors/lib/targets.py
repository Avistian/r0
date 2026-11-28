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
    fixed_noise: torch.Tensor | None = None,
    fixed_uniforms: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, tuple[str, ...]]:
    device = scores.device
    outputs = recipe.classes if recipe.task_type == "multiclass" else 1
    for draw in (fixed_noise, fixed_uniforms):
        if draw is not None and (
            type(draw) is not torch.Tensor
            or draw.shape != (scores.shape[0],)
            or draw.dtype != scores.dtype
            or draw.device != scores.device
            or not torch.isfinite(draw).all()
        ):
            raise ValueError("fixed target draws must align with finite score rows")
    if fixed_uniforms is not None and not torch.all((fixed_uniforms >= 0) & (fixed_uniforms < 1)):
        raise ValueError("fixed label uniforms must lie in [0,1)")
    noise = (
        fixed_noise
        if fixed_noise is not None
        else torch.tensor(noise_rng.normal(size=recipe.rows), dtype=torch.float64, device=device)
    )
    targets = scores[:, 0] + recipe.noise_ratio * noise
    probabilities: torch.Tensor | None = None
    classes: tuple[str, ...] = ()
    if recipe.task_type != "regression":
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
        uniforms = (
            fixed_uniforms
            if fixed_uniforms is not None
            else torch.tensor(
                label_rng.uniform(size=recipe.rows), dtype=torch.float64, device=device
            )
        )
        targets = torch.sum(uniforms[:, None] >= probabilities.cumsum(dim=1)[:, :-1], dim=1)
        classes = tuple(f"class-{i}" for i in range(recipe.classes))
    if not torch.isfinite(targets).all():
        raise ValueError("nonfinite generated targets; task was not retried")
    return targets, probabilities, classes
