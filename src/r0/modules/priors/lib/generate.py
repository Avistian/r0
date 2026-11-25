"""One task entrypoint and versioned decoder for all implemented training families."""

from typing import Any, cast

from r0.modules.priors.lib.fourier import FourierRecipe, generate_fourier
from r0.modules.priors.lib.inputs import InputDistribution
from r0.modules.priors.lib.linear import LinearRecipe, generate_linear
from r0.modules.priors.lib.polynomial import PolynomialRecipe, generate_polynomial
from r0.modules.priors.lib.training_task import TrainingTask

type Recipe = LinearRecipe | FourierRecipe | PolynomialRecipe


def recipe_from_dict(data: object) -> Recipe:
    """Decode JSON/TOML recipe fields without mutating the caller's record."""
    if not isinstance(data, dict):
        raise ValueError("recipe must be an object")
    fields = dict(cast(dict[str, Any], data))
    version = fields.get("version", "linear-v1")
    if version == "linear-v1":
        if "class_offsets" in fields:
            fields["class_offsets"] = tuple(fields["class_offsets"])
        return LinearRecipe(**fields)
    if version in ("fourier-v1", "polynomial-v1"):
        task = recipe_from_dict(fields.get("task", {}))
        if not isinstance(task, LinearRecipe):
            raise ValueError("task controls must use linear-v1 settings")
        fields["task"] = task
        if version == "polynomial-v1":
            fields["inputs"] = InputDistribution(**fields.get("inputs", {}))
            return PolynomialRecipe(**fields)
        return FourierRecipe(**fields)
    raise ValueError(f"unsupported recipe version: {version!r}")


def generate_task(recipe: Recipe, *, seed: int, task_id: int, device: str = "cpu") -> TrainingTask:
    """Generate the selected family without substituting another mechanism."""
    if isinstance(recipe, FourierRecipe):
        return generate_fourier(recipe, seed=seed, task_id=task_id, device=device)
    if isinstance(recipe, PolynomialRecipe):
        return generate_polynomial(recipe, seed=seed, task_id=task_id, device=device)
    return generate_linear(recipe, seed=seed, task_id=task_id, device=device)
