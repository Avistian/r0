"""One task entrypoint and versioned decoder for all implemented training families."""

from typing import Any, cast

from r0.modules.priors.lib.exact_gp import GaussianProcessRecipe, generate_exact_gp
from r0.modules.priors.lib.fourier import FourierRecipe, generate_fourier
from r0.modules.priors.lib.grouped_linear import GroupedLinearRecipe, generate_grouped_linear
from r0.modules.priors.lib.inputs import InputDistribution
from r0.modules.priors.lib.kernels import StationaryKernel
from r0.modules.priors.lib.linear import LinearRecipe, generate_linear
from r0.modules.priors.lib.polynomial import PolynomialRecipe, generate_polynomial
from r0.modules.priors.lib.quadratic import QuadraticRecipe, generate_quadratic
from r0.modules.priors.lib.smooth import SmoothRecipe, generate_smooth
from r0.modules.priors.lib.training_task import TrainingTask

type Recipe = (
    LinearRecipe
    | FourierRecipe
    | PolynomialRecipe
    | QuadraticRecipe
    | GroupedLinearRecipe
    | SmoothRecipe
    | GaussianProcessRecipe
)


def _kernel_from_dict(data: dict[str, Any]) -> StationaryKernel:
    fields = dict(data)
    for key in ("axes", "feature_scales"):
        if key in fields:
            fields[key] = tuple(fields[key])
    return StationaryKernel(**fields)


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
    if version in (
        "fourier-v1",
        "polynomial-v1",
        "quadratic-v1",
        "grouped-linear-v1",
        "smooth-v1",
        "exact-gp-v1",
    ):
        task = recipe_from_dict(fields.get("task", {}))
        if not isinstance(task, LinearRecipe):
            raise ValueError("task controls must use linear-v1 settings")
        fields["task"] = task
        if version in (
            "polynomial-v1",
            "quadratic-v1",
            "grouped-linear-v1",
            "smooth-v1",
            "exact-gp-v1",
        ):
            fields["inputs"] = InputDistribution(**fields.get("inputs", {}))
            if version == "exact-gp-v1":
                if "kernels" in fields:
                    fields["kernels"] = tuple(_kernel_from_dict(k) for k in fields["kernels"])
                return GaussianProcessRecipe(**fields)
            if version == "smooth-v1":
                return SmoothRecipe(**fields)
            if version == "grouped-linear-v1":
                return GroupedLinearRecipe(**fields)
            return (
                PolynomialRecipe(**fields)
                if version == "polynomial-v1"
                else QuadraticRecipe(**fields)
            )
        return FourierRecipe(**fields)
    raise ValueError(f"unsupported recipe version: {version!r}")


def generate_task(recipe: Recipe, *, seed: int, task_id: int, device: str = "cpu") -> TrainingTask:
    """Generate the selected family without substituting another mechanism."""
    if isinstance(recipe, FourierRecipe):
        return generate_fourier(recipe, seed=seed, task_id=task_id, device=device)
    if isinstance(recipe, PolynomialRecipe):
        return generate_polynomial(recipe, seed=seed, task_id=task_id, device=device)
    if isinstance(recipe, QuadraticRecipe):
        return generate_quadratic(recipe, seed=seed, task_id=task_id, device=device)
    if isinstance(recipe, GroupedLinearRecipe):
        return generate_grouped_linear(recipe, seed=seed, task_id=task_id, device=device)
    if isinstance(recipe, SmoothRecipe):
        return generate_smooth(recipe, seed=seed, task_id=task_id, device=device)
    if isinstance(recipe, GaussianProcessRecipe):
        return generate_exact_gp(recipe, seed=seed, task_id=task_id, device=device)
    return generate_linear(recipe, seed=seed, task_id=task_id, device=device)
