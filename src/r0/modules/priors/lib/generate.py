"""One task entrypoint and versioned decoder for all implemented training families."""

from typing import Any, cast

from r0.modules.priors.lib.compositions import (
    Combination,
    CompositeRecipe,
    FunctionComponent,
    generate_composite,
)
from r0.modules.priors.lib.exact_gp import GaussianProcessRecipe
from r0.modules.priors.lib.fourier import FourierRecipe
from r0.modules.priors.lib.grouped_linear import GroupedLinearRecipe
from r0.modules.priors.lib.inputs import InputDistribution
from r0.modules.priors.lib.kernels import (
    GibbsKernel,
    Kernel,
    RationalQuadraticKernel,
    SpectralKernel,
    StationaryKernel,
)
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.polynomial import PolynomialRecipe
from r0.modules.priors.lib.quadratic import QuadraticRecipe
from r0.modules.priors.lib.sampling import BaseRecipe, generate_base_task
from r0.modules.priors.lib.smooth import SmoothRecipe
from r0.modules.priors.lib.training_task import TrainingTask

type Recipe = BaseRecipe | CompositeRecipe


def _kernel_from_dict(data: dict[str, Any]) -> Kernel:
    fields = dict(data)
    for key in (
        "axes",
        "feature_scales",
        "frequencies",
        "centers",
        "transition_scales",
        "length_slopes",
        "amplitude_slopes",
    ):
        if key in fields:
            fields[key] = tuple(fields[key])
    if fields.get("kind") == "gibbs":
        return GibbsKernel(**fields)
    if fields.get("kind") == "rational_quadratic":
        return RationalQuadraticKernel(**fields)
    if fields.get("kind") == "spectral":
        return SpectralKernel(**fields)
    return StationaryKernel(**fields)


def _expression_from_dict(value: Any) -> int | Combination:
    if type(value) is int:
        return value
    fields = dict(value)
    fields["children"] = tuple(_expression_from_dict(child) for child in fields["children"])
    return Combination(**fields)


def recipe_from_dict(data: object) -> Recipe:
    """Decode JSON/TOML recipe fields without mutating the caller's record."""
    if not isinstance(data, dict):
        raise ValueError("recipe must be an object")
    fields = dict(cast(dict[str, Any], data))
    version = fields.get("version", "linear-v1")
    if version == "composite-v1":
        task = recipe_from_dict(fields.get("task", {}))
        if not isinstance(task, LinearRecipe):
            raise ValueError("task controls must use linear-v1 settings")
        fields["task"] = task
        fields["inputs"] = InputDistribution(**fields.get("inputs", {}))
        components: list[FunctionComponent] = []
        for item in fields["components"]:
            component = dict(item)
            child = recipe_from_dict(component.pop("recipe"))
            if isinstance(child, CompositeRecipe):
                raise ValueError("nest combinations in expression, not in component recipes")
            component["recipe"] = child
            if "axes" in component:
                component["axes"] = tuple(component["axes"])
            components.append(FunctionComponent(**component))
        fields["components"] = tuple(components)
        fields["expression"] = _expression_from_dict(fields.get("expression", 0))
        return CompositeRecipe(**fields)
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
    if isinstance(recipe, CompositeRecipe):
        return generate_composite(recipe, seed=seed, task_id=task_id, device=device)
    return generate_base_task(recipe, seed=seed, task_id=task_id, device=device)
