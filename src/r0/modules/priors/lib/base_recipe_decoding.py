"""Decode the original standalone task families with their existing field contracts."""

from typing import Any

from r0.modules.priors.lib.exact_gp import GaussianProcessRecipe
from r0.modules.priors.lib.fourier import FourierRecipe
from r0.modules.priors.lib.grouped_linear import GroupedLinearRecipe
from r0.modules.priors.lib.inputs import InputDistribution
from r0.modules.priors.lib.kernel_decoding import kernel_from_dict
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.polynomial import PolynomialRecipe
from r0.modules.priors.lib.quadratic import QuadraticRecipe
from r0.modules.priors.lib.sampling import BaseRecipe
from r0.modules.priors.lib.smooth import SmoothRecipe


def base_recipe_from_dict(fields: dict[str, Any]) -> BaseRecipe:
    from r0.modules.priors.lib.generate import recipe_from_dict

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
                    fields["kernels"] = tuple(kernel_from_dict(k) for k in fields["kernels"])
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
