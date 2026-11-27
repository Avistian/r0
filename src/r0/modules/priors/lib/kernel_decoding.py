"""Decode the existing explicit Gaussian-process kernel records."""

from typing import Any

from r0.modules.priors.lib.kernels import (
    GibbsKernel,
    Kernel,
    RationalQuadraticKernel,
    SpectralKernel,
    StationaryKernel,
)


def kernel_from_dict(data: dict[str, Any]) -> Kernel:
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
