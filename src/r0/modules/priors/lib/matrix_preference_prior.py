"""Named categorical and numerical contracts for shared matrix sampling."""

import math
from dataclasses import dataclass

from r0.modules.priors.lib.categorical_preferences import CategoricalParameter
from r0.modules.priors.lib.parameter_preferences import NumericalParameter

FAMILY_VERSIONS = {
    "gaussian": "gaussian-weights-v1",
    "importance": "importance-weights-v1",
    "factorized": "importance-factorized-weights-v1",
    "kernel": "laplace-kernel-weights-v1",
    "transformed": "transformed-weights-v1",
}


@dataclass(frozen=True, slots=True)
class CausalMatrixPrior:
    mode: str = "meta"
    categorical_policy: str = "shared"
    vector_family: str = "polynomial"
    families: tuple[str, ...] = tuple(FAMILY_VERSIONS)
    simple_families: tuple[str, ...] = tuple(FAMILY_VERSIONS)[:4]
    point_laws: tuple[str, ...] = ("gaussian-mixture", "random-covariance")
    covariance_bases: tuple[str, ...] = ("gaussian", "uniform")
    version: str = "causal-matrix-prior-v1"

    def __post_init__(self) -> None:
        if self.version != "causal-matrix-prior-v1" or self.mode not in ("meta", "local", "global"):
            raise ValueError("unsupported causal matrix prior version or mode")
        if self.categorical_policy not in ("shared", "refresh"):
            raise ValueError("unsupported matrix categorical policy")
        if self.vector_family not in ("polynomial", "uniform"):
            raise ValueError("matrix vectors must be polynomial or uniform")
        for choices, allowed in (
            (self.families, tuple(FAMILY_VERSIONS)),
            (self.simple_families, tuple(FAMILY_VERSIONS)[:4]),
            (self.point_laws, ("gaussian-mixture", "random-covariance")),
            (self.covariance_bases, ("gaussian", "uniform")),
        ):
            if (
                type(choices) is not tuple
                or not choices
                or any(type(c) is not str or c not in allowed for c in choices)
                or len(set(choices)) != len(choices)
            ):
                raise ValueError("matrix choices must be distinct nonempty supported tuples")

    def family_request(self, *, simple: bool) -> CategoricalParameter:
        return CategoricalParameter(
            "simple_random_matrix_type" if simple else "random_matrix_type",
            tuple(FAMILY_VERSIONS[k] for k in (self.simple_families if simple else self.families)),
        )

    def point_request(self) -> CategoricalParameter:
        return CategoricalParameter("kernel_point_law", self.point_laws)

    def base_request(self) -> CategoricalParameter:
        return CategoricalParameter("kernel_covariance_base", self.covariance_bases)


def vector_requests(width: int) -> tuple[NumericalParameter, NumericalParameter]:
    # Scalar calls permit global quantile coupling across different vector widths.
    return (
        NumericalParameter(
            "simple_random_weights_decay_rate",
            0.1 / math.log1p(width),
            6.0,
            scale="log",
            boundary_mass=True,
        ),
        NumericalParameter("simple_random_weights_std_scale", 0.0001, 10.0, scale="log"),
    )


def component_request() -> NumericalParameter:
    return NumericalParameter("kernel_mixture_components", 1, 16, scale="log", integer="floor")


def distance_request() -> NumericalParameter:
    return NumericalParameter("kernel_distance_scale", 0.1, 10.0, scale="log")
