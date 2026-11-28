"""Bounded r0 priors for ensemble size, member selection and score weights."""

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np


@dataclass(frozen=True, slots=True)
class TreeEnsemblePrior:
    min_members: int = 1
    max_members: int = 4
    count_probabilities: tuple[float, ...] = ()
    member_probabilities: tuple[float, ...] = ()
    replacement: bool = True
    weight_law: str = "equal"
    concentration: float = 1.0
    scale: float = 1.0
    version: str = "r0-tree-ensemble-prior-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "r0-tree-ensemble-prior-v1"
            or type(self.min_members) is not int
            or type(self.max_members) is not int
            or not 1 <= self.min_members <= self.max_members <= 128
            or type(self.replacement) is not bool
            or self.weight_law not in ("equal", "dirichlet", "lognormal", "signed-normal")
            or type(self.concentration) not in (int, float)
            or not math.isfinite(self.concentration)
            or not 0.05 <= self.concentration <= 100
            or type(self.scale) not in (int, float)
            or not math.isfinite(self.scale)
            or not 0 < self.scale <= 4
        ):
            raise ValueError("invalid bounded ensemble prior")
        for values in (self.count_probabilities, self.member_probabilities):
            if (
                type(values) is not tuple
                or len(values) > 128
                or any(
                    type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
                    for v in values
                )
            ):
                raise ValueError("ensemble probabilities must be bounded finite tuples")
            if values and not math.isclose(math.fsum(values), 1, rel_tol=0, abs_tol=1e-12):
                raise ValueError("ensemble probabilities must sum to one")
        if self.count_probabilities and len(self.count_probabilities) != (
            self.max_members - self.min_members + 1
        ):
            raise ValueError("count probabilities must cover every declared count")

    def validate_pool(self, count: int, reduction: str) -> None:
        if self.member_probabilities and len(self.member_probabilities) != count:
            raise ValueError("member probabilities must match the template pool")
        support = (
            sum(p > 0 for p in self.member_probabilities) if self.member_probabilities else count
        )
        if not self.replacement and self.max_members > support:
            raise ValueError("without replacement requires enough positive-probability templates")
        if self.weight_law == "signed-normal" and reduction != "sum":
            raise ValueError("signed-normal weights require sum reduction")


def ensemble_prior_from_dict(value: Any) -> TreeEnsemblePrior:
    if type(value) is not dict:
        raise ValueError("ensemble prior must be an object")
    data = cast(dict[str, Any], value).copy()
    if set(data) != {
        "min_members",
        "max_members",
        "count_probabilities",
        "member_probabilities",
        "replacement",
        "weight_law",
        "concentration",
        "scale",
        "version",
    }:
        raise ValueError("ensemble prior fields differ")
    for key in ("count_probabilities", "member_probabilities"):
        if type(data[key]) is not list:
            raise ValueError("ensemble prior probabilities must be arrays")
        data[key] = tuple(cast(list[Any], data[key]))
    return TreeEnsemblePrior(**data)


def sample_ensemble_prior(
    prior: TreeEnsemblePrior, pool_size: int, *, construction_seed: int, seed: int, task_id: int
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([0x45505249, construction_seed, seed, task_id]))
    )
    count = int(
        rng.choice(
            np.arange(prior.min_members, prior.max_members + 1), p=prior.count_probabilities or None
        )
    )
    indices = tuple(
        int(i)
        for i in rng.choice(
            pool_size, size=count, replace=prior.replacement, p=prior.member_probabilities or None
        )
    )
    if prior.weight_law == "equal":
        weights = (1.0,) * count
    elif prior.weight_law == "dirichlet":
        weights = tuple(float(v) for v in rng.dirichlet(np.full(count, prior.concentration)))
    elif prior.weight_law == "lognormal":
        weights = tuple(float(v) for v in rng.lognormal(mean=0, sigma=prior.scale, size=count))
    else:
        weights = tuple(float(v) for v in rng.normal(loc=0, scale=prior.scale, size=count))
    return indices, weights
