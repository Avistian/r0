"""Exact membership of integer category codes; no ordinal threshold approximation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from r0.modules.priors.lib.causal_activations import CausalActivation

KIND = "category-subset-v1"


def validate_category_subset(parameters: dict[str, float]) -> None:
    count = parameters.get("count")
    if type(count) is not int or not 0 <= count <= 1024:
        raise ValueError("category subset requires 0..1024 codes")
    if len(parameters) != count + 1 or set(parameters) != {"count"} | {
        f"category_{i}" for i in range(count)
    }:
        raise ValueError("category subset fields differ from declared count")
    codes = tuple(parameters[f"category_{i}"] for i in range(count))
    if (
        any(type(code) is not int or not -(2**53) <= code <= 2**53 for code in codes)
        or len(set(codes)) != count
    ):
        raise ValueError("category subset requires distinct exactly representable integer codes")


def evaluate_category_subset(value: torch.Tensor, parameters: dict[str, float]) -> torch.Tensor:
    result = torch.zeros_like(value, dtype=torch.bool)
    for i in range(int(parameters["count"])):
        code = parameters[f"category_{i}"]
        represented = value.new_tensor(code)
        if represented.item() != code:
            raise ValueError("category code is not exactly representable in input dtype")
        result |= value == represented
    return result.to(value.dtype)


def category_subset_activation(categories: tuple[int, ...]) -> CausalActivation:
    from r0.modules.priors.lib.causal_activations import CausalActivation

    if type(categories) is not tuple or len(categories) > 1024:
        raise ValueError("category subset requires a bounded immutable tuple")
    return CausalActivation(
        KIND,
        (("count", len(categories)),)
        + tuple((f"category_{i}", code) for i, code in enumerate(categories)),
    )
