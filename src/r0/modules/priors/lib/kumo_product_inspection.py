"""Expose two functions reading the same original input before their product."""

from collections.abc import Callable, Sequence
from dataclasses import asdict

import torch

from r0.modules.priors.lib.frozen_function_evaluation import evaluate_admitted_function
from r0.modules.priors.lib.frozen_kumo_product import FrozenKumoProductFunction


def kumo_product_steps(
    function: FrozenKumoProductFunction,
    inputs: torch.Tensor,
    *,
    prefix: str,
    parents: list[str],
    indices: Sequence[int],
    add: Callable[..., None],
) -> list[str]:
    result: list[str] = []
    for name, child in (("left", function.left), ("right", function.right)):
        identifier = f"{prefix}-{name}"
        add(
            identifier,
            f"Kumo product {name} child",
            parents,
            f"f_{name}(same input)",
            evaluate_admitted_function(child, inputs)[list(indices)],
            asdict(child),
        )
        result.append(identifier)
    return result
