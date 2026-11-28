"""Retain effective parent-member equations after surgical function replacement."""

from collections.abc import Callable, Sequence

import torch

from r0.modules.priors.lib.frozen_parent_function import FrozenParentFunction
from r0.modules.priors.lib.parent_function_inspection import parent_function_steps


def intervention_parent_steps(
    function: FrozenParentFunction,
    inputs: torch.Tensor,
    *,
    node: int,
    indices: Sequence[int],
    add: Callable[..., None],
) -> list[str]:
    parents: list[str] = []
    start = 0
    for member, child in enumerate(function.children):
        stop = start + child.input_width
        identifier = f"intervention-{node}-parent-input-{member}"
        add(
            identifier,
            f"Effective parent member {member} input",
            [f"intervention-{node}-inputs"],
            f"ordered effective equation input slice [{start}:{stop}]",
            inputs[list(indices), start:stop],
            {"input_start": start, "input_stop": stop},
        )
        parents.append(identifier)
        start = stop
    return parent_function_steps(
        function, inputs, node=node, parents=parents, indices=indices, add=add
    )
