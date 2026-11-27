"""Expose distinct parent slices, joint members and the final reduction equation."""

from collections.abc import Callable, Sequence
from dataclasses import asdict

import torch

from r0.modules.priors.lib.frozen_function_evaluation import evaluate_admitted_function
from r0.modules.priors.lib.frozen_parent_function import FrozenParentFunction


def parent_function_steps(
    function: FrozenParentFunction,
    inputs: torch.Tensor,
    *,
    node: int,
    parents: list[str],
    indices: Sequence[int],
    add: Callable[..., None],
) -> list[str]:
    members: list[str] = []
    if function.joint:
        identifier = f"parent-joint-{node}"
        add(
            identifier,
            f"Joint member {node}",
            parents,
            "f_joint(concat(parent states))",
            evaluate_admitted_function(function.joint, inputs)[list(indices)],
            asdict(function.joint),
        )
        members.append(identifier)
    offset = 0
    for index, child in enumerate(function.children):
        identifier = f"parent-member-{node}-{index}"
        stop = offset + child.input_width
        add(
            identifier,
            f"Parent {index} transform for node {node}",
            [parents[index]],
            f"f_parent_{index}(parent state); input slice [{offset}:{stop}]",
            evaluate_admitted_function(child, inputs[:, offset:stop])[list(indices)],
            {"input_start": offset, "input_stop": stop, "function": asdict(child)},
        )
        members.append(identifier)
        offset = stop
    return members
