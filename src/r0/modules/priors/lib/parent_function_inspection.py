"""Expose distinct parent slices, joint members and the final reduction equation."""

from collections.abc import Callable, Sequence
from dataclasses import asdict

import torch

from r0.modules.priors.lib.feature_conv_inspection import feature_conv_steps
from r0.modules.priors.lib.feature_conv_records import FrozenFeatureConv1D
from r0.modules.priors.lib.feature_conv_sampling_records import SampledFeatureConv1D
from r0.modules.priors.lib.frozen_function_evaluation import evaluate_admitted_function
from r0.modules.priors.lib.frozen_kumo_product import FrozenKumoProductFunction
from r0.modules.priors.lib.frozen_parent_function import FrozenParentFunction
from r0.modules.priors.lib.kumo_product_inspection import kumo_product_steps
from r0.modules.priors.lib.network_inspection import network_steps
from r0.modules.priors.lib.recursive_forest_function import FrozenRecursiveForestFunction
from r0.modules.priors.lib.recursive_function_evaluation import recursive_function_steps
from r0.modules.priors.lib.structural_network import FrozenNetworkFunction


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
        joint_parents = (
            kumo_product_steps(
                function.joint, inputs, prefix=identifier, parents=parents, indices=indices, add=add
            )
            if isinstance(function.joint, FrozenKumoProductFunction)
            else parents
        )
        if isinstance(function.joint, FrozenRecursiveForestFunction):
            joint_parents = recursive_function_steps(
                function.joint, inputs, prefix=identifier, parents=parents, indices=indices, add=add
            )
        if isinstance(function.joint, FrozenNetworkFunction):
            joint_parents = network_steps(
                function.joint,
                inputs,
                prefix=identifier,
                indices=indices,
                add=add,
                input_parents=parents,
            )
        if isinstance(function.joint, (FrozenFeatureConv1D, SampledFeatureConv1D)):
            joint_parents = feature_conv_steps(
                function.joint,
                inputs,
                prefix=identifier,
                parents=parents,
                indices=indices,
                add=add,
            )
        add(
            identifier,
            f"Joint member {node}",
            joint_parents,
            "f_joint(concat(parent states))",
            evaluate_admitted_function(function.joint, inputs)[list(indices)],
            asdict(function.joint),
        )
        members.append(identifier)
    offset = 0
    for index, child in enumerate(function.children):
        identifier = f"parent-member-{node}-{index}"
        stop = offset + child.input_width
        child_parents = (
            kumo_product_steps(
                child,
                inputs[:, offset:stop],
                prefix=identifier,
                parents=[parents[index]],
                indices=indices,
                add=add,
            )
            if isinstance(child, FrozenKumoProductFunction)
            else [parents[index]]
        )
        if isinstance(child, FrozenRecursiveForestFunction):
            child_parents = recursive_function_steps(
                child,
                inputs[:, offset:stop],
                prefix=identifier,
                parents=[parents[index]],
                indices=indices,
                add=add,
            )
        if isinstance(child, FrozenNetworkFunction):
            child_parents = network_steps(
                child,
                inputs[:, offset:stop],
                prefix=identifier,
                indices=indices,
                add=add,
                input_parents=[parents[index]],
            )
        if isinstance(child, (FrozenFeatureConv1D, SampledFeatureConv1D)):
            child_parents = feature_conv_steps(
                child,
                inputs[:, offset:stop],
                prefix=identifier,
                parents=[parents[index]],
                indices=indices,
                add=add,
            )
        add(
            identifier,
            f"Parent {index} transform for node {node}",
            child_parents,
            f"f_parent_{index}(parent state); input slice [{offset}:{stop}]",
            evaluate_admitted_function(child, inputs[:, offset:stop])[list(indices)],
            {"input_start": offset, "input_stop": stop, "function": asdict(child)},
        )
        members.append(identifier)
        offset = stop
    return members
