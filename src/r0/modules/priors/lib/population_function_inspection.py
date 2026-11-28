"""Detailed frozen mechanism stages inside an original-row population DAG trace."""

from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any

import torch

from r0.modules.priors.lib.feature_conv_inspection import feature_conv_steps
from r0.modules.priors.lib.feature_conv_records import FrozenFeatureConv1D
from r0.modules.priors.lib.feature_conv_sampling_records import SampledFeatureConv1D
from r0.modules.priors.lib.frozen_function_composites import (
    FrozenMLPFunction,
    FrozenProductFunction,
)
from r0.modules.priors.lib.frozen_function_evaluation import (
    FrozenFunction,
    evaluate_admitted_function,
)
from r0.modules.priors.lib.frozen_kumo_product import FrozenKumoProductFunction
from r0.modules.priors.lib.frozen_parent_function import FrozenParentFunction
from r0.modules.priors.lib.network_block_function import FrozenNetworkBlockFunction
from r0.modules.priors.lib.network_inspection import network_steps
from r0.modules.priors.lib.recursive_forest_function import FrozenRecursiveForestFunction
from r0.modules.priors.lib.recursive_function_evaluation import recursive_function_steps
from r0.modules.priors.lib.structural_network import FrozenNetworkFunction


def population_function_steps(
    function: FrozenFunction,
    inputs: torch.Tensor,
    *,
    prefix: str,
    parents: list[str],
    indices: Sequence[int],
    add: Callable[..., None],
) -> list[str]:
    def scoped(
        key: str,
        label: str,
        dependencies: list[str],
        formula: str,
        values: torch.Tensor,
        parameters: dict[str, Any],
    ) -> None:
        metadata = (
            parameters
            if "scope" in parameters
            else {**parameters, "scope": "original dataset rows", "original_rows": list(indices)}
        )
        add(key, label, dependencies, formula, values, metadata)

    if isinstance(function, (FrozenFeatureConv1D, SampledFeatureConv1D)):
        return feature_conv_steps(
            function, inputs, prefix=prefix, parents=parents, indices=indices, add=scoped
        )
    if isinstance(function, (FrozenNetworkFunction, FrozenNetworkBlockFunction)):
        return network_steps(
            function, inputs, prefix=prefix, indices=indices, add=scoped, input_parents=parents
        )
    if isinstance(function, FrozenRecursiveForestFunction):
        return recursive_function_steps(
            function, inputs, prefix=prefix, parents=parents, indices=indices, add=scoped
        )
    if isinstance(function, FrozenMLPFunction):
        value, activation_index, dependencies = inputs, 0, parents

        def activate() -> None:
            nonlocal value, activation_index, dependencies
            activation = function.activations[activation_index]
            value = activation.apply(value)
            key = f"{prefix}-activation-{activation_index}"
            scoped(
                key,
                "Frozen calibrated activation",
                dependencies,
                "apply retained calibration constants and activation; no refit",
                value[list(indices)],
                asdict(activation),
            )
            dependencies = [key]
            activation_index += 1

        if function.start_activation:
            activate()
        for index, linear in enumerate(function.linears):
            value = evaluate_admitted_function(linear, value)
            key = f"{prefix}-linear-{index}"
            scoped(
                key,
                "MLP affine",
                dependencies,
                "frozen affine transform",
                value[list(indices)],
                asdict(linear),
            )
            dependencies = [key]
            if index < len(function.linears) - 1 or function.end_activation:
                activate()
        return dependencies
    if isinstance(function, (FrozenKumoProductFunction, FrozenProductFunction)):
        members: list[str] = []
        for name, child in (("left", function.left), ("right", function.right)):
            identifier = f"{prefix}-{name}"
            dependencies = population_function_steps(
                child, inputs, prefix=identifier, parents=parents, indices=indices, add=scoped
            )
            scoped(
                identifier,
                f"Product {name} member",
                dependencies,
                f"f_{name}(same original input)",
                evaluate_admitted_function(child, inputs)[list(indices)],
                asdict(child),
            )
            members.append(identifier)
        identifier = f"{prefix}-product"
        scoped(
            identifier,
            "Elementwise product",
            members,
            "left * right",
            evaluate_admitted_function(function, inputs)[list(indices)],
            asdict(function),
        )
        return [identifier]
    if isinstance(function, FrozenParentFunction):
        members = []
        member_values: list[torch.Tensor] = []
        if function.joint is not None:
            identifier = f"{prefix}-joint"
            dependencies = population_function_steps(
                function.joint,
                inputs,
                prefix=identifier,
                parents=parents,
                indices=indices,
                add=scoped,
            )
            scoped(
                identifier,
                "Joint parent member",
                dependencies,
                "f_joint(all original parent states)",
                evaluate_admitted_function(function.joint, inputs)[list(indices)],
                asdict(function.joint),
            )
            members.append(identifier)
            member_values.append(evaluate_admitted_function(function.joint, inputs))
        offset = 0
        for index, child in enumerate(function.children):
            stop = offset + child.input_width
            identifier = f"{prefix}-parent-{index}"
            sliced = inputs[:, offset:stop]
            slice_id = f"{identifier}-parent-slice"
            scoped(
                slice_id,
                f"Parent {index} input slice",
                parents,
                "select the declared consecutive parent input channels",
                sliced[list(indices)],
                {"input_start": offset, "input_stop": stop},
            )
            dependencies = population_function_steps(
                child, sliced, prefix=identifier, parents=[slice_id], indices=indices, add=scoped
            )
            scoped(
                identifier,
                f"Parent {index} member",
                dependencies,
                "evaluate this parent slice",
                evaluate_admitted_function(child, sliced)[list(indices)],
                {"input_start": offset, "input_stop": stop, "function": asdict(child)},
            )
            members.append(identifier)
            member_values.append(evaluate_admitted_function(child, sliced))
            offset = stop
        if function.reducer is not None:
            combined = torch.cat(member_values, dim=1)
            key = f"{prefix}-reducer-input"
            scoped(
                key,
                "Ordered reducer members",
                members,
                "concatenate joint first, then parents",
                combined[list(indices)],
                {},
            )
            members = population_function_steps(
                function.reducer,
                combined,
                prefix=f"{prefix}-reducer",
                parents=[key],
                indices=indices,
                add=scoped,
            )
        identifier = f"{prefix}-reduction"
        scoped(
            identifier,
            "Parent member reduction",
            members,
            f"{function.aggregation}(joint first, then ordered parent members)",
            evaluate_admitted_function(function, inputs)[list(indices)],
            {
                "aggregation": function.aggregation,
                "weights": function.weights,
                "reducer": asdict(function.reducer) if function.reducer else None,
            },
        )
        return [identifier]
    return parents
