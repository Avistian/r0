"""Original-row architecture stages, separate from surrounding generation process."""

import json
from collections.abc import Callable, Sequence
from itertools import accumulate
from typing import Any

import torch

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.network_block_function import FrozenNetworkBlockFunction
from r0.modules.priors.lib.network_evaluation import evaluate_network, evaluate_network_block
from r0.modules.priors.lib.structural_network import FrozenNetworkFunction
from r0.modules.priors.lib.training_task import TrainingTask


def network_steps(
    function: FrozenNetworkFunction | FrozenNetworkBlockFunction,
    inputs: torch.Tensor,
    *,
    prefix: str,
    indices: Sequence[int],
    add: Callable[..., None],
    input_parents: list[str] | None = None,
) -> list[str]:
    sliced = isinstance(function, FrozenNetworkBlockFunction)

    def state_parent(source: int) -> str:
        if sliced:
            return f"{prefix}-incoming-{source}"
        return f"{prefix}-input" if source == 0 else f"{prefix}-block-{source}-state"

    def trace(name: str, value: torch.Tensor, parameters: dict[str, Any]) -> None:
        parts = name.split("-")
        index = int(parts[1])
        block = (
            function.block
            if isinstance(function, FrozenNetworkBlockFunction)
            else function.blocks[index - 1]
        )
        if name.endswith("-inputs"):
            parents = [state_parent(p) for p in block.parents]
            formula = "concatenate ordered earlier architecture states"
        elif name.endswith("-state"):
            parents = [
                f"{prefix}-block-{index}-expert-{j}-"
                + ("activation" if e.placement == "after" else "affine")
                for j, e in enumerate(block.experts)
            ]
            if block.gate:
                parents.append(f"{prefix}-block-{index}-weights")
            parents.extend(
                f"{prefix}-block-{index}-skip-{j}-"
                + ("activation" if skip.affine.placement == "after" else "affine")
                for j, skip in enumerate(block.skips)
            )
            formula = "softmax-weighted experts (or one expert) plus projected skips"
        elif name.endswith("-weights"):
            parents = [
                f"{prefix}-block-{index}-gate-"
                + (
                    "activation"
                    if block.gate is not None and block.gate.placement == "after"
                    else "affine"
                )
            ]
            formula = "softmax(gate logits)"
        else:
            branch = name.rsplit("-", 1)[0]
            stage = name.rsplit("-", 1)[1]
            if "-expert-" in name:
                e = block.experts[int(parts[3])]
            elif "-skip-" in name:
                e = block.skips[int(parts[3])].affine
            else:
                e = block.gate
            assert e is not None
            first = "activation" if e.placement == "before" else "affine"
            if stage != first:
                parents = [f"{prefix}-{branch}-{first}"]
            elif "-skip-" in name:
                source = block.skips[int(parts[3])].source
                parents = [state_parent(source)]
            else:
                parents = [f"{prefix}-block-{index}-inputs"]
            formula = (
                "frozen activation"
                if stage == "activation"
                else "(frozen weight * fixed mask * scale) @ input + bias"
            )
        add(f"{prefix}-{name}", name, parents, formula, value[list(indices)], parameters)

    add(
        prefix + "-input",
        "Network input",
        input_parents or [],
        "input state 0",
        inputs[list(indices)],
        {},
    )
    if isinstance(function, FrozenNetworkBlockFunction):
        incoming = [
            inputs[:, end - width : end]
            for width, end in zip(
                function.incoming_widths, accumulate(function.incoming_widths), strict=True
            )
        ]
        for i, value in enumerate(incoming):
            add(
                f"{prefix}-incoming-{i}",
                f"Incoming state {i}",
                [input_parents[i]]
                if input_parents and len(input_parents) == len(incoming)
                else [prefix + "-input"],
                "declared ordered incoming state slice",
                value[list(indices)],
                {"width": function.incoming_widths[i]},
            )
        evaluate_network_block(function.block, incoming, 1, trace=trace)
        return [f"{prefix}-block-1-state"]
    evaluate_network(function, inputs, trace=trace)
    return [f"{prefix}-block-{len(function.blocks)}-state"]


def standalone_network_steps(
    task: TrainingTask,
    indices: Sequence[int],
    add: Callable[..., None],
    *,
    limits: CausalLimits | None,
) -> dict[str, Any]:
    from r0.modules.priors.lib.network_recipe import generate_network, network_recipe_from_dict

    record = json.loads(task.recipe_json)
    recipe = network_recipe_from_dict(record["recipe"], limits=limits or CausalLimits())
    replay = generate_network(recipe, seed=record["seed"], task_id=record["task_id"])
    if not torch.equal(replay.inputs, task.inputs.cpu()) or not torch.equal(
        replay.targets, task.targets.cpu()
    ):
        raise ValueError("saved network task disagrees with replay")
    parents = network_steps(
        recipe.function, replay.inputs, prefix="network", indices=indices, add=add
    )
    add(
        "signal",
        "Network scores",
        parents,
        "frozen standalone network",
        replay.scores[list(indices)],
        {},
    )
    add(
        "targets",
        "Task targets",
        ["signal"],
        "shared target adapter",
        replay.targets[list(indices), None],
        {},
    )
    return {
        "version": "generation-graph-v1",
        "episode_id": task.episode_id,
        "rows": list(indices),
        "component_count": 0,
        "signal_max_error": 0.0,
        "architecture": "standalone network; frozen structural dropout",
    }
