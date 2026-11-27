"""Original-row intervention locations, bypasses, innovations and propagated states."""

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any

import torch

from r0.modules.priors.lib.causal_intervention_decoding import intervention_from_dict
from r0.modules.priors.lib.causal_intervention_execution import intervention_execution
from r0.modules.priors.lib.causal_intervention_generation import generate_intervened
from r0.modules.priors.lib.causal_intervention_records import causal_runtime
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.converted_causal_records import ConvertedCausalNode
from r0.modules.priors.lib.converter_intervention_execution import ConverterInterventionExecution
from r0.modules.priors.lib.converter_intervention_inspection import (
    converted_trace_order,
    converted_trace_parents,
)
from r0.modules.priors.lib.native_causal import NativeCausalNode
from r0.modules.priors.lib.native_vector_records import NativeVectorNode
from r0.modules.priors.lib.network_inspection import network_steps
from r0.modules.priors.lib.structural_network import FrozenNetworkFunction
from r0.modules.priors.lib.training_task import TrainingTask


def intervention_steps(
    task: TrainingTask,
    indices: Sequence[int],
    add: Callable[..., None],
    *,
    limits: CausalLimits | None,
) -> dict[str, Any]:
    record = json.loads(task.recipe_json)
    recipe = intervention_from_dict(record["recipe"], limits=limits or CausalLimits())
    execution = intervention_execution(recipe)
    replay = generate_intervened(
        recipe, seed=record["seed"], task_id=record["task_id"], execution=execution
    )
    if not torch.equal(replay.inputs, task.inputs.cpu()) or not torch.equal(
        replay.targets, task.targets.cpu()
    ):
        raise ValueError("saved intervention task differs from original-row replay")
    runtime = causal_runtime(recipe.base)
    selected = list(indices)
    for i, node in enumerate(runtime.nodes):
        equation = (
            node.equation
            if isinstance(node, (NativeCausalNode, NativeVectorNode, ConvertedCausalNode))
            else node
        )
        effective = recipe.effective_runtime().nodes[i]
        effective_equation = (
            effective.equation
            if isinstance(effective, (NativeCausalNode, NativeVectorNode, ConvertedCausalNode))
            else effective
        )
        if (
            isinstance(effective_equation.function, FrozenNetworkFunction)
            and "inputs" in execution.traces[i]
        ):
            network_steps(
                effective_equation.function,
                execution.traces[i]["inputs"],
                prefix=f"network-{i}",
                indices=indices,
                add=add,
                input_parents=[f"intervention-{p}-state" for p in effective_equation.parents]
                if effective_equation.parents
                else [f"intervention-{i}-root"],
            )
        changes = [asdict(c) for c in recipe.interventions if c.node == i]
        items = execution.traces[i].items()
        if isinstance(execution, ConverterInterventionExecution):
            items = sorted(items, key=lambda item: converted_trace_order(item[0]))
        for role, value in items:
            matrix = value[:, None] if value.ndim == 1 else value
            if isinstance(execution, ConverterInterventionExecution):
                parents = converted_trace_parents(execution, i, role)
            elif (
                (
                    i in execution.bypassed
                    or role.startswith(("uniform:", "forced-"))
                    or role
                    in (
                        "root",
                        "exogenous",
                        "uniform",
                        "forced_probability",
                    )
                )
                or isinstance(node, ConvertedCausalNode)
                and "function" not in execution.traces[i]
            ):
                parents = []
            elif role == "inputs":
                parents = [f"intervention-{p}-state" for p in equation.parents]
            elif role == "function":
                parents = [f"intervention-{i}-inputs"]
            elif role == "scale":
                parents = [f"intervention-{i}-function"]
            else:
                parents = [f"intervention-{i}-function", f"intervention-{i}-exogenous"]
            add(
                f"intervention-{i}-{role}",
                f"Node {i}: {role}",
                parents,
                "surgical replacement; original equation bypassed"
                if i in execution.bypassed
                else "equation and preprocessing bypassed; innovation and importance retained"
                if isinstance(execution, ConverterInterventionExecution)
                and i in execution.stage_bypassed
                else "frozen equation evaluated with declared intervention locations",
                matrix[selected],
                {
                    "changes": changes,
                    "role": role,
                    "original_equation": asdict(equation),
                    **(
                        {
                            "original_converters": [asdict(c) for c in node.converters],
                            "effective_converters": [
                                asdict(c) for c in execution.nodes[i].converters
                            ]
                            if isinstance(execution, ConverterInterventionExecution)
                            else [],
                        }
                        if isinstance(node, ConvertedCausalNode)
                        else {}
                    ),
                    **(
                        {
                            "variable_spans": [asdict(s) for s in node.spans],
                            "domains": [asdict(d) for d in node.domains],
                        }
                        if isinstance(node, NativeVectorNode)
                        else {}
                    ),
                },
            )
    add(
        "inputs",
        "Observed intervention readouts",
        [],
        "selected declared readouts; state and captured measurements remain distinct",
        task.inputs.cpu()[selected],
        {},
    )
    add(
        "targets",
        "Intervention target",
        [],
        "declared typed capture or numerical readout; no adapter"
        if isinstance(execution, ConverterInterventionExecution)
        else "declared target with original frozen adapter",
        task.targets.cpu()[selected, None],
        {},
    )
    return {
        "version": "generation-graph-v1",
        "episode_id": task.episode_id,
        "rows": selected,
        "component_count": 0,
        "intervention": execution.metadata(),
    }
