"""Original-row population paths, structural stages, typed emissions and target adapters."""

from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any

import torch

from r0.modules.priors.lib.causal_intervention_execution import intervention_execution
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.converted_causal_records import ConvertedCausalRecipe
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_causal_evaluation import functional_rng
from r0.modules.priors.lib.native_vector_records import NativeVectorRecipe
from r0.modules.priors.lib.population_task_semantics import (
    population_task_record,
    validate_population_task,
)
from r0.modules.priors.lib.recursive_population_graph_intervention import (
    RecursivePopulationGraphIntervention,
)
from r0.modules.priors.lib.recursive_population_graph_plan import population_equations
from r0.modules.priors.lib.recursive_population_graph_records import RecursivePopulationGraph
from r0.modules.priors.lib.training_task import TrainingTask


def population_graph_steps(
    task: TrainingTask,
    indices: Sequence[int],
    add: Callable[..., None],
    *,
    limits: CausalLimits | None,
) -> dict[str, Any]:
    from r0.modules.priors.lib.generate import generate_task, recipe_from_dict
    from r0.modules.priors.lib.target_inspection import target_steps

    record = population_task_record(task)
    recipe = recipe_from_dict(record["recipe"], limits=limits or CausalLimits())
    if not isinstance(recipe, (RecursivePopulationGraph, RecursivePopulationGraphIntervention)):
        raise ValueError("population DAG trace requires its explicit archive")
    replay = generate_task(recipe, seed=record["seed"], task_id=record["task_id"])
    validate_population_task(task, replay, record)
    base = recipe.base if isinstance(recipe, RecursivePopulationGraphIntervention) else recipe
    change = recipe.change if isinstance(recipe, RecursivePopulationGraphIntervention) else None
    runtime = change.effective_runtime() if change else base.request.base
    result = recipe.replay(limits=limits)
    context = result.context
    rows = list(indices)
    for i, equation in enumerate(population_equations(runtime)):
        stages = context.stages[i]
        node = runtime.nodes[i]
        metadata = {
            "original_rows": rows,
            "effective_node": asdict(node),
            "interventions": [asdict(c) for c in change.interventions if c.node == i]
            if change
            else [],
        }
        prefix = f"dag-{i}"
        current: list[str] = []
        if i in context.inputs:
            key = f"{prefix}-inputs"
            add(
                key,
                f"Node {i} original inputs",
                [f"dag-{p}-state" for p in equation.parents],
                "concatenate encoded parent states"
                if equation.parents
                else "draw original keyed root law",
                context.inputs[i][rows],
                metadata,
            )
            current = [key]
            if context.replaces(i):
                current = context.steps(i, parents=current, indices=indices, add=add)
            else:
                from r0.modules.priors.lib.population_function_inspection import (
                    population_function_steps,
                )

                current = population_function_steps(
                    equation.function,
                    context.inputs[i],
                    prefix=f"{prefix}-detail",
                    parents=current,
                    indices=indices,
                    add=add,
                )
            key = f"{prefix}-function"
            add(
                key,
                f"Node {i} mechanism",
                current,
                "row-owned forest aggregation"
                if context.replaces(i)
                else "evaluate effective frozen function",
                stages["function"][rows],
                context.metadata(i) if context.replaces(i) else asdict(equation.function),
            )
            key_before = f"{prefix}-before-noise"
            add(
                key_before,
                f"Node {i} pipeline before noise",
                [key],
                "apply effective pre-innovation pipeline",
                stages["before_noise"][rows],
                metadata,
            )
            noise, scale = f"{prefix}-innovation", f"{prefix}-scale"
            add(
                noise,
                f"Node {i} structural innovation",
                [],
                "original keyed exogenous draw",
                stages["exogenous"][rows],
                asdict(equation.exogenous),
            )
            add(
                scale,
                f"Node {i} innovation scale",
                [key],
                "noise_scale * signal-dependent response",
                stages["scale"][rows],
                {"noise_scale": equation.noise_scale, "response": asdict(equation.noise_response)},
            )
            key_sum = f"{prefix}-pre-finish"
            add(
                key_sum,
                f"Node {i} noisy stage",
                [key_before, noise, scale],
                "before_noise + scale * innovation",
                stages["pre_finish"][rows],
                {},
            )
            key_post = f"{prefix}-post-finish"
            add(
                key_post,
                f"Node {i} pipeline output before cuts",
                [key_sum],
                "finish effective pipeline",
                stages["post_finish"][rows],
                metadata,
            )
            current = [key_post]
        # Emission cuts are displayed as their own source nodes, never as original logits.
        for name, value in stages.items():
            if name.startswith("forced-score:"):
                key = f"{prefix}-{name}"
                add(
                    key,
                    f"Node {i} forced score span",
                    [],
                    "explicit logical-variable clamp",
                    value[rows],
                    metadata,
                )
                current.append(key)
        if "scores" in stages:
            key = f"{prefix}-scores"
            add(
                key,
                f"Node {i} effective scores",
                current,
                "effective post-pipeline scores with explicit cuts",
                stages["scores"][rows],
                metadata,
            )
            current = [key]
        values: list[str] = []
        value_names = [
            name
            for name in stages
            if name.startswith("value:") or name.startswith("capture:") and name.endswith(":value")
        ]
        # Numeric readouts and categorical captures coexist on converted nodes.
        # Keep both: targets/observations select either, descendants consume state.
        if "readout" in stages:
            value_names.append("readout")
        for name in value_names:
            stem = name.rsplit(":", 1)[0] if name.startswith("capture:") else ""
            suffix = name.split(":")[1] if name.startswith("value:") else ""
            prob_name = (
                f"{stem}:probability"
                if stem
                else f"probability:{suffix}"
                if suffix
                else "probability"
            )
            uniform_name = (
                f"{stem}:uniform" if stem else f"uniform:{suffix}" if suffix else "uniform"
            )
            parents = list(current)
            if stem and f"{stem}:scores" in stages:
                score_id = f"{prefix}-{stem}:scores"
                add(
                    score_id,
                    f"Node {i} captured emission scores",
                    [f"{prefix}-function"] if i in context.inputs else [],
                    "converter captures before structural innovation/importance",
                    stages[f"{stem}:scores"][rows],
                    metadata,
                )
                parents = [score_id]
            if prob_name in stages:
                prob_id = f"{prefix}-{prob_name}"
                add(
                    prob_id,
                    f"Node {i} emission probabilities or forced indicator",
                    parents,
                    (
                        "effective domain emission; forced indicator is not original "
                        "stochastic probability"
                    ),
                    stages[prob_name][rows],
                    metadata,
                )
                parents = [prob_id]
            if uniform_name in stages:
                uniform_id = f"{prefix}-{uniform_name}"
                add(
                    uniform_id,
                    f"Node {i} emission uniforms",
                    [],
                    "original row-keyed uniform draws",
                    stages[uniform_name][rows, None],
                    {},
                )
                parents.append(uniform_id)
            key = f"{prefix}-{name}"
            add(
                key,
                f"Node {i} {name}",
                parents,
                "emit native value/captured readout or explicit cut",
                stages[name][rows],
                metadata,
            )
            values.append(key)
        add(
            f"{prefix}-state",
            f"Node {i} descendant state",
            list(dict.fromkeys(current + values)),
            "effective encoded state; categorical descendants receive one-hot values",
            stages["state"][rows],
            metadata,
        )
    if isinstance(runtime, FunctionalCausalRecipe):
        parents = list(dict.fromkeys(f"dag-{i}-readout" for i, _ in runtime.target))
        add(
            "signal",
            "Declared target readouts",
            parents,
            "select target channels",
            replay.scores[rows],
            {"target": runtime.target},
        )
        execution = intervention_execution(change) if change else None

        def rng(key: str):
            def original(*keys: object):
                return functional_rng(base.seed, base.task_id, *keys)

            return execution.rng(original, key) if execution else original(key)

        targets = target_steps(asdict(runtime.task), replay.scores[rows], indices, rng, add)
        if not torch.equal(targets, replay.targets[rows]):
            raise ValueError("population graph target adapter trace differs from replay")
    else:
        if isinstance(runtime, ConvertedCausalRecipe):
            ref = runtime.target
            parent = (
                f"dag-{ref.node}-capture:{ref.capture}:value"
                if ref.capture
                else f"dag-{ref.node}-readout"
            )
        elif isinstance(runtime, NativeVectorRecipe):
            parent = f"dag-{runtime.target[0]}-value:{runtime.target[1]}"
        else:
            parent = f"dag-{runtime.target[0]}-readout"
        add(
            "target",
            "Direct typed target",
            [parent],
            "select actual emitted value without target adapter",
            replay.targets[rows, None],
            {"target_origin": replay.target_origin, "classes": replay.classes},
        )
    return {
        "version": "generation-graph-v1",
        "recipe_version": recipe.version,
        "episode_id": task.episode_id,
        "rows": rows,
        "component": None,
        "component_count": 1,
        "target_error": 0.0,
    }
