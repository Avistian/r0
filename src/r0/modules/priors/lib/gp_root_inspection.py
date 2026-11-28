"""Saved GP-root task custody and original-row structural inspection."""

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from typing import Any

import torch

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.converted_causal_records import ConvertedCausalRecipe
from r0.modules.priors.lib.converted_inspection import converted_steps
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_causal_evaluation import functional_rng
from r0.modules.priors.lib.functional_steps import functional_steps
from r0.modules.priors.lib.gp_column_inspection import gp_population_matrices
from r0.modules.priors.lib.gp_root_dags import GPRootDAGRecipe
from r0.modules.priors.lib.gp_root_task_validation import validate_gp_root_task
from r0.modules.priors.lib.native_causal import NativeCausalRecipe
from r0.modules.priors.lib.native_inspection import native_steps
from r0.modules.priors.lib.native_vector_inspection import native_vector_steps
from r0.modules.priors.lib.representatives import encode
from r0.modules.priors.lib.target_inspection import target_steps
from r0.modules.priors.lib.training_task import TrainingTask


def gp_root_steps(
    task: TrainingTask,
    indices: Sequence[int],
    add: Callable[..., None],
    *,
    limits: CausalLimits | None,
) -> dict[str, Any]:
    record = json.loads(task.recipe_json)
    recipe, expected = validate_gp_root_task(task, limits=limits)
    assert isinstance(recipe, GPRootDAGRecipe)
    context = recipe.context()
    rows = list(indices)
    add(
        "gp-population",
        "Retained finite GP feature columns",
        [],
        "X[:,j] = L[j] @ Z[:,j], all original rows retained",
        torch.tensor(recipe.source.draw.values, dtype=torch.float64)[rows],
        {"population_id": recipe.source.population_id, "row_ids": rows},
    )
    bindings = {b.node: b for b in recipe.bindings}
    for b in recipe.bindings:
        values = context.values(b.node, recipe.task.rows, len(b.columns), "cpu")
        assert values is not None
        add(
            f"gp-binding-{b.node}",
            f"External root input {b.node}",
            ["gp-population"],
            "select retained GP columns in declared order",
            values[rows],
            asdict(b),
        )

    def bound_add(
        key: str,
        label: str,
        parents: list[str],
        formula: str,
        values: torch.Tensor,
        parameters: dict[str, Any],
    ):
        if key.startswith(("root-", "function-")):
            suffix = key.split("-")[-1]
            if suffix.isdecimal() and int(suffix) in bindings and not parents:
                parents = [f"gp-binding-{suffix}"]
                if key.startswith("root-"):
                    formula = "retained external GP population; placeholder iid law not sampled"
                    parameters = context.metadata(int(suffix))
        add(key, label, parents, formula, values, parameters)

    runtime_record = {
        **record,
        "recipe": asdict(recipe.base),
        "execution": {**record["execution"], "implementation": recipe.base.version},
    }
    runtime_card = json.loads(expected.mechanism_json)["runtime"]
    virtual = replace(
        expected, recipe_json=encode(runtime_record), mechanism_json=encode(runtime_card)
    )
    if isinstance(recipe.base, FunctionalCausalRecipe):
        signal = functional_steps(
            runtime_record,
            runtime_card,
            indices,
            bound_add,
            limits=limits,
            root_populations=context,
        )
        targets = target_steps(
            asdict(recipe.task),
            signal,
            indices,
            lambda role: functional_rng(record["seed"], record["task_id"], role),
            bound_add,
        )
        if not torch.allclose(targets, expected.targets[rows], atol=1e-10, rtol=1e-10):
            raise ValueError("GP root target inspection differs from retained task")
        add(
            "inputs",
            "Observed DAG channels",
            [f"readout-{i}" for i, _ in recipe.base.observed],
            "declared observed channels",
            expected.inputs[rows],
            {"observed": recipe.base.observed},
        )
    elif isinstance(recipe.base, NativeCausalRecipe):
        native_steps(virtual, indices, bound_add, limits=limits, root_populations=context)
    elif isinstance(recipe.base, ConvertedCausalRecipe):
        converted_steps(virtual, indices, bound_add, limits=limits, root_populations=context)
    else:
        native_vector_steps(virtual, indices, bound_add, limits=limits, root_populations=context)
    return {
        "version": "generation-graph-v1",
        "kind": "gp_root_dag",
        "episode_id": task.episode_id,
        "rows": rows,
        "population_id": recipe.source.population_id,
        "shared_matrices": gp_population_matrices(recipe.source),
        "bindings": [asdict(b) for b in recipe.bindings],
        "component": None,
        "component_count": 0,
    }
