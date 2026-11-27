"""Explicit pair invariants for immutable networks and their input views."""

import json
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import torch

from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.matched_network_view import MatchedNetworkInputView
from r0.modules.priors.lib.network_input_view import NetworkInputView
from r0.modules.priors.lib.network_recipe import NetworkRecipe
from r0.modules.priors.lib.network_variance_match import fit_network_variance_match
from r0.modules.priors.lib.representatives import record_fingerprint
from r0.modules.priors.lib.training_task import TrainingTask

type NetworkSource = NetworkRecipe | NetworkInputView


def input_mask(recipe: NetworkSource) -> tuple[int, ...]:
    return (
        recipe.active_inputs
        if isinstance(recipe, NetworkInputView)
        else tuple(range(recipe.function.input_width))
    )


def network_input_pair(
    recipe: NetworkSource,
    controls: LinearRecipe,
    *,
    intervention: str,
    allow_estimated_variance: bool = False,
) -> NetworkSource:
    if intervention == "sparsity" and isinstance(recipe, MatchedNetworkInputView):
        raise ValueError("already matched source sparsity requires a new baseline protocol")
    if intervention not in ("irrelevant", "sparsity"):
        return replace(recipe, task=controls)
    if intervention == "irrelevant" and isinstance(recipe, NetworkInputView):
        return replace(recipe, task=controls)
    base = recipe.base if isinstance(recipe, NetworkInputView) else recipe
    current = input_mask(recipe)
    width = base.function.input_width
    if recipe.task.active_features != len(current) or recipe.task.signal_features != width:
        raise ValueError("network comparison requires task controls matching actual input mask")
    active = current
    if intervention == "sparsity":
        order = (*current, *(i for i in range(width) if i not in current))
        active = tuple(sorted(order[: controls.active_features]))
    if intervention == "sparsity" and allow_estimated_variance:
        identity = record_fingerprint((asdict(base), current, active, "network-pair-matching-v1"))
        matching = fit_network_variance_match(
            base,
            before_inputs=current,
            after_inputs=active,
            rows=256,
            rng=np.random.default_rng(int(identity, 16)),
        )
        return MatchedNetworkInputView(base, controls, active, matching=matching)
    return NetworkInputView(base, controls, active)


def _configuration(recipe: NetworkSource) -> dict[str, Any]:
    base = recipe.base if isinstance(recipe, NetworkInputView) else recipe
    return {
        "network": asdict(base.network),
        "root": asdict(base.root) if base.root else None,
        "limits": asdict(base.limits),
    }


def _verify_task(task: TrainingTask, recipe: NetworkSource) -> None:
    from r0.modules.priors.lib.generate import generate_task

    record = json.loads(task.recipe_json)
    if record["execution"].get("implementation") != recipe.version:
        raise ValueError("network comparison implementation differs from its recipe")
    replay = generate_task(
        recipe,
        seed=record["seed"],
        task_id=record["task_id"],
        device=record["execution"]["device"],
    )
    for field in ("inputs", "scores", "targets", "context_indices", "query_indices"):
        if not torch.equal(getattr(replay, field).cpu(), getattr(task, field).cpu()):
            raise ValueError(f"network comparison {field} differs from replay")
    if (replay.probabilities is None) != (task.probabilities is None):
        raise ValueError("network comparison probabilities differ from replay")
    if (
        replay.probabilities is not None
        and task.probabilities is not None
        and not torch.equal(replay.probabilities.cpu(), task.probabilities.cpu())
    ):
        raise ValueError("network comparison probabilities differ from replay")
    if (
        replay.episode_id != task.episode_id
        or replay.source_task_id != task.source_task_id
        or replay.classes != task.classes
        or replay.feature_schema != task.feature_schema
        or replay.target_origin != task.target_origin
        or replay.task_type != task.task_type
        or replay.mechanism_json != task.mechanism_json
    ):
        raise ValueError("network comparison identity or mechanism differs from replay")


def network_pair_equalities(
    before: TrainingTask, after: TrainingTask, left: NetworkSource, right: NetworkSource
) -> dict[str, bool]:
    _verify_task(before, left)
    _verify_task(after, right)
    executions = [json.loads(task.recipe_json)["execution"] for task in (before, after)]
    same_network = _configuration(left) == _configuration(right)
    return {
        "execution": (
            {k: v for k, v in executions[0].items() if k != "implementation"}
            == {k: v for k, v in executions[1].items() if k != "implementation"}
        ),
        "frozen_network": same_network,
        "mechanism": same_network and input_mask(left) == input_mask(right),
    }
