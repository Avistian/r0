"""Generated task identity and construction cards for jointly sampled random-size forests."""

import json
from dataclasses import asdict, replace
from typing import Any

from r0.modules.priors.lib.functional_causal_evaluation import generate_functional_causal
from r0.modules.priors.lib.joint_hyperprior_graph import JointHyperpriorGraph
from r0.modules.priors.lib.representatives import encode, record_fingerprint
from r0.modules.priors.lib.training_task import TrainingTask


def joint_hyperprior_card(recipe: JointHyperpriorGraph) -> dict[str, Any]:
    return dict(
        version=recipe.version,
        family="joint_hyperprior_graph",
        construction="independent-topology-and-joint-source-mechanisms-r0-v1",
        topology=asdict(recipe.topology.topology),
        topology_components=recipe.topology.components,
        topology_layout=recipe.request.layout,
        topology_receipt_fingerprint=record_fingerprint(asdict(recipe.topology)),
        mechanism_receipt_fingerprint=record_fingerprint(asdict(recipe.graph)),
        resource_admission=recipe.request.admit(),
    )


def generate_joint_hyperprior_graph(
    recipe: JointHyperpriorGraph, *, seed: int, task_id: int, device: str = "cpu"
) -> TrainingTask:
    recipe.validate()
    task = generate_functional_causal(recipe.runtime, seed=seed, task_id=task_id, device=device)
    record, card = json.loads(task.recipe_json), json.loads(task.mechanism_json)
    record["recipe"] = asdict(recipe)
    record["execution"]["implementation"] = recipe.version
    card.update(joint_hyperprior_card(recipe))
    config = asdict(recipe)
    controls = {k: v for k, v in asdict(recipe.task).items() if k not in ("context_rows", "split")}
    config["request"]["task"] = controls
    config["graph"]["request"]["task"] = controls
    config["graph"]["runtime"]["task"] = controls
    identity = record_fingerprint((config, seed, task_id))
    return replace(
        task,
        source_task_id=identity,
        episode_id=record_fingerprint((identity, recipe.task.split, recipe.task.context_rows)),
        recipe_json=encode(record),
        mechanism_json=encode(card),
    )
