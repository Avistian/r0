"""Full topology/geometry custody in shared task identity and original-row traces."""

import json
from collections.abc import Sequence
from dataclasses import asdict, replace
from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.hyperprior_task_validation import validate_hyperprior_task
from r0.modules.priors.lib.representatives import encode, record_fingerprint
from r0.modules.priors.lib.training_task import TrainingTask
from r0.modules.priors.lib.typed_topology_decoding import typed_topology_graph_from_dict
from r0.modules.priors.lib.typed_topology_graph import TypedTopologyGraph


def generate_typed_topology(
    recipe: TypedTopologyGraph, *, seed: int, task_id: int, device: str = "cpu"
) -> TrainingTask:
    from r0.modules.priors.lib.generate import generate_task

    if device != "cpu" or any(type(v) is not int or not 0 <= v < 2**63 for v in (seed, task_id)):
        raise ValueError("typed topology requires exact bounded identities and CPU execution")
    valid = typed_topology_graph_from_dict(asdict(recipe), limits=recipe.limits)
    task = generate_task(valid.graph, seed=seed, task_id=task_id, device=device)
    record, card = json.loads(task.recipe_json), json.loads(task.mechanism_json)
    record["recipe"] = asdict(valid)
    record["execution"]["implementation"] = valid.version
    card.update(
        version=valid.version,
        family="typed_topology",
        runtime_family=valid.request.family,
        construction="topology-geometry-typed-mechanisms-r0-v1",
        topology_receipt=record_fingerprint(asdict(valid.topology)),
        mechanism_receipt=record_fingerprint(asdict(valid.graph)),
        geometry_receipt=None
        if valid.geometry is None
        else record_fingerprint(asdict(valid.geometry)),
        source_semantics=(
            "P2 redirection reconstruction composed with typed r0 geometry; "
            "not a full paper reproduction"
        ),
        domain_width_semantics=(
            "native scalar continuous width; vector variable count; "
            "geometry latent plus reserved capture channels"
        ),
    )
    source = record_fingerprint((record["recipe"], seed, task_id))
    return replace(
        task,
        recipe_json=encode(record),
        mechanism_json=encode(card),
        source_task_id=source,
        episode_id=record_fingerprint((source, valid.task.split, valid.task.context_rows)),
    )


def typed_topology_trace(
    task: TrainingTask,
    indices: Sequence[int],
    *,
    limits: CausalLimits,
    column_start: int = 0,
) -> dict[str, Any]:
    from r0.modules.priors.lib.generate import generate_task
    from r0.modules.priors.lib.generation_graph import explain_generation

    record = json.loads(task.recipe_json)
    validate_hyperprior_task(task, record, limits=limits)
    recipe = typed_topology_graph_from_dict(record["recipe"], limits=limits)
    child = generate_task(recipe.graph, seed=record["seed"], task_id=record["task_id"])
    result = explain_generation(child, indices, limits=limits, column_start=column_start)
    result["episode_id"] = task.episode_id
    result["topology_envelope"] = record_fingerprint(record["recipe"])
    result["nodes"].insert(
        0,
        dict(
            id="typed-topology-construction",
            label="Retained topology and typed geometry",
            parents=[],
            formula="one topology draw; typed geometry preserves every parent edge",
            shape=[1, 2],
            column_indices=[0, 1],
            values=[[len(recipe.topology.parents), len(recipe.topology.components)]],
            parameters=dict(
                scope="global construction metadata",
                columns=["nodes", "components"],
                family=recipe.request.family,
                receipt=result["topology_envelope"],
                source="P2 topology reconstruction + explicit r0 typed composition",
            ),
        ),
    )
    records = list(recipe.topology.raw)
    if recipe.geometry is not None:
        records.extend(recipe.geometry.raw)
    for i, raw in enumerate(records):
        result["nodes"].insert(
            i + 1,
            dict(
                id=f"typed-topology-draw-{i}",
                label=raw.role,
                parents=["typed-topology-construction"],
                formula=raw.distribution,
                shape=[1, len(raw.values)],
                column_indices=list(range(min(32, len(raw.values)))),
                values=[list(raw.values[:32])],
                parameters=dict(
                    scope="global construction draw",
                    owner=raw.owner,
                    stream=raw.stream,
                    raw_shape=raw.shape,
                    parameters=raw.parameters,
                    state_before=raw.state_before,
                    state_after=raw.state_after,
                ),
            ),
        )
    return result
