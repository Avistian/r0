"""Expose sampled native constructions through the same typed task contract."""

import json
from dataclasses import asdict, replace

from r0.modules.priors.lib.native_vector_output import generate_native_vector
from r0.modules.priors.lib.representatives import encode, record_fingerprint
from r0.modules.priors.lib.sampled_native_vector_graph import SampledNativeVectorGraph
from r0.modules.priors.lib.training_task import TrainingTask


def generate_sampled_native_vector(
    graph: SampledNativeVectorGraph, *, seed: int, task_id: int, device: str = "cpu"
) -> TrainingTask:
    graph.validate_replay()
    task = generate_native_vector(graph.recipe, seed=seed, task_id=task_id, device=device)
    config = asdict(graph)
    source = {
        **config,
        "request": {
            k: v for k, v in asdict(graph.request).items() if k not in ("context_rows", "split")
        },
        "recipe": {
            **asdict(graph.recipe),
            "task": {
                k: v for k, v in asdict(graph.task).items() if k not in ("context_rows", "split")
            },
        },
    }
    source_id = record_fingerprint((source, seed, task_id))
    record = json.loads(task.recipe_json)
    record["recipe"] = config
    record["execution"]["implementation"] = graph.version
    mechanism = json.loads(task.mechanism_json)
    mechanism["construction_version"] = graph.version
    mechanism["construction_digest"] = record_fingerprint(source)
    return replace(
        task,
        source_task_id=source_id,
        episode_id=record_fingerprint((source_id, graph.task.split, graph.task.context_rows)),
        recipe_json=encode(record),
        mechanism_json=encode(mechanism),
    )
