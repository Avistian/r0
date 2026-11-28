"""Per-task tree construction with request-bound, archived random-draw receipts."""

import json
from dataclasses import asdict, replace
from typing import Any

import numpy as np

from r0.modules.generation.lib.construction import GraphConstruction
from r0.modules.priors import (
    CausalLimits,
    Recipe,
    SampledTreeExtension,
    TrainingTask,
    generate_task,
    sample_tree_extension,
    tree_extension_request_from_dict,
)


def construct_tree_extension(
    construction: GraphConstruction, *, limits: CausalLimits, seed: int, task_id: int
) -> SampledTreeExtension:
    if any(type(v) is not int or not 0 <= v < 2**63 for v in (seed, task_id)):
        raise ValueError("tree construction requires exact integer episode identities")
    controls = json.loads(construction.controls_json)
    if set(controls) != {"topology", "request"} or controls["topology"] != "tree-extension":
        raise ValueError("tree construction requires topology and request controls")
    request = tree_extension_request_from_dict(controls["request"], limits=limits)
    # Explicit domain tag keeps this stream independent of task row generation.
    rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([0x54524545, construction.seed, seed, task_id]))
    )
    return sample_tree_extension(request, rng=rng, limits=limits)


def tree_construction_record(
    construction: GraphConstruction, sampled: SampledTreeExtension, *, seed: int, task_id: int
) -> dict[str, Any]:
    return json.loads(
        json.dumps(
            {
                "version": "r0-tree-construction-v1",
                "construction_seed": construction.seed,
                "seed": seed,
                "task_id": task_id,
                "sampled": asdict(sampled),
            },
            allow_nan=False,
        )
    )


def construct_tree_bundle(
    construction: GraphConstruction, *, limits: CausalLimits, seed: int, task_id: int
) -> tuple[Recipe, dict[str, Any]]:
    if json.loads(construction.controls_json).get("topology") == "tree-ensemble":
        from r0.modules.generation.lib.tree_ensemble_construction import construct_tree_ensemble

        return construct_tree_ensemble(construction, limits=limits, seed=seed, task_id=task_id)
    sampled = construct_tree_extension(construction, limits=limits, seed=seed, task_id=task_id)
    return sampled.recipe, tree_construction_record(
        construction, sampled, seed=seed, task_id=task_id
    )


def generate_tree_extension(
    construction: GraphConstruction, *, limits: CausalLimits, seed: int, task_id: int, device: str
) -> TrainingTask:
    recipe, receipt = construct_tree_bundle(construction, limits=limits, seed=seed, task_id=task_id)
    task = generate_task(recipe, seed=seed, task_id=task_id, device=device)
    record = json.loads(task.recipe_json)
    record["construction"] = receipt
    return replace(
        task, recipe_json=json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )


def tree_record_matches(
    record: dict[str, Any],
    construction: GraphConstruction,
    *,
    limits: CausalLimits,
    seed: int,
    task_id: int,
) -> bool:
    recipe, expected = construct_tree_bundle(
        construction, limits=limits, seed=seed, task_id=task_id
    )
    # JSON equality distinguishes bool/float aliases from integer identities.
    return all(
        json.dumps(actual, sort_keys=True, allow_nan=False)
        == json.dumps(wanted, sort_keys=True, allow_nan=False)
        for actual, wanted in (
            (record.get("construction"), expected),
            (record.get("recipe"), asdict(recipe)),
            (record.get("seed"), seed),
            (record.get("task_id"), task_id),
            (record.get("sampling"), None),
        )
    )
