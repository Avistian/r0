"""A successful numerical replay cannot authorize contradictory semantic metadata."""

import json
from dataclasses import replace

import pytest
import torch

from r0.modules import priors as p


def emitted():
    task = p.LinearRecipe(
        rows=16,
        context_rows=8,
        features=1,
        signal_features=1,
        active_features=1,
        task_type="multiclass",
        classes=3,
    )
    fn = p.FrozenNetworkFunction(
        1, (p.NetworkBlock((0,), (p.NetworkAffine(((1.0,), (2.0,), (3.0,)), (0.0, 0.0, 0.0)),)),)
    )
    return p.generate_task(
        p.PrototypeBatchNetworkRecipe(p.NetworkRecipe(task, fn)), seed=5, task_id=7
    )


@pytest.mark.parametrize(
    "field",
    [
        "task_type",
        "source_task_id",
        "episode_id",
        "feature_schema",
        "target_origin",
        "mechanism",
        "recipe-extra",
        "input-dtype",
        "target-dtype",
        "probability-dtype",
    ],
)
def test_batch_trace_rejects_contradictory_semantics(field: str):
    task = emitted()
    if field == "task_type":
        bad = replace(task, task_type="regression")
    elif field == "source_task_id":
        bad = replace(task, source_task_id="invented")
    elif field == "episode_id":
        bad = replace(task, episode_id="invented")
    elif field == "feature_schema":
        bad = replace(task, feature_schema=(p.TrainingFeature(),))
    elif field == "target_origin":
        bad = replace(
            task, feature_schema=(p.TrainingFeature(),), target_origin="native-variable-v1"
        )
    elif field == "mechanism":
        card = json.loads(task.mechanism_json)
        card["population_digest"] = "invented"
        bad = replace(task, mechanism_json=json.dumps(card))
    elif field == "recipe-extra":
        raw = json.loads(task.recipe_json)
        raw["unverified"] = True
        bad = replace(task, recipe_json=json.dumps(raw))
    elif field == "input-dtype":
        bad = replace(task, inputs=task.inputs.to(torch.float32))
    elif field == "target-dtype":
        bad = replace(task, targets=task.targets.to(torch.float64))
    else:
        assert task.probabilities is not None
        bad = replace(task, probabilities=task.probabilities.to(torch.float32))
    with pytest.raises(ValueError, match="replay"):
        p.explain_generation(bad, [15, 0, 15])


def test_batch_trace_accepts_original_and_reordered_json_metadata():
    task = emitted()
    reordered = replace(
        task,
        recipe_json=json.dumps(json.loads(task.recipe_json), indent=2),
        mechanism_json=json.dumps(json.loads(task.mechanism_json), indent=2),
    )
    original = p.explain_generation(task, [15, 0, 15])
    assert p.explain_generation(reordered, [15, 0, 15]) == original
