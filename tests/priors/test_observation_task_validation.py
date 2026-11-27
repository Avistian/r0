"""Saved observation traces reject semantic tampering, including unselected rows."""

import json
from dataclasses import replace
from typing import Literal

import numpy as np
import pytest
import torch

from r0.modules import priors as p


@pytest.fixture(params=["regression", "binary", "multiclass"])
def saved(request: pytest.FixtureRequest) -> p.TrainingTask:
    task_type: Literal["regression", "binary", "multiclass"] = request.param
    task = p.LinearRecipe(
        rows=16,
        context_rows=7,
        features=1,
        signal_features=1,
        active_features=1,
        task_type="multiclass",
        classes=3,
    )
    first = p.NetworkBlock((0,), (p.NetworkAffine(((1.0,), (2.0,), (3.0,)), (0.0,) * 3),))
    second = p.NetworkBlock((1,), (p.NetworkAffine(((1.0, 0.0, 0.0),) * 3, (1.0,) * 3),))
    source = p.NetworkRecipe(task, p.FrozenNetworkFunction(1, (first, second)))
    receipt = p.sample_network_observations(
        p.NetworkObservationRequest(source, 1, outputs=3 if task_type == "multiclass" else 1),
        rng=np.random.default_rng(2),
    )
    recipe = p.NetworkObservationRecipe(
        replace(
            task,
            task_type=task_type,
            classes={"regression": 0, "binary": 2, "multiclass": 3}[task_type],
        ),
        receipt,
    )
    return p.generate_task(recipe, seed=37, task_id=2**54 + 17)


@pytest.mark.parametrize(
    "field",
    ["task_type", "classes", "source_task_id", "episode_id", "feature_schema", "target_origin"],
)
def test_observation_semantic_metadata_rejects(saved: p.TrainingTask, field: str) -> None:
    changes = {
        "task_type": "binary" if saved.task_type != "binary" else "regression",
        "classes": ("forged",),
        "source_task_id": "bad",
        "episode_id": "bad",
        "feature_schema": (p.TrainingFeature(),),
        "target_origin": "native-variable-v1",
    }
    if field == "target_origin":
        bad = replace(
            saved, target_origin="native-variable-v1", feature_schema=(p.TrainingFeature(),)
        )
    else:
        bad = replace(saved, **{field: changes[field]})
    with pytest.raises(ValueError, match="metadata"):
        p.explain_generation(bad, [0])


@pytest.mark.parametrize(
    "field", ["inputs", "targets", "scores", "context_indices", "query_indices", "probabilities"]
)
@pytest.mark.parametrize("change", ["value", "dtype", "shape", "layout"])
def test_observation_full_arrays_reject(saved: p.TrainingTask, field: str, change: str) -> None:
    tensor = getattr(saved, field)
    if tensor is None:
        altered = torch.zeros((16, 1), dtype=torch.float64)
    elif change == "value":
        altered = tensor.clone()
        altered[-1] += 1
    elif change == "dtype":
        altered = tensor.float() if tensor.dtype != torch.float32 else tensor.double()
    elif change == "shape":
        altered = tensor.unsqueeze(-1)
    else:
        altered = tensor.to_sparse()
    with pytest.raises(ValueError, match="array|probability"):
        p.explain_generation(replace(saved, **{field: altered}), [0])


@pytest.mark.parametrize(
    "change",
    [
        "seed-bool",
        "seed-float",
        "id-string",
        "id-range",
        "implementation",
        "device",
        "threads",
        "python",
        "cuda",
        "missing",
        "extra",
        "record-extra",
        "card",
    ],
)
def test_observation_archive_schema_rejects(saved: p.TrainingTask, change: str) -> None:
    record = json.loads(saved.recipe_json)
    if change == "seed-bool":
        record["seed"] = True
    elif change == "seed-float":
        record["seed"] = 37.0
    elif change == "id-string":
        record["task_id"] = str(record["task_id"])
    elif change == "id-range":
        record["task_id"] = 2**63
    elif change == "implementation":
        record["execution"]["implementation"] = "linear-v1"
    elif change == "device":
        record["execution"]["device"] = "invalid"
    elif change == "threads":
        record["execution"]["cpu_threads"] = True
    elif change == "python":
        record["execution"]["python"] = 1
    elif change == "cuda":
        record["execution"]["cuda"] = False
    elif change == "missing":
        del record["execution"]["numpy"]
    elif change == "extra":
        record["execution"]["extra"] = "unknown"
    elif change == "record-extra":
        record["extra"] = 1
    else:
        card = json.loads(saved.mechanism_json)
        card["family"] = "forged"
        saved = replace(saved, mechanism_json=json.dumps(card))
    with pytest.raises(ValueError, match="observation"):
        p.explain_generation(replace(saved, recipe_json=json.dumps(record)), [0])


def test_descriptive_historical_environment_and_order_permitted(saved: p.TrainingTask) -> None:
    record = json.loads(saved.recipe_json)
    record["execution"].update(
        python="historical",
        numpy="historical",
        torch="historical",
        cpu_threads=7,
        device="cuda",
        cuda="historical",
        gpu="historical",
    )
    old = replace(
        saved,
        recipe_json=json.dumps(record, indent=2),
        mechanism_json=json.dumps(json.loads(saved.mechanism_json), indent=2),
    )
    assert p.explain_generation(old, [15, 0, 15])["target_max_error"] == 0


def test_observation_intervention_metadata_rejects(saved: p.TrainingTask) -> None:
    recipe = p.recipe_from_dict(json.loads(saved.recipe_json)["recipe"])
    assert isinstance(recipe, p.NetworkObservationRecipe)
    ref = recipe.runtime.target[0]
    pair = p.generate_causal_pair(
        recipe,
        interventions=(p.HardIntervention(ref[0], channels=(ref[1],), values=(4.0,)),),
        seed=37,
        task_id=2**54 + 17,
    )
    assert p.explain_generation(pair.after, [0])["rows"] == [0]
    with pytest.raises(ValueError, match="metadata"):
        p.explain_generation(replace(pair.after, episode_id="bad"), [0])
