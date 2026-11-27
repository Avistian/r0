"""Public inspection rejects tensor and record semantic aliases while retaining provenance."""

import json
from dataclasses import replace
from typing import Any

import pytest
import torch

from r0.modules import priors as p


@pytest.fixture(scope="module")
def task() -> p.TrainingTask:
    controls = p.LinearRecipe(
        rows=16, context_rows=8, features=1, signal_features=1, active_features=1
    )
    root = p.FunctionalCausalNode(
        (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
    )
    child = p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0,),)))
    ancestors = p.FunctionalCausalRecipe(controls, (root, child), ((0, 0),), ((1, 0),))
    plan = p.RecursiveForestPlan(
        input_width=1,
        intervals=((-2.0, 2.0),),
        estimators=2,
        min_depth=1,
        max_depth=1,
        task="classification",
        leaf_mean=(),
        leaf_scale=(),
        classes=3,
        class_assignment="random-start-cycle",
        tie_break="first-tree",
        lifecycle="fresh-forest-per-row",
        populations=16,
    )
    recipe = p.realize_recursive_population(
        p.RecursivePopulationRequest(ancestors, ((1, 0),), plan, ("a", "b", "c")),
        seed=17,
        task_id=2**54 + 5,
    )
    return p.generate_task(recipe, seed=17, task_id=2**54 + 5)


@pytest.mark.parametrize("field", ["inputs", "scores", "context_indices", "query_indices"])
def test_tensor_dtype_alias_rejected(task: p.TrainingTask, field: str):
    value = getattr(task, field).to(torch.float32)
    with pytest.raises(ValueError, match="population.*semantic|population.*replay"):
        p.explain_generation(replace(task, **{field: value}), [15, 0, 15])


@pytest.mark.parametrize(
    "field", ["inputs", "scores", "targets", "context_indices", "query_indices"]
)
def test_tensor_shape_alias_rejected(task: p.TrainingTask, field: str):
    value = getattr(task, field).unsqueeze(-1)
    with pytest.raises(ValueError, match="population.*semantic|population.*replay"):
        p.explain_generation(replace(task, **{field: value}), [0])


@pytest.mark.parametrize("field", ["inputs", "scores"])
def test_sparse_layout_rejected_before_indexing(task: p.TrainingTask, field: str):
    value = getattr(task, field).to_sparse()
    with pytest.raises(ValueError, match="population.*semantic|population.*replay"):
        p.explain_generation(replace(task, **{field: value}), [0])


@pytest.mark.parametrize("field", ["inputs", "scores"])
def test_autograd_contract_rejected(task: p.TrainingTask, field: str):
    value = getattr(task, field).clone().requires_grad_()
    with pytest.raises(ValueError, match="population.*semantic|population.*replay"):
        p.explain_generation(replace(task, **{field: value}), [0])


@pytest.mark.parametrize("key", ["recipe", "seed", "task_id", "execution"])
def test_missing_outer_record_field_rejected(task: p.TrainingTask, key: str):
    record = json.loads(task.recipe_json)
    del record[key]
    with pytest.raises(ValueError, match="population.*record|population.*semantic"):
        p.explain_generation(replace(task, recipe_json=json.dumps(record)), [0])


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("implementation", True),
        ("device", True),
        ("device", "tpu"),
        ("python", 3.11),
        ("numpy", None),
        ("torch", []),
        ("cpu_threads", True),
        ("cpu_threads", 1.0),
        ("cpu_threads", 0),
        ("cuda", False),
        ("gpu", {}),
    ],
)
def test_invalid_execution_primitives_rejected(task: p.TrainingTask, key: str, value: Any):
    record = json.loads(task.recipe_json)
    record["execution"][key] = value
    with pytest.raises(ValueError, match="population.*execution|population.*semantic"):
        p.explain_generation(replace(task, recipe_json=json.dumps(record)), [0])


@pytest.mark.parametrize("location", ["outer", "execution", "mechanism"])
def test_unknown_semantic_field_rejected(task: p.TrainingTask, location: str):
    record = json.loads(task.mechanism_json if location == "mechanism" else task.recipe_json)
    target = record["execution"] if location == "execution" else record
    target["unexpected_semantics"] = 1
    changed = replace(
        task,
        **{("mechanism_json" if location == "mechanism" else "recipe_json"): json.dumps(record)},
    )
    with pytest.raises(
        ValueError, match="population.*record|population.*semantic|population.*execution"
    ):
        p.explain_generation(changed, [0])


def test_historical_descriptors_and_noncontiguous_values_remain_valid(task: p.TrainingTask):
    record = json.loads(task.recipe_json)
    record["execution"].update(
        {
            "device": "cuda",
            "python": "3.10.99",
            "numpy": "1.20.0",
            "torch": "1.9.0",
            "cpu_threads": 8,
            "cuda": "11.0",
            "gpu": "historical descriptor only",
        }
    )
    scores = task.scores.T.contiguous().T
    assert not scores.is_contiguous()
    changed = replace(task, recipe_json=json.dumps(record, indent=2), scores=scores)
    assert p.explain_generation(changed, [15, 0, 15])["target_error"] == 0
    assert changed.recipe_json == json.dumps(record, indent=2)


@pytest.mark.parametrize("field", ["recipe_json", "mechanism_json"])
def test_duplicate_json_fields_rejected(task: p.TrainingTask, field: str):
    original = getattr(task, field)
    duplicate = (
        '{"seed":0,' + original[1:]
        if field == "recipe_json"
        else '{"family":"ignored",' + original[1:]
    )
    with pytest.raises(ValueError, match="population.*duplicate"):
        p.explain_generation(replace(task, **{field: duplicate}), [0])


def test_equal_values_changed_regression_target_dtype_rejected(task: p.TrainingTask):
    recipe = p.recipe_from_dict(json.loads(task.recipe_json)["recipe"])
    assert isinstance(recipe, p.RecursivePopulationRecipe)
    plan = replace(
        recipe.request.plan,
        task="regression",
        leaf_mean=(0.0,),
        leaf_scale=(0.0,),
        classes=0,
        class_assignment="none",
        tie_break="none",
    )
    request = replace(recipe.request, plan=plan, vocabulary=())
    realized = p.realize_recursive_population(request, seed=17, task_id=7)
    regression = p.generate_task(realized, seed=17, task_id=7)
    with pytest.raises(ValueError, match="population.*semantic"):
        p.explain_generation(replace(regression, targets=regression.targets.to(torch.float32)), [0])
