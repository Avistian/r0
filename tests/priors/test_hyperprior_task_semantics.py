"""Inspection binds the whole retained task before selecting original trace rows."""

import json
from dataclasses import replace
from typing import Any, Literal

import numpy as np
import pytest
import torch

from r0.modules import priors as p


@pytest.fixture(scope="module", params=[False, True])
def saved(request: pytest.FixtureRequest) -> tuple[p.TrainingTask, p.CausalLimits]:
    controls = p.LinearRecipe(
        rows=16, context_rows=8, features=1, signal_features=1, active_features=1
    )
    limits = p.CausalLimits(parameters=8_000_000, working_bytes=2 * 1024**3)
    if request.param:
        prior = p.JointFunctionPrior("lin", matrix=p.JointMatrixPrior(families=("gaussian",)))
        plan = p.JointHyperpriorGraphRequest(
            controls,
            p.FixedSizeTopologyHyperprior(4),
            calibration_rows=16,
            root_prior=p.JointRootPrior(("gaussian",), prior),
            function_prior=prior,
            limits=limits,
        )
        streams = [np.random.default_rng(i + 71) for i in range(5)]
        recipe = p.sample_joint_hyperprior_graph(
            plan,
            topology_rng=streams[0],
            named_rng=streams[1],
            raw_rng=streams[2],
            calibration_rng=streams[3],
            fit_rng=streams[4],
        )
    else:
        recipe = p.sample_hyperprior_graph(
            p.HyperpriorGraphRequest(
                controls,
                p.FixedSizeTopologyHyperprior(4),
                p.FunctionalCausalNode(
                    (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
                ),
                p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0,),))),
                limits=limits,
            ),
            rng=np.random.default_rng(71),
        )
    return p.generate_task(recipe, seed=2**53 + 17, task_id=2**53 + 29), limits


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_type", "binary"),
        ("classes", ("a", "b")),
        ("source_task_id", "wrong"),
        ("episode_id", "wrong"),
        ("feature_schema", (p.TrainingFeature(),)),
    ],
)
def test_semantic_headers_rejected(
    saved: tuple[p.TrainingTask, p.CausalLimits], field: str, value: Any
):
    task, limits = saved
    with pytest.raises(ValueError, match="hyperprior"):
        p.explain_generation(replace(task, **{field: value}), [7, 1, 7], limits=limits)


@pytest.mark.parametrize(
    "field", ["inputs", "targets", "scores", "context_indices", "query_indices"]
)
@pytest.mark.parametrize("mutation", ["dtype", "values", "shape", "layout"])
def test_entire_tensor_contract_rejected(
    saved: tuple[p.TrainingTask, p.CausalLimits], field: str, mutation: str
):
    task, limits = saved
    original = getattr(task, field)
    value = original.clone()
    if mutation == "dtype":
        value = value.to(torch.float32)
    elif mutation == "values":
        value[-1] += 1
    elif mutation == "shape":
        value = value[:-1]
    else:
        value = value.to_sparse()
    with pytest.raises(ValueError, match="hyperprior"):
        p.explain_generation(replace(task, **{field: value}), [0], limits=limits)


@pytest.mark.parametrize(
    ("key", "value"), [("seed", 1.0), ("task_id", True), ("unknown", "forged")]
)
def test_exact_record_headers_rejected(
    saved: tuple[p.TrainingTask, p.CausalLimits], key: str, value: Any
):
    task, limits = saved
    record = json.loads(task.recipe_json)
    record[key] = value
    with pytest.raises(ValueError, match="hyperprior"):
        p.explain_generation(replace(task, recipe_json=json.dumps(record)), [0], limits=limits)


@pytest.mark.parametrize(
    ("key", "value"),
    [("implementation", "other"), ("cpu_threads", True), ("unknown", "forged"), ("numpy", 1)],
)
def test_execution_semantics_rejected(
    saved: tuple[p.TrainingTask, p.CausalLimits], key: str, value: Any
):
    task, limits = saved
    record = json.loads(task.recipe_json)
    record["execution"][key] = value
    with pytest.raises(ValueError, match="hyperprior"):
        p.explain_generation(replace(task, recipe_json=json.dumps(record)), [0], limits=limits)


def test_valid_historical_provenance_and_strict_caller_limits(
    saved: tuple[p.TrainingTask, p.CausalLimits],
):
    task, limits = saved
    record = json.loads(task.recipe_json)
    record["execution"]["numpy"] = "historical-version"
    record["execution"]["cpu_threads"] = 3
    result = p.explain_generation(
        replace(task, recipe_json=json.dumps(record)), [15, 0, 15], limits=limits
    )
    assert result["signal_max_error"] == result["target_max_error"] == 0
    with pytest.raises(ValueError, match="limit|budget|parameters"):
        p.explain_generation(task, [0], limits=replace(limits, parameters=1))


@pytest.mark.parametrize(("task_type", "classes"), [("binary", 2), ("multiclass", 3)])
def test_classification_probabilities_and_full_metadata(
    saved: tuple[p.TrainingTask, p.CausalLimits],
    task_type: Literal["binary", "multiclass"],
    classes: int,
):
    original, limits = saved
    recipe = p.recipe_from_dict(json.loads(original.recipe_json)["recipe"], limits=limits)
    assert isinstance(recipe, (p.HyperpriorGraph, p.JointHyperpriorGraph))
    controls = replace(recipe.task, task_type=task_type, classes=classes)
    if isinstance(recipe, p.JointHyperpriorGraph):
        streams = [np.random.default_rng(i + 91) for i in range(5)]
        recipe = p.sample_joint_hyperprior_graph(
            replace(recipe.request, task=controls),
            topology_rng=streams[0],
            named_rng=streams[1],
            raw_rng=streams[2],
            calibration_rng=streams[3],
            fit_rng=streams[4],
        )
    else:
        recipe = recipe.with_task(controls)
    task = p.generate_task(recipe, seed=19, task_id=29)
    assert p.explain_generation(task, [15, 0, 15], limits=limits)["target_max_error"] == 0
    assert task.probabilities is not None
    wrong = task.probabilities.clone()
    wrong[-1] = 0
    for probabilities in (None, task.probabilities.float(), wrong):
        with pytest.raises(ValueError, match="hyperprior"):
            p.explain_generation(replace(task, probabilities=probabilities), [0], limits=limits)
    card = json.loads(task.mechanism_json)
    card["unknown"] = 1
    with pytest.raises(ValueError, match="hyperprior"):
        p.explain_generation(replace(task, mechanism_json=json.dumps(card)), [0], limits=limits)
    with pytest.raises(ValueError, match="hyperprior"):
        p.explain_generation(
            replace(
                task, target_origin="native-variable-v1", feature_schema=(p.TrainingFeature(),)
            ),
            [0],
            limits=limits,
        )
