"""Retained synthetic fit contracts must agree without rerunning the backend."""

import json
from dataclasses import replace

import numpy as np
import pytest

from r0.modules import priors


def fit(plan: priors.FittedTreePlan) -> priors.FittedTreeRealization:
    return priors.fit_fitted_tree(
        plan,
        input_rng=np.random.default_rng(51),
        target_rng=np.random.default_rng(52),
        backend_rng=np.random.default_rng(53),
    )


@pytest.fixture(scope="module", params=["dt", "rf", "extra_trees", "gbdt"])
def fitted(request: pytest.FixtureRequest) -> priors.FittedTreeRealization:
    return fit(
        priors.FittedTreePlan(
            family=request.param, task="classification", rows=32, classes=6, depth=2, estimators=2
        )
    )


def test_declared_backend_controls_are_bound(fitted: priors.FittedTreeRealization) -> None:
    fields = ["random_state", "max_depth"]
    family = fitted.plan.family
    if family != "gbdt":
        fields.append("criterion")
    fields += ["splitter"] if family == "dt" else ["n_estimators"]
    if family in ("rf", "extra_trees"):
        fields += ["n_jobs", "bootstrap", "max_features"]
    if family == "gbdt":
        fields += ["learning_rate", "loss", "init"]
    for field in fields:
        params = json.loads(fitted.backend_parameters[0])
        params[field] = "contradicts-declared-contract"
        with pytest.raises(ValueError, match="backend parameters"):
            replace(fitted, backend_parameters=(json.dumps(params),))
        del params[field]
        with pytest.raises(ValueError, match="backend parameters"):
            replace(fitted, backend_parameters=(json.dumps(params),))


def test_plan_controls_and_output_learning_rate_are_bound(
    fitted: priors.FittedTreeRealization,
) -> None:
    family = fitted.plan.family
    changes: dict[str, str | int | float] = {
        "depth": 3,
        "classes": 7,
        "task": "regression",
        "classification_rule": "majority",
    }
    if family == "dt":
        changes["splitter"] = "best"
    else:
        changes["estimators"] = 3
    if family in ("rf", "extra_trees"):
        changes["family"] = "extra_trees" if family == "rf" else "rf"
    if family == "gbdt":
        del changes["classification_rule"]
        changes["learning_rate"] = 0.9
    for key, value in changes.items():
        with pytest.raises(ValueError, match="differ"):
            replace(fitted, plan=replace(fitted.plan, **{key: value}))
    if family == "gbdt":
        output = replace(fitted.function.outputs[0], learning_rate=0.9)
        with pytest.raises(ValueError, match="learning rate"):
            replace(fitted, function=replace(fitted.function, outputs=(output,)))


def test_observed_classes_match_retained_targets() -> None:
    fitted = fit(priors.FittedTreePlan(task="classification", rows=32, classes=100))
    output = fitted.function.outputs[0]
    absent = next(c for c in range(100) if c not in output.classes)
    changed = tuple(sorted((absent, *output.classes[1:])))
    assert len(changed) == len(output.classes)
    with pytest.raises(ValueError, match="class map"):
        replace(
            fitted, function=replace(fitted.function, outputs=(replace(output, classes=changed),))
        )


def test_frozen_depth_cannot_exceed_declared_depth() -> None:
    fitted = fit(priors.FittedTreePlan(rows=32, depth=3, splitter="best"))
    assert len(fitted.function.outputs[0].trees[0].features) > 3
    params = json.loads(fitted.backend_parameters[0])
    params["max_depth"] = 1
    with pytest.raises(ValueError, match="depth"):
        replace(
            fitted, plan=replace(fitted.plan, depth=1), backend_parameters=(json.dumps(params),)
        )


def test_public_graph_decoder_rejects_contradictory_fit_plan() -> None:
    from dataclasses import asdict

    linear = priors.FrozenLinearFunction(((1.0,),))
    skeleton = priors.FunctionalCausalRecipe(
        priors.LinearRecipe(
            rows=16, context_rows=8, features=1, signal_features=1, active_features=1
        ),
        (
            priors.FunctionalCausalNode((), linear, priors.JointRootBase("gaussian", 1)),
            priors.FunctionalCausalNode((0,), linear),
        ),
        ((0, 0),),
        ((1, 0),),
    )
    streams = [np.random.default_rng(i + 56) for i in range(4)]
    graph = priors.sample_fitted_graph(
        skeleton,
        ((1, priors.FittedTreePlan(family="gbdt", rows=32, depth=1, estimators=2)),),
        prefix_rng=streams[0],
        input_rng=streams[1],
        target_rng=streams[2],
        backend_rng=streams[3],
    )
    data = json.loads(json.dumps(asdict(graph)))
    assert priors.recipe_from_dict(data) == graph
    data["fits"][0]["realization"]["plan"]["learning_rate"] = 0.9
    with pytest.raises(ValueError, match="backend parameters"):
        priors.recipe_from_dict(data)
