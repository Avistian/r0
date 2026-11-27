"""Fitted mechanism provenance survives whole generated task archive round trips."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from r0.modules import generation as g
from r0.modules import priors as p


def fitted_graph():
    skeleton = p.FunctionalCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1),
        (
            p.FunctionalCausalNode(
                (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
            ),
            p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((1.0,), (2.0,)))),
        ),
        ((0, 0),),
        ((1, 0),),
    )
    fit = p.fit_xgboost(
        p.XGBoostFitPlan(input_width=1, output_width=2, rows=32, estimators=2),
        input_rng=np.random.default_rng(1),
        target_rng=np.random.default_rng(2),
        backend_rng=np.random.default_rng(3),
    )
    return p.with_xgboost_fits(skeleton, (p.XGBoostNodeFit(1, fit),))


def test_task_archive_keeps_fit_raw_and_native_export(tmp_path: Path):
    recipe = fitted_graph()
    reopened = p.recipe_from_dict(asdict(recipe))
    assert reopened == recipe
    task = p.generate_task(reopened, seed=17, task_id=2**54 + 11)
    raw = json.loads(task.recipe_json)["recipe"]
    assert raw["fits"][0]["realization"]["model_json"] == recipe.fits[0].realization.model_json
    path = tmp_path / "xgb-fit.zip"
    g.save_group(path, [(2**54 + 11, task)], max_bytes=4 * 1024**2)
    restored = next(g.load_group(path))
    again = p.generate_task(
        p.recipe_from_dict(json.loads(restored.recipe_json)["recipe"]), seed=17, task_id=2**54 + 11
    )
    torch.testing.assert_close(again.targets, task.targets, rtol=0, atol=0)
    trace = p.explain_generation(restored, [15, 0, 15])
    assert trace["signal_max_error"] == 0


def test_fit_binding_rejects_other_function():
    recipe = fitted_graph()
    node = replace(recipe.nodes[1], function=p.FrozenLinearFunction(((1.0,), (2.0,))))
    with pytest.raises(ValueError, match="binding"):
        replace(recipe, nodes=(recipe.nodes[0], node))


def test_fit_archive_rejects_missing_fit_custody():
    data = asdict(fitted_graph())
    del data["fits"][0]["realization"]["initial_rng"]
    with pytest.raises(ValueError, match="fields"):
        p.recipe_from_dict(data)


def test_graph_and_fit_aggregate_limit_precedes_any_snapshot_restore(
    monkeypatch: pytest.MonkeyPatch,
):
    recipe = fitted_graph()
    boundary = recipe.admit()["estimated_working_bytes"]
    limited = replace(recipe.limits, working_bytes=boundary - 1)
    calls: list[bool] = []

    def forbidden(*args: object, **kwargs: object):
        calls.append(True)
        raise AssertionError("aggregate graph/fit admission must precede snapshot restoration")

    monkeypatch.setattr(type(recipe.fits[0].realization.initial_rng[0]), "restore", forbidden)
    with pytest.raises(ValueError, match="working|limit"):
        p.recipe_from_dict(asdict(recipe), limits=limited)
    assert calls == []


def test_graph_exact_workspace_boundary_roundtrips():
    recipe = fitted_graph()
    limits = replace(recipe.limits, working_bytes=recipe.admit()["estimated_working_bytes"])
    bounded = replace(recipe, limits=limits)
    assert p.recipe_from_dict(asdict(bounded), limits=limits) == bounded


def test_fit_graph_emission_and_reopen_never_import_backend(monkeypatch: pytest.MonkeyPatch):
    import importlib

    recipe = fitted_graph()
    original = importlib.import_module

    def reject_backend(name: str, package: str | None = None):
        if name == "xgboost":
            raise AssertionError("frozen replay imported fitting backend")
        return original(name, package)

    monkeypatch.setattr(importlib, "import_module", reject_backend)
    restored = p.recipe_from_dict(asdict(recipe))
    assert isinstance(restored, p.XGBoostFittedGraph)
    first = p.generate_task(restored, seed=19, task_id=4)
    changed = replace(restored, task=replace(restored.task, rows=24))
    later = p.generate_task(changed, seed=19, task_id=4)
    assert asdict(changed.fits[0]) == asdict(recipe.fits[0])
    torch.testing.assert_close(first.inputs, later.inputs[:16], rtol=0, atol=0)
    torch.testing.assert_close(first.targets, later.targets[:16], rtol=0, atol=0)


def test_root_intervention_preserves_fitted_function_custody():
    recipe = fitted_graph()
    change = p.StageIntervention(0, root=p.JointRootBase("uniform", 1))
    pair = p.generate_causal_pair(recipe, interventions=(change,), seed=31, task_id=2**54 + 13)
    assert json.loads(pair.before.recipe_json)["recipe"] == json.loads(json.dumps(asdict(recipe)))
    reopened = p.recipe_from_dict(json.loads(pair.after.recipe_json)["recipe"])
    assert isinstance(reopened, p.CausalInterventionRecipe)
    trace = p.explain_generation(pair.after, [15, 0, 15])
    assert trace["rows"] == [15, 0, 15]
