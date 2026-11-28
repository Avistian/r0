"""Sampled parameter provenance survives generated task archives and atomic failures."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from r0.modules import generation as g
from r0.modules import priors as p


def skeleton() -> p.FunctionalCausalRecipe:
    return p.FunctionalCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1),
        (
            p.FunctionalCausalNode(
                (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
            ),
            p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((1.0,),))),
        ),
        ((0, 0),),
        ((1, 0),),
    )


def sampled() -> p.SampledXGBoostGraph:
    return p.sample_parameterized_xgboost_graph(
        skeleton(),
        ((1, p.XGBoostParameterPrior(p.XGBoostFitPlan(input_width=1, output_width=1, rows=32))),),
        parameter_rng=np.random.default_rng(1),
        input_rng=np.random.default_rng(2),
        target_rng=np.random.default_rng(3),
        backend_rng=np.random.default_rng(4),
    )


def test_sampled_xgboost_graph_task_save_replay(tmp_path: Path) -> None:
    graph = sampled()
    assert graph.parameter_samples[0][1].plan == graph.fits[0].realization.plan
    restored = p.recipe_from_dict(json.loads(json.dumps(asdict(graph))))
    assert restored == graph
    task = p.generate_task(restored, seed=28, task_id=2**54 + 7)
    path = tmp_path / "sampled-xgboost.zip"
    g.save_group(path, [(2**54 + 7, task)], max_bytes=4 * 1024**2)
    saved = next(g.load_group(path))
    metadata = json.loads(saved.recipe_json)
    assert metadata["recipe"]["parameter_samples"][0][1] == json.loads(
        json.dumps(asdict(graph.parameter_samples[0][1]))
    )
    again = p.generate_task(p.recipe_from_dict(metadata["recipe"]), seed=28, task_id=2**54 + 7)
    torch.testing.assert_close(again.targets, task.targets, rtol=0, atol=0)
    assert p.explain_generation(saved, [15, 0, 15])["signal_max_error"] == 0


def test_sampled_graph_binding_and_exact_workspace_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = sampled()
    boundary = graph.admit()["estimated_working_bytes"]
    bounded = replace(graph, limits=replace(graph.limits, working_bytes=boundary))
    assert p.recipe_from_dict(asdict(bounded), limits=bounded.limits) == bounded
    from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("restored before aggregate admission")

    monkeypatch.setattr(GeneratorSnapshot, "restore", forbidden)
    with pytest.raises(ValueError, match="limit|working"):
        p.recipe_from_dict(asdict(graph), limits=replace(graph.limits, working_bytes=boundary - 1))


def test_sampled_graph_backend_failure_rolls_back_all_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import r0.modules.priors.lib.sampled_xgboost_graph as module

    streams = {
        name: np.random.default_rng(i)
        for i, name in enumerate(("parameter_rng", "input_rng", "target_rng", "backend_rng"))
    }
    before = json.dumps({k: v.bit_generator.state for k, v in streams.items()})

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected fitting failure")

    monkeypatch.setattr(module, "fit_xgboost", fail)
    with pytest.raises(RuntimeError, match="injected"):
        p.sample_parameterized_xgboost_graph(
            skeleton(),
            (
                (
                    1,
                    p.XGBoostParameterPrior(
                        p.XGBoostFitPlan(input_width=1, output_width=1, rows=32)
                    ),
                ),
            ),
            **streams,
        )
    assert json.dumps({k: v.bit_generator.state for k, v in streams.items()}) == before


def test_two_sampled_fits_preserve_stream_chain_and_aggregate_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import r0.modules.priors.lib.sampled_xgboost_graph as module

    base = skeleton()
    base = replace(
        base,
        nodes=base.nodes + (p.FunctionalCausalNode((1,), p.FrozenLinearFunction(((1.0,),))),),
        target=((2, 0),),
    )
    prior = p.XGBoostParameterPrior(p.XGBoostFitPlan(input_width=1, output_width=1, rows=32))
    streams = {
        name: np.random.default_rng(i)
        for i, name in enumerate(("parameter_rng", "input_rng", "target_rng", "backend_rng"))
    }
    graph = p.sample_parameterized_xgboost_graph(base, ((1, prior), (2, prior)), **streams)
    assert graph.parameter_samples[0][1].final_rng == graph.parameter_samples[1][1].initial_rng
    assert p.recipe_from_dict(asdict(graph)) == graph
    limited = replace(base, limits=replace(base.limits, parameters=2500))
    prior.admit(limited.limits)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("fit before aggregate parameter admission")

    monkeypatch.setattr(module, "fit_xgboost", forbidden)
    with pytest.raises(ValueError, match="parameter limit"):
        p.sample_parameterized_xgboost_graph(limited, ((1, prior), (2, prior)), **streams)


def test_replay_without_backend_and_root_intervention_keeps_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    graph = sampled()
    original = importlib.import_module

    def reject_backend(name: str, package: str | None = None):
        if name == "xgboost":
            raise AssertionError("replay must not fit")
        return original(name, package)

    monkeypatch.setattr(importlib, "import_module", reject_backend)
    restored = p.recipe_from_dict(asdict(graph))
    assert isinstance(restored, p.SampledXGBoostGraph)
    pair = p.generate_causal_pair(
        restored,
        interventions=(p.StageIntervention(0, root=p.JointRootBase("uniform", 1)),),
        seed=28,
        task_id=17,
    )
    assert (
        json.loads(pair.before.recipe_json)["recipe"]["parameter_samples"]
        == json.loads(json.dumps(asdict(graph)))["parameter_samples"]
    )
    assert p.explain_generation(pair.after, [15, 0, 15])["rows"] == [15, 0, 15]
