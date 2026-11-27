"""Outer population interventions reject invalid semantics before any receipt restore."""

import copy
from dataclasses import asdict
from typing import Any

import pytest

from r0.modules import priors as p


@pytest.fixture
def population_packet() -> tuple[dict[str, Any], type[Any]]:
    task = p.LinearRecipe(rows=16, context_rows=7, features=2, signal_features=2, active_features=2)
    root = p.FunctionalCausalNode(
        (), p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))), p.JointRootBase("gaussian", 2)
    )
    child = p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((1.0, 2.0),)))
    base = p.FunctionalCausalRecipe(task, (root, child), ((0, 0), (0, 1)), ((1, 0),))
    forest = p.RecursiveForestPlan(
        input_width=2,
        estimators=1,
        min_depth=1,
        max_depth=1,
        lifecycle="fresh-forest-per-row",
        populations=16,
    )
    pop = p.realize_recursive_population_graph(
        p.RecursivePopulationGraphRequest(base, (p.RecursiveNodePopulationPlan(1, forest),)),
        seed=19,
        task_id=23,
    )
    change = p.CausalInterventionRecipe(
        base, (p.HardIntervention(0, (0, 1), (1.0, -2.0)),), mode="population", population=3
    )
    return asdict(p.intervene_recursive_population_graph(pop, change)), type(
        pop.bindings[0].sample.initial
    )


@pytest.mark.parametrize(
    "case",
    [
        "mode",
        "population-bool",
        "population-zero",
        "outer-version",
        "change-version",
        "empty",
        "duplicate",
        "channel",
        "category",
        "location",
        "values",
        "soft-width",
        "soft-empty",
        "stage-root",
        "samples-missing",
        "samples-extra",
        "input-node",
        "input-width",
        "input-count",
        "binding-version",
        "baseline-version",
        "sample-version",
        "sample-plan",
        "input-nonfinite",
        "outer-field",
        "seed-bool",
    ],
)
def test_outer_intervention_rejects_before_restore(
    case: str, population_packet: tuple[dict[str, Any], type[Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    packet, cls = population_packet
    raw = copy.deepcopy(packet)
    change = raw["change"]
    interventions = list(change["interventions"])
    if case == "mode":
        change["mode"] = "invalid"
    elif case == "population-bool":
        change["population"] = True
    elif case == "population-zero":
        change["population"] = 0
    elif case == "outer-version":
        raw["version"] = "invalid"
    elif case == "change-version":
        change["version"] = "invalid"
    elif case == "empty":
        change["interventions"] = []
    elif case == "duplicate":
        change["interventions"] = interventions * 2
    elif case == "channel":
        interventions[0]["channels"] = (0, 5)
    elif case == "category":
        interventions[0].update(channels=(), values=(), category="wrong")
    elif case == "location":
        interventions[0]["location"] = "invalid"
    elif case == "values":
        interventions[0]["values"] = (1.0,)
    elif case == "soft-width":
        change["interventions"] = [
            asdict(p.SoftIntervention(1, function=p.FrozenLinearFunction(((1.0,),))))
        ]
    elif case == "soft-empty":
        change["interventions"] = [
            {
                "node": 1,
                "function": None,
                "noise_scale": None,
                "noise_response": None,
                "exogenous": None,
                "version": "soft-causal-intervention-v1",
            }
        ]
    elif case == "stage-root":
        change["interventions"] = [
            asdict(p.StageIntervention(1, root=p.JointRootBase("gaussian", 2)))
        ]
    elif case == "samples-missing":
        raw["samples"] = []
    elif case == "samples-extra":
        raw["samples"] = list(raw["samples"]) * 2
    elif case == "input-node":
        raw["inputs"] = [(0, raw["inputs"][0][1])]
    elif case == "input-width":
        raw["inputs"] = [(1, ((1.0,),) * 16)]
    elif case == "input-count":
        raw["inputs"] = []
    elif case == "binding-version":
        raw["base"]["bindings"][0]["version"] = "invalid"
    elif case == "baseline-version":
        raw["base"]["version"] = "invalid"
    elif case == "sample-version":
        raw["samples"][0]["version"] = "invalid"
    elif case == "sample-plan":
        raw["samples"][0]["plan"]["min_depth"] = 0
    elif case == "input-nonfinite":
        raw["inputs"] = [(1, ((float("inf"), 1.0),) * 16)]
    elif case == "outer-field":
        raw["extra"] = True
    elif case == "seed-bool":
        raw["base"]["seed"] = True
    restores: list[int] = []
    original = cls.restore

    def tracked(self: Any, *args: Any, **kwargs: Any) -> Any:
        restores.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(cls, "restore", tracked)
    with pytest.raises(ValueError, match="."):
        p.recipe_from_dict(raw)
    assert restores == []


def test_sampled_root_and_function_preview_preserves_original_custody(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np
    import torch

    task = p.LinearRecipe(rows=16, context_rows=7, features=2, signal_features=2, active_features=2)
    bank = p.sample_prototype_root(
        p.PrototypeRootPlan(p.JointRootBase("gaussian", 2), 32, 0.125),
        rng=np.random.default_rng(21),
        limits=p.CausalLimits(parameters=10000, working_bytes=32 * 1024**2),
    )
    source_plan = p.RecursiveForestPlan(input_width=2, estimators=1, min_depth=1, max_depth=1)
    sampled = p.sample_recursive_forest(
        source_plan,
        rng=np.random.default_rng(22),
        limits=p.CausalLimits(parameters=10000, working_bytes=32 * 1024**2),
    )
    replacement = p.sample_recursive_forest(
        source_plan,
        rng=np.random.default_rng(23),
        limits=p.CausalLimits(parameters=10000, working_bytes=32 * 1024**2),
    )
    root = p.FunctionalCausalNode(
        (),
        p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))),
        p.PrototypeRootBase(bank),
        version="functional-causal-node-v2",
    )
    child = p.FunctionalCausalNode((0,), p.FrozenRecursiveForestFunction(sampled))
    base = p.FunctionalCausalRecipe(task, (root, child), ((0, 0), (0, 1)), ((1, 0),))
    population_plan = p.RecursiveForestPlan(
        input_width=2,
        estimators=1,
        min_depth=1,
        max_depth=1,
        lifecycle="fresh-forest-per-row",
        populations=16,
    )
    pop = p.realize_recursive_population_graph(
        p.RecursivePopulationGraphRequest(
            base, (p.RecursiveNodePopulationPlan(1, population_plan),)
        ),
        seed=19,
        task_id=23,
    )
    changed = p.intervene_recursive_population_graph(
        pop,
        p.CausalInterventionRecipe(
            base, (p.SoftIntervention(1, function=p.FrozenRecursiveForestFunction(replacement)),)
        ),
    )
    raw = asdict(changed)
    reopened = p.recipe_from_dict(raw)
    assert asdict(reopened) == raw
    original = p.generate_task(changed, seed=19, task_id=23)
    replay = p.generate_task(reopened, seed=19, task_id=23)
    assert original.recipe_json == replay.recipe_json
    assert torch.equal(original.targets, replay.targets)
    raw["change"]["interventions"] = list(raw["change"]["interventions"]) * 2
    cls = type(sampled.initial)
    calls: list[int] = []
    restore = cls.restore

    def tracked(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return restore(self, *args, **kwargs)

    monkeypatch.setattr(cls, "restore", tracked)
    with pytest.raises(ValueError, match="overlap"):
        p.recipe_from_dict(raw)
    assert calls == []
