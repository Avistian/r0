"""Exact retained target geometry is checked before any row-owned source restoration."""

import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest

from r0.modules import generation as g
from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession


def binary_source(tmp_path: Path):
    controls = next(
        v
        for k, v in ExplorerSession(tmp_path)
        .request("construction_examples", {})["examples"]
        .items()
        if k.startswith("Population DAG: functional linear")
    )
    controls["request"]["base"]["task"].update(task_type="binary", classes=2)
    source = g.GraphConstruction(json.dumps(controls), 17).build(
        p.CausalLimits(), episode_seed=31, task_id=2**55 + 7
    )
    assert isinstance(source, p.RecursivePopulationGraph)
    return source


@pytest.mark.parametrize("field", ["scores", "target_classes"])
def test_binary_receipt_exact_geometry_precedes_replay(tmp_path: Path, field: str):
    source = binary_source(tmp_path)
    result = p.evaluate_population_graph_reference(source, row=7, feature=1, values=(-2.0, 2.0))
    raw = asdict(result)
    raw[field] = ((0.0, 0.0), (0.0, 0.0)) if field == "scores" else ("wrong", "labels")
    with (
        patch.object(
            type(source.bindings[0].sample.initial),
            "restore",
            side_effect=AssertionError("early restore"),
        ),
        pytest.raises(ValueError, match="reference|geometry|vocabulary"),
    ):
        p.population_graph_reference_from_dict(raw)


@pytest.mark.parametrize("family", ["functional", "native", "vector", "converted"])
@pytest.mark.parametrize("mode", ["original", "matched", "population"])
def test_effective_categorical_geometry_and_labels_before_restore(
    tmp_path: Path, family: str, mode: str
):
    controls = next(
        v
        for k, v in ExplorerSession(tmp_path)
        .request("construction_examples", {})["examples"]
        .items()
        if k.startswith(f"Population DAG: {family} vector scores and stochastic categories")
    )
    base = g.GraphConstruction(json.dumps(controls), 17).build(
        p.CausalLimits(), episode_seed=31, task_id=2**55 + 7
    )
    assert isinstance(base, p.RecursivePopulationGraph)
    source = base
    if mode != "original":
        cut = (
            p.NativeVectorHardIntervention(1, 0, category="b")
            if family == "vector"
            else p.ConverterHardIntervention(1, "label", "b", "capture")
            if family == "converted"
            else p.HardIntervention(1, category="b")
            if family == "native"
            else p.HardIntervention(1, (0,), (2.0,), location="state_and_readout")
        )
        source = p.intervene_recursive_population_graph(
            base, p.CausalInterventionRecipe(base.request.base, (cut,), mode=mode, population=7)
        )
    result = p.evaluate_population_graph_reference(source, row=7, feature=1, values=(-2.0, 2.0))
    assert len(result.scores[0]) == len(result.target_classes) == 3
    assert p.population_graph_reference_from_dict(asdict(result)) == result
    for field, bad in (
        ("scores", ((0.0,), (0.0,))),
        ("probabilities", ((1.0,), (1.0,))),
        ("target_classes", ("different", "unique", "labels")),
    ):
        raw = asdict(result)
        raw[field] = bad
        with (
            patch.object(
                type(base.bindings[0].sample.initial),
                "restore",
                side_effect=AssertionError("early restore"),
            ),
            pytest.raises(ValueError, match="reference|geometry|vocabulary"),
        ):
            p.population_graph_reference_from_dict(raw)


@pytest.mark.parametrize("family", ["native", "vector"])
@pytest.mark.parametrize("mode", ["original", "matched", "population"])
def test_threshold_scores_preserve_scalar_and_forced_indicator_semantics(
    tmp_path: Path, family: str, mode: str
):
    controls = next(
        v
        for k, v in ExplorerSession(tmp_path)
        .request("construction_examples", {})["examples"]
        .items()
        if k.startswith(f"Population DAG: {family} linear")
    )
    raw_base = controls["request"]["base"]
    raw_base["task"].update(task_type="multiclass", classes=3)
    domain = asdict(
        p.VariableDomain("categorical", ("low", "middle", "high"), "threshold", (-0.5, 0.5))
    )
    if family == "native":
        raw_base["nodes"][2]["domain"] = domain
    else:
        raw_base["nodes"][2]["domains"] = (domain,)
    base = g.GraphConstruction(json.dumps(controls), 17).build(
        p.CausalLimits(), episode_seed=31, task_id=2**55 + 7
    )
    assert isinstance(base, p.RecursivePopulationGraph)
    source = base
    if mode != "original":
        cut = (
            p.HardIntervention(2, category="middle")
            if family == "native"
            else p.NativeVectorHardIntervention(2, 0, category="middle")
        )
        source = p.intervene_recursive_population_graph(
            base, p.CausalInterventionRecipe(base.request.base, (cut,), mode=mode, population=7)
        )
    result = p.evaluate_population_graph_reference(source, row=7, feature=1, values=(-2.0, 2.0))
    width = 3 if family == "native" and mode != "original" else 1
    assert len(result.scores[0]) == width
    assert result.target_classes == ("low", "middle", "high")
    assert result.probabilities is not None
    assert len(result.probabilities[0]) == 3
    assert p.population_graph_reference_from_dict(asdict(result)) == result
    if mode != "original":
        assert result.targets == (1, 1)
        assert result.probabilities == ((0.0, 1.0, 0.0),) * 2
        assert result.scores == (((0.0, 1.0, 0.0) if family == "native" else (1.0,)),) * 2
    raw = asdict(result)
    raw["scores"] = ((0.0,) * (1 if width == 3 else 3),) * 2
    with (
        patch.object(
            type(base.bindings[0].sample.initial),
            "restore",
            side_effect=AssertionError("early restore"),
        ),
        pytest.raises(ValueError, match="reference|geometry"),
    ):
        p.population_graph_reference_from_dict(raw)
