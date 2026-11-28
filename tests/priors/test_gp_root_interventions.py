"""GP-root comparisons retain joint draws and independent target fits."""

import json
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch
from gp_root_fixture import gp_root_recipe

from r0.modules import priors as p


@pytest.fixture(scope="module", params=["functional", "native", "vector", "converted"])
def source(request: pytest.FixtureRequest):
    original = gp_root_recipe(request.param)
    calibration = p.calibrate_gp_root_dag(
        original, calibration_rows=32, calibration_seed=61, calibration_task_id=67
    )
    query = p.realize_gp_root_calibration_query(calibration, seed=71, task_id=73)
    return original, calibration, query


@pytest.mark.parametrize("mode", ["matched", "population"])
@pytest.mark.parametrize("lifecycle", [0, 1, 2])
def test_gp_interventions_keep_full_source_and_fixed_fit(
    source: tuple[p.GPRootDAGRecipe, p.GPRootCalibration, p.GPRootCalibrationQuery],
    mode: str,
    lifecycle: int,
):
    base = source[lifecycle]
    current = base if isinstance(base, p.GPRootDAGRecipe) else base.original
    change = p.CausalInterventionRecipe(
        current.base,
        (p.SoftIntervention(1, function=p.FrozenLinearFunction(((-1.0, 4.0),))),),
        mode=mode,
        population=11,
    )
    result = p.intervene_gp_root_dag(base, change)
    task = p.generate_task(result, seed=current.source.seed, task_id=current.source.task_id)
    values = np.asarray(result.population.draw.values)
    expected = -values[:, 0] + 4 * values[:, 1]
    if not isinstance(base, p.GPRootDAGRecipe):
        assert not isinstance(result.base, p.GPRootDAGRecipe)
        expected = (expected - base.calibration.center[0]) / base.calibration.scale[0]
        assert asdict(result.base.fit) == asdict(base.fit)
        assert asdict(result.base.calibration) == asdict(base.calibration)
    np.testing.assert_allclose(task.scores[:, 0], expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(task.targets, expected, rtol=0, atol=1e-12)
    assert asdict(result.base) == asdict(base)
    if mode == "matched":
        assert asdict(result.population) == asdict(current.source)
    else:
        assert result.population.population_id != current.source.population_id
        assert asdict(result.population.request) == asdict(current.source.request)
    decoded = p.recipe_from_dict(json.loads(task.recipe_json)["recipe"], limits=result.limits)
    replay = p.generate_task(decoded, seed=current.source.seed, task_id=current.source.task_id)
    assert torch.equal(replay.inputs, task.inputs)
    assert torch.equal(replay.targets, task.targets)
    assert replay.recipe_json == task.recipe_json
    graph = p.explain_generation(task, [13, 2, 13], limits=result.limits)
    assert graph["rows"] == [13, 2, 13]
    assert graph["population_id"] == result.population.population_id
    nodes = {node["id"]: node for node in graph["nodes"]}
    np.testing.assert_allclose(nodes["targets"]["values"], task.targets[[13, 2, 13], None])
    assert all(parent in nodes for node in nodes.values() for parent in node["parents"])


def test_gp_intervention_wrong_graph_rejects_before_replay(
    source: tuple[p.GPRootDAGRecipe, p.GPRootCalibration, p.GPRootCalibrationQuery],
    monkeypatch: pytest.MonkeyPatch,
):
    base = source[1]
    change = p.CausalInterventionRecipe(
        source[0].base,
        (p.SoftIntervention(1, function=p.FrozenLinearFunction(((-1.0, 4.0),))),),
    )
    raw = asdict(change)
    raw["base"]["task"]["context_rows"] = 7
    wrong = p.recipe_from_dict(raw)
    assert isinstance(wrong, p.CausalInterventionRecipe)

    def forbidden(*args: object, **kwargs: object):
        pytest.fail("early replay")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    with pytest.raises(ValueError, match="original|base|graph"):
        p.intervene_gp_root_dag(base, wrong)


def test_gp_hard_pair_preserves_unaffected_column_and_fixed_fit(
    source: tuple[p.GPRootDAGRecipe, p.GPRootCalibration, p.GPRootCalibrationQuery],
):
    base = source[2]
    intervention = (
        p.NativeVectorHardIntervention(0, 0, value=2.5)
        if isinstance(base.original.base, p.NativeVectorRecipe)
        else p.HardIntervention(
            0,
            (0,),
            (2.5,),
            location="state"
            if isinstance(base.original.base, p.NativeCausalRecipe)
            else "state_and_readout",
        )
    )
    pair = p.generate_gp_root_pair(
        base, p.CausalInterventionRecipe(base.original.base, (intervention,))
    )
    expected = (5 - pair.before.inputs[:, 1] - base.calibration.center[0]) / base.calibration.scale[
        0
    ]
    torch.testing.assert_close(pair.after.targets, expected)
    assert torch.equal(pair.after.inputs[:, 1], pair.before.inputs[:, 1])
    assert torch.all(pair.after.inputs[:, 0] == 2.5)
    card = p.compare_causal_pair(pair.before, pair.after)
    assert card["query_fit"] is False
    assert card["original_population_id"] == card["intervention_population_id"]
    with pytest.raises(ValueError, match="tensor|targets|differ"):
        p.compare_gp_root_pair(pair.before, replace(pair.after, targets=pair.after.targets + 1))
    graph = p.explain_generation(pair.after, [9, 9])
    nodes = {node["id"]: node for node in graph["nodes"]}
    assert nodes["fit-statistics"]["parameters"]["row_labels"] == ["center", "scale"]
    assert all(parent in nodes for node in nodes.values() for parent in node["parents"])


def test_gp_full_root_bypass_uses_clamps_not_retained_values(
    source: tuple[p.GPRootDAGRecipe, p.GPRootCalibration, p.GPRootCalibrationQuery],
):
    base = source[0]
    changes = (
        (
            p.NativeVectorHardIntervention(0, 0, value=2.5),
            p.NativeVectorHardIntervention(0, 1, value=-0.5),
        )
        if isinstance(base.base, p.NativeVectorRecipe)
        else (
            p.HardIntervention(
                0,
                (0, 1),
                (2.5, -0.5),
                location="state"
                if isinstance(base.base, p.NativeCausalRecipe)
                else "state_and_readout",
            ),
        )
    )
    altered = p.intervene_gp_root_dag(base, p.CausalInterventionRecipe(base.base, changes))
    task = p.generate_task(altered, seed=47, task_id=59)
    assert torch.all(task.targets == 5.5)
    graph = p.explain_generation(task, [15, 0, 15])
    nodes = {node["id"]: node for node in graph["nodes"]}
    assert all(parent in nodes for node in nodes.values() for parent in node["parents"])
