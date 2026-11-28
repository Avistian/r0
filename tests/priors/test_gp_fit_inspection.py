"""Independent fitting populations retain their own rows and covariance identities."""

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from r0.modules import priors as p


def calibrated(family: str = "functional"):
    request = p.GPColumnRequest(rows=16, kernels=(p.GPColumnKernel(length_scale=0.1),))
    source = p.realize_gp_columns(request, seed=61, task_id=2**55 + 11)
    task = p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2)
    root = p.FunctionalCausalNode(
        (), p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))), p.JointRootBase("gaussian", 2)
    )
    child = p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0, -1.0),)))
    if family == "functional":
        base = p.FunctionalCausalRecipe(task, (root, child), ((0, 0), (0, 1)), ((1, 0),))
    elif family == "native":
        base = p.NativeCausalRecipe(
            task, (p.NativeCausalNode(root), p.NativeCausalNode(child)), ((0, 0), (0, 1)), (1, 0)
        )
    elif family == "vector":
        base = p.NativeVectorRecipe(
            task,
            (
                p.NativeVectorNode(root, (p.VariableDomain(),) * 2),
                p.NativeVectorNode(child, (p.VariableDomain(),)),
            ),
            ((0, 0), (0, 1)),
            (1, 0),
        )
    else:
        base = p.ConvertedCausalRecipe(
            task,
            (p.ConvertedCausalNode(root), p.ConvertedCausalNode(child)),
            (p.ConvertedReadout(0, channel=0), p.ConvertedReadout(0, channel=1)),
            p.ConvertedReadout(1, channel=0),
        )
    original = p.GPRootDAGRecipe(source, base, (p.GPRootBinding(0, (0, 1)),))
    fit = p.calibrate_gp_root_dag(
        original, calibration_rows=32, calibration_seed=67, calibration_task_id=2**55 + 13
    )
    return fit, p.generate_task(fit, seed=61, task_id=2**55 + 11)


@pytest.mark.parametrize("family", ["functional", "native", "vector", "converted"])
def test_fit_trace_uses_fit_population_and_all_fit_statistics(family: str):
    recipe, task = calibrated(family)
    graph = p.explain_gp_calibration_fit(task, [31, 2, 31])
    assert graph["rows"] == [31, 2, 31]
    assert graph["episode_id"] == task.episode_id
    assert graph["population_role"] == "fit"
    assert graph["population_id"] == recipe.fit.source.population_id
    assert graph["population_id"] != recipe.original.source.population_id
    assert len(graph["shared_matrices"][0]["covariance"]) == 32
    nodes = {n["id"]: n for n in graph["nodes"]}
    values = np.asarray(recipe.fit.source.draw.values)
    np.testing.assert_allclose(nodes["gp-population"]["values"], values[[31, 2, 31]], atol=1e-12)
    np.testing.assert_allclose(nodes["gp-binding-0"]["values"], values[[31, 2, 31]], atol=1e-12)
    signal = 2 * values[:, 0] - values[:, 1]
    np.testing.assert_allclose(
        nodes["fit-statistics"]["values"], [[signal.mean()], [signal.std()]], atol=1e-12
    )
    assert nodes["fit-statistics"]["parameters"]["rows"] == 32
    ids = set(nodes)
    assert all(parent in ids for node in nodes.values() for parent in node["parents"])


@pytest.mark.parametrize("indices", [[32], [-1], [True], [], list(range(17))])
def test_bad_fit_rows_reject_before_replay(indices: list[Any], monkeypatch: pytest.MonkeyPatch):
    _, task = calibrated()

    def forbidden(*args: object, **kwargs: object):
        pytest.fail("invalid fit row selection reconstructed covariance")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    with pytest.raises(ValueError, match="fit.*rows|fit.*index"):
        p.explain_gp_calibration_fit(task, indices)


def test_forged_query_task_is_not_accepted_for_fit_inspection():
    _, task = calibrated()
    with pytest.raises(ValueError, match="targets|tensor|differ"):
        p.explain_gp_calibration_fit(replace(task, targets=task.targets + 1), [31, 2, 31])


def test_guarded_fit_trace_preserves_active_query(tmp_path: Path):
    from r0.modules.explorer import ExplorerSession

    recipe, task = calibrated()
    session = ExplorerSession(tmp_path)
    original = session.request(
        "generate",
        {"recipe": asdict(recipe), "seed": 61, "task_id": 2**55 + 11, "max_seconds": 120},
    )
    graph = session.request(
        "graph", {"indices": [31, 2, 31], "population": "fit", "max_seconds": 120}
    )["graph"]
    assert graph["population_id"] == recipe.fit.source.population_id
    assert int(graph["fit_task_id"]) == recipe.fit.source.task_id
    assert graph["rows"] == [31, 2, 31]
    assert session.task is not None
    assert session.task.episode_id == task.episode_id
    assert session.request("view", {})["view"]["episode_id"] == original["view"]["episode_id"]


def test_query_and_fit_statistics_have_global_named_axes():
    _, task = calibrated()
    for explain in (p.explain_generation, p.explain_gp_calibration_fit):
        graph = explain(task, [9, 9])
        node = next(n for n in graph["nodes"] if n["id"] == "fit-statistics")
        assert node["parameters"]["row_labels"] == ["center", "scale"]
        assert node["parameters"]["row_heading"] == "Statistic"
        assert "all 32" in node["parameters"]["axes"]
