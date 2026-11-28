"""Guarded retained GP curves preserve the dataset and exact reference receipt."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession


def gp_root_recipe(family: str):
    population = p.realize_gp_columns(p.GPColumnRequest(rows=16), seed=47, task_id=59)
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
                p.NativeVectorNode(root, (p.VariableDomain(), p.VariableDomain())),
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
    return p.GPRootDAGRecipe(population, base, (p.GPRootBinding(0, (0, 1)),))


@pytest.mark.parametrize("family", ["functional", "native", "vector", "converted"])
def test_gp_root_reference_guarded_session(family: str, tmp_path: Path):
    original = gp_root_recipe(family)
    source = p.calibrate_gp_root_dag(
        original, calibration_rows=32, calibration_seed=101, calibration_task_id=103
    )
    session = ExplorerSession(tmp_path)
    session.task = p.generate_task(source, seed=47, task_id=59)
    before = session.task.recipe_json
    result = session.request(
        "curve",
        {
            "feature": 0,
            "anchor_row": 3,
            "points": 3,
            "inputs": [[1.0, 2.0], [1.0, 2.0], [-1.0, 0.0]],
            "reference": "zero",
        },
    )["curve"]
    receipt = p.gp_root_reference_from_dict(json.loads(result["reference_json"]))
    assert receipt.inputs == ((1.0, 2.0), (1.0, 2.0), (-1.0, 0.0))
    raw = np.asarray([0.0, 0.0, -2.0])
    expected = (raw - source.calibration.center[0]) / source.calibration.scale[0]
    np.testing.assert_allclose(receipt.targets, expected, atol=1e-12)
    assert session.task.recipe_json == before
    assert json.loads(receipt.source_json) == json.loads(json.dumps(asdict(source)))


@pytest.mark.parametrize("bad", [[[True, 2.0]], [[0.5]], [[float("inf"), 1.0]], [], {"bad": 1}])
def test_malformed_reference_worker_preserves_active_task(bad: object, tmp_path: Path):
    session = ExplorerSession(tmp_path)
    session.task = p.generate_task(gp_root_recipe("functional"), seed=47, task_id=59)
    before = session.task
    with pytest.raises(ValueError, match="GP reference|JSON"):
        session.request("curve", {"feature": 0, "anchor_row": 0, "points": 3, "inputs": bad})
    assert session.task is before
    assert session.archive is None


def test_gp_reference_shared_notebook_callback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import sys
    from types import ModuleType
    from typing import Any

    from r0.modules.explorer import show_colab_explorer

    callbacks: dict[str, Any] = {}
    shown: list[str] = []
    output = ModuleType("google.colab.output")
    display = ModuleType("IPython.display")

    def register(name: str, callback: Any) -> None:
        callbacks[name] = callback

    def wrap(value: Any) -> dict[str, Any]:
        return {"application/json": value}

    monkeypatch.setattr(output, "register_callback", register, raising=False)
    monkeypatch.setattr(display, "JSON", wrap, raising=False)
    monkeypatch.setattr(display, "HTML", str, raising=False)
    monkeypatch.setattr(display, "display", shown.append, raising=False)
    monkeypatch.setitem(sys.modules, "google.colab.output", output)
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    invoke = callbacks[show_colab_explorer(tmp_path)]
    assert 'id="gp-root-reference-controls"' in shown[0]
    controls = invoke("construction_examples", {})["application/json"]["examples"][
        "GP root DAG: functional linear / regression"
    ]
    original = invoke(
        "construct", {"controls": controls, "seed": "7", "task_id": "9", "construction_seed": "0"}
    )["application/json"]
    result = invoke(
        "curve", {"feature": 0, "anchor_row": 1, "points": 3, "inputs": [[1.0, 2.0], [-1.0, 0.0]]}
    )["application/json"]["curve"]
    raw = json.loads(result["reference_json"])
    np.testing.assert_allclose(raw["raw_scores"], [[0.0], [-2.0]], atol=1e-12)
    assert (
        invoke("view", {})["application/json"]["view"]["episode_id"]
        == original["view"]["episode_id"]
    )
