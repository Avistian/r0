"""Guarded feature-axis convolution construction, custody and linked authored edits."""

import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from r0.modules import generation as g
from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession


def examples(session: ExplorerSession) -> dict[str, Any]:
    return {
        k: v
        for k, v in session.request("construction_examples", {})["examples"].items()
        if k.startswith("Feature convolution:")
    }


def construct(
    session: ExplorerSession, label: str = "sampled dense odd kernel (r0)"
) -> dict[str, Any]:
    return session.request(
        "construct",
        {
            "controls": examples(session)["Feature convolution: " + label],
            "construction_seed": "17",
            "seed": "37",
            "task_id": str(2**54 + 17),
        },
    )


def test_all_convolution_presets_preserve_source_and_typed_traces(tmp_path: Path) -> None:
    catalog = examples(ExplorerSession(tmp_path))
    assert len(catalog) == 17
    for name, controls in catalog.items():
        recipe = g.GraphConstruction(json.dumps(controls), 17).build(p.CausalLimits())
        assert p.recipe_from_dict(asdict(recipe)) == recipe
        task = p.generate_task(recipe, seed=37, task_id=2**54 + 17)
        trace = p.explain_generation(task, [31, 0, 31])
        assert trace["rows"] == [31, 0, 31]
        assert any("convolution-1-layer" in n["id"] for n in trace["nodes"])
        assert any("construction-draw" in n["id"] for n in trace["nodes"]) == (
            "authored" not in name
        )


def test_constructor_rejects_incompatible_or_overbudget_before_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = examples(ExplorerSession(tmp_path))
    controls = catalog["Feature convolution: sampled dense odd kernel (r0)"]

    def forbidden(*args: object, **kwargs: object):
        raise AssertionError("sampling began before aggregate construction admission")

    monkeypatch.setattr(p, "sample_feature_conv1d", forbidden)
    with pytest.raises(ValueError, match="limit|budget|cap"):
        g.GraphConstruction(json.dumps(controls), 17).build(
            p.CausalLimits(working_bytes=32 * 1024**2)
        )
    bad = deepcopy(controls)
    bad["nodes"][0]["request"]["layers"][0]["geometries"][0]["output_channels"] = 3
    with pytest.raises(ValueError, match="widths"):
        g.GraphConstruction(json.dumps(bad), 17).build(p.CausalLimits())
    bad = deepcopy(controls)
    bad["nodes"].append(deepcopy(bad["nodes"][0]))
    with pytest.raises(ValueError, match="unique"):
        g.GraphConstruction(json.dumps(bad), 17).build(p.CausalLimits())


def test_editor_trace_reference_and_frozen_next_stay_in_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = ExplorerSession(tmp_path)
    baseline = construct(session)
    recipe = p.recipe_from_dict(json.loads(baseline["recipe_json"])["recipe"])
    assert isinstance(recipe, p.FunctionalCausalRecipe)
    function = recipe.nodes[1].function
    assert isinstance(function, p.SampledFeatureConv1D)

    def forbidden(*args: object, **kwargs: object):
        raise AssertionError("sample replay escaped bounded worker")

    monkeypatch.setattr(type(function.initial[0]), "restore", forbidden)
    editor = session.request("mechanism", {})
    assert editor["convolution"]
    assert (
        editor["editable_recipe"]["nodes"][1]["function"]["version"]
        == "frozen-feature-conv1d-r0-v1"
    )
    assert editor["nodes"][1]["source_function"]["version"] == "sampled-feature-conv1d-r0-v1"
    assert session.request("graph", {"indices": [31, 0, 31]})["graph"]["rows"] == [31, 0, 31]
    session.request("curve", {"feature": 0, "points": 3})
    for action, data in (
        ("mechanism", {}),
        ("graph", {"indices": [0]}),
        ("convolution_draft", {"recipe": editor["editable_recipe"]}),
    ):
        with pytest.raises(ValueError, match="timed out"):
            session.request(action, {**data, "max_seconds": 0.001})
        assert session.request("view", {}) == baseline
    malformed = deepcopy(editor["editable_recipe"])
    malformed["nodes"][1]["function"] = {}
    with pytest.raises(ValueError, match="Convolution inspection failed"):
        session.request("convolution_draft", {"recipe": malformed})
    assert session.request("view", {}) == baseline
    advanced = session.request(
        "generate",
        {
            "record_json": baseline["recipe_json"],
            "frozen_identity": {"seed": "37", "task_id": str(2**54 + 18)},
        },
    )
    assert (
        json.loads(advanced["recipe_json"])["recipe"]
        == json.loads(baseline["recipe_json"])["recipe"]
    )


def test_authored_curve_literal_slope_and_linked_kernel_edit(tmp_path: Path) -> None:
    session = ExplorerSession(tmp_path)
    baseline = construct(session, "authored affine shared kernel")
    curve = session.request("curve", {"feature": 0, "anchor_row": 3, "points": 7})["curve"]
    np.testing.assert_allclose(
        np.diff(np.array(curve["scores"])[:, 0]), np.diff(curve["values"]), atol=1e-12
    )
    editor = session.request("mechanism", {})
    draft = editor["editable_recipe"]
    draft["nodes"][1]["function"]["layers"][0]["bias"] = [1.25]
    validated = session.request("convolution_draft", {"recipe": draft})
    saved = session.request("edit", {"recipe": validated["editable_recipe"]})
    assert saved["recipe_json"] == baseline["recipe_json"]
    assert saved["edit"]
    session.request("open_edit", {"path": saved["edit"]["path"]})
    assert session.request("view", {})["recipe_json"] == baseline["recipe_json"]


@pytest.mark.parametrize("kind", ["native", "vector", "converted"])
def test_typed_editor_and_explicit_pending_reference(tmp_path: Path, kind: str) -> None:
    session = ExplorerSession(tmp_path)
    baseline = construct(session, f"sampled {kind} graph")
    editor = session.request("mechanism", {})
    assert editor["native"] == (kind != "converted")
    assert editor["converted"] == (kind == "converted")
    session.request("convolution_draft", {"recipe": editor["editable_recipe"]})
    session.request("intervention_controls", {})
    with pytest.raises(ValueError, match="pending"):
        session.request("curve", {"feature": 0, "points": 3})
    assert session.request("view", {}) == baseline
