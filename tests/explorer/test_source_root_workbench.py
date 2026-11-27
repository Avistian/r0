"""Explicit source-root provenance and guarded independent normalization controls."""

import json
from pathlib import Path
from typing import Any

import pytest

from r0.modules.explorer import ExplorerSession


def test_root_presets_expose_all_coordinate_and_architecture_options(tmp_path: Path):
    examples = ExplorerSession(tmp_path).request("construction_examples", {})["examples"]
    roots = [v for v in examples.values() if v.get("topology") == "source-root-network"]
    assert len(roots) == 107
    assert {v["source"]["network_prior"]["families"][0] for v in roots} == {
        "dense",
        "individual-sparse",
        "block-sparse",
        "skip",
        "residual",
        "irregular",
        "gated",
    }
    assert {v["profile"]["selector"] for v in roots} == {
        "p1-code-normal",
        "p1-code-uniform",
        "p1-code-mixed",
        "p2-node-mixture-r0",
        "forced-r0",
    }
    assert {v["modes"][0] for v in roots} == {"identity", "center", "standardize"}


def test_profile_root_constructs_frozen_archive(tmp_path: Path):
    session = ExplorerSession(tmp_path)
    controls = session.request("construction_examples", {})["examples"][
        "Source root: forced unit_uniform (r0) / dense"
    ]
    result = session.request(
        "construct",
        {"controls": controls, "construction_seed": 7, "seed": "11", "task_id": str(2**54 + 3)},
    )
    root = json.loads(result["recipe_json"])["recipe"]["root"]
    assert root["version"] == "sampled-source-root-base-v1"
    assert root["calibration_rows"] == 0
    assert root["calibration"] is None


def test_source_root_editor_replay_is_guarded_and_failure_preserves_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from dataclasses import asdict

    from r0.modules import priors as p

    session = ExplorerSession(tmp_path)
    controls = session.request("construction_examples", {})["examples"]["Standalone network: dense"]
    controls["task"].update(rows=16, context_rows=8)
    controls["root"] = asdict(
        p.SourceRootBase(
            p.SourceRootLaw(
                (
                    p.SourceRootCoordinate("gaussian", location=3),
                    p.SourceRootCoordinate("unit_uniform"),
                )
            )
        )
    )
    baseline = session.request(
        "construct",
        {
            "controls": controls,
            "construction_seed": "5",
            "seed": "7",
            "task_id": "9007199254741001",
        },
    )
    original = p.recipe_from_dict

    def parent_decode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("expensive source-root decode ran in parent")

    monkeypatch.setattr(p, "recipe_from_dict", parent_decode)
    monkeypatch.setattr(p, "explain_generation", parent_decode)
    graph = session.request("graph", {"indices": [15, 0, 15]})["graph"]
    assert graph["rows"] == [15, 0, 15]
    with pytest.raises(ValueError, match="timed out"):
        session.request("graph", {"indices": [0], "max_seconds": 1e-9})
    editor = session.request("mechanism", {})
    assert editor["editable_recipe"]["root"]["law"]["coordinates"][0]["location"] == 3
    with pytest.raises(ValueError, match="timed out"):
        session.request("mechanism", {"max_seconds": 1e-9})
    assert session.task is not None
    assert session.task.recipe_json == baseline["recipe_json"]
    monkeypatch.setattr(p, "recipe_from_dict", original)
    draft = editor["editable_recipe"]
    draft["root"]["law"]["coordinates"][0]["location"] = 8
    validated = session.request("network_draft", {"recipe": draft})
    edited = session.request("edit", validated)
    reopened = ExplorerSession(tmp_path).request("open_edit", {"path": edited["edit"]["path"]})
    assert reopened["edit"]["card"] == edited["edit"]["card"]
    assert session.request("edit_export", {})["data"]
    assert session.request("view", {})["recipe_json"] == baseline["recipe_json"]


def test_profile_modes_and_raw_custody_survive_next_edit_reopen(tmp_path: Path):
    from copy import deepcopy

    session = ExplorerSession(tmp_path)
    controls = session.request("construction_examples", {})["examples"][
        "Source root: heterogeneous / gated / independent standardize (r0)"
    ]
    controls["source"]["task"].update(rows=16, context_rows=8)
    controls["calibration_rows"] = 32
    controls["modes"][0] = "center"
    controls["modes"][1] = "identity"
    baseline = session.request(
        "construct",
        {
            "controls": controls,
            "construction_seed": "7",
            "seed": "11",
            "task_id": "18014398509481987",
        },
    )
    record = json.loads(baseline["recipe_json"])
    root = record["recipe"]["root"]
    assert root["modes"] == controls["modes"]
    assert root["calibration_raw"]
    assert root["profile"]["raw"]
    details = baseline["view"]["construction_details"]["source_root"]
    assert details["independent_calibration"]["rows"] == 32
    next_view = session.request(
        "generate",
        {
            "record_json": baseline["recipe_json"],
            "frozen_identity": {"seed": "11", "task_id": "18014398509481988"},
        },
    )
    assert json.loads(next_view["recipe_json"])["recipe"]["root"] == root
    assert (
        session.request("generate", {"record_json": baseline["recipe_json"]})["recipe_json"]
        == baseline["recipe_json"]
    )
    draft = session.request("mechanism", {})["editable_recipe"]
    assert draft["root"] == root
    draft["network"]["blocks"][-1]["experts"][0]["bias"][0] += 1
    edited = session.request("edit", {"recipe": draft})
    assert edited["edit"]["after"]["recipe"]["recipe"]["root"] == root
    assert (
        ExplorerSession(tmp_path).request("open_edit", {"path": edited["edit"]["path"]})["edit"][
            "card"
        ]
        == edited["edit"]["card"]
    )
    bad = deepcopy(draft)
    bad["root"]["modes"][0] = "identity"
    with pytest.raises(ValueError, match="Invalid network draft"):
        session.request("network_draft", {"recipe": bad})
    assert session.request("view", {})["recipe_json"] == baseline["recipe_json"]


@pytest.mark.parametrize(
    "source_label",
    [
        "Standalone network: sparse + gated + skip (authored)",
        "Neural hyperpriors: I1 pinned code / gated / individual widths (r0 architecture)",
    ],
)
def test_root_wrapper_composes_with_authored_and_profiled_architectures(
    tmp_path: Path, source_label: str
):
    session = ExplorerSession(tmp_path)
    examples = session.request("construction_examples", {})["examples"]
    controls = examples["Source root: forced unit_uniform (r0) / dense"]
    controls["source"] = examples[source_label]
    controls["source"]["task"].update(rows=16, context_rows=8)
    result = session.request(
        "construct", {"controls": controls, "construction_seed": "0", "seed": "11", "task_id": "9"}
    )
    recipe = json.loads(result["recipe_json"])["recipe"]
    assert recipe["root"]["version"] == "sampled-source-root-base-v1"
    assert recipe["network"]["version"] == (
        "frozen-structural-network-v1"
        if "authored" in source_label
        else "sampled-hyperprior-network-v1"
    )
