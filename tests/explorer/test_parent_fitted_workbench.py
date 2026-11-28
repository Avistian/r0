"""Public construction, frozen editing and archive workflows for parent/tree mechanisms."""

import base64
import csv
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from r0.modules.explorer import ExplorerSession


def _controls(session: ExplorerSession, label: str) -> dict[str, Any]:
    controls = session.request("construction_examples", {})["examples"][label]
    if controls["topology"] == "fitted":
        controls["skeleton"]["task"].update(rows=16, context_rows=8)
        for fit in controls["fits"]:
            fit["plan"]["rows"] = 32
    elif controls["topology"] == "sampled-native":
        controls.update(rows=16, context_rows=8, calibration_rows=16)
    else:
        controls["task"].update(rows=16, context_rows=8)
        controls["calibration_rows"] = 16
        if "width_prior" in controls:
            controls["width_prior"].update(low=1, high=2)
    return controls


def _construct(session: ExplorerSession, controls: dict[str, Any]) -> dict[str, Any]:
    return session.request(
        "construct", {"controls": controls, "construction_seed": 28, "seed": 19, "task_id": 7}
    )


def _export(session: ExplorerSession) -> None:
    with zipfile.ZipFile(
        io.BytesIO(base64.b64decode(session.request("export", {})["data"]))
    ) as archive:
        rows = list(csv.DictReader(io.StringIO(archive.read("predictor.csv").decode())))
        assert len(rows) == 16
        assert all(row["context_target"] == "" for row in rows if row["role"] == "query")


@pytest.mark.parametrize(
    "label",
    [
        "Parent composition: tabiclv2 explicit",
        "Parent composition: mitra2 explicit",
        "Parent composition: extension explicit",
        "Parent composition: geometry",
        "Parent composition: pipeline historical-tuple",
        "Parent composition: pipeline historical-graph",
        "Parent composition: pipeline readout-only",
        "Parent composition: native domains",
    ],
)
def test_parent_construction_inspection_export_and_reopen(tmp_path: Path, label: str) -> None:
    session = ExplorerSession(tmp_path)
    controls = _controls(session, label)
    source = controls["composition"]["source"]
    controls["composition"].update(
        mode="sample" if source == "mitra2" else "aggregate",
        aggregation="mean" if source != "tabiclv2" else "sum",
    )
    result = _construct(session, controls)
    details = result["view"]["construction_details"]
    assert details["kind"] == "parent-composition"
    assert details["composition"]["source"] == source
    assert session.request("graph", {"indices": [15, 0, 15]})["graph"]["rows"] == [15, 0, 15]
    editor = session.request("mechanism", {})
    assert any(node["function"] == "frozen-parent-function-v1" for node in editor["nodes"])
    if controls["topology"] == "sampled-native":
        assert editor["native"] is True
        assert result["view"]["target_origin"] == "native-variable-v1"
    draft = json.loads(json.dumps(editor["editable_recipe"]))
    equations = [n["equation"] if editor.get("native") else n for n in draft["nodes"]]
    selected = next(n for n in equations if n["function"]["version"] == "frozen-parent-function-v1")
    selected["function"]["aggregation"] = "max"
    edited = session.request("edit", {"recipe": draft})
    assert edited["edit"]["card"]["after_episode_id"] != result["view"]["episode_id"]
    assert session.request("edit_export", {})["data"]
    assert (
        ExplorerSession(tmp_path).request("open_edit", {"path": edited["edit"]["path"]})["edit"][
            "card"
        ]
        == edited["edit"]["card"]
    )
    _export(session)
    reopened = ExplorerSession(tmp_path).request(
        "open", {"archive": result["archive"], "task_id": 7}
    )
    assert reopened["recipe_json"] == result["recipe_json"]


@pytest.mark.parametrize("family", ["dt", "extra_trees", "rf", "gbdt"])
@pytest.mark.parametrize("classification", [False, True])
def test_fitted_construction_freezes_edits_and_replays(
    tmp_path: Path, family: str, classification: bool
) -> None:
    session = ExplorerSession(tmp_path)
    controls = _controls(session, f"Fitted structural tree: {family}")
    if classification:
        for fit in controls["fits"]:
            fit["plan"].update(task="classification", classes=3)
    result = _construct(session, controls)
    raw = json.loads(result["recipe_json"])["recipe"]
    assert raw["version"] == "fitted-functional-causal-v1"
    assert len(raw["fits"]) == 2
    assert all(f["realization"]["plan"]["family"] == family for f in raw["fits"])
    assert result["view"]["construction_details"]["kind"] == "independent-fitted-trees"
    assert "adapter" in result["view"]["construction_details"]["notice"]
    assert session.request("graph", {"indices": [15, 0, 15]})["graph"]["signal_max_error"] == 0
    editor = session.request("mechanism", {})
    assert editor["editable_recipe"]["version"] == "functional-causal-v1"
    assert "fits" not in editor["editable_recipe"]
    assert "does not refit" in editor["notice"]
    # Change a root coefficient, retaining the exact downstream fitted tables.
    edited = json.loads(json.dumps(editor["editable_recipe"]))
    edited["nodes"][0]["function"]["matrix"][0][0] = 2.0
    comparison = session.request("edit", {"recipe": edited})
    assert comparison["view"]["episode_id"] == result["view"]["episode_id"]
    assert comparison["edit"]["card"]["after_episode_id"] != result["view"]["episode_id"]
    assert session.request("edit_export", {})["data"]
    reopened_edit = ExplorerSession(tmp_path).request(
        "open_edit", {"path": comparison["edit"]["path"]}
    )
    assert reopened_edit["edit"]["card"] == comparison["edit"]["card"]
    _export(session)
    replay = session.request("generate", {"record_json": result["recipe_json"]})
    assert replay["view"]["episode_id"] == result["view"]["episode_id"]
    next_task = session.request(
        "generate",
        {"record_json": result["recipe_json"], "frozen_identity": {"seed": "20", "task_id": "8"}},
    )
    next_record = json.loads(next_task["recipe_json"])
    assert next_record["recipe"]["fits"] == raw["fits"]
    assert (next_record["seed"], next_record["task_id"]) == (20, 8)
    assert next_task["view"]["episode_id"] != result["view"]["episode_id"]
    reopened = ExplorerSession(tmp_path).request(
        "open", {"archive": result["archive"], "task_id": 7}
    )
    assert reopened["recipe_json"] == result["recipe_json"]


def test_fitted_build_is_guarded_and_invalid_fit_retains_active_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from r0.modules import priors
    from r0.modules.generation import GraphConstruction

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("construction must run inside the guarded child")

    monkeypatch.setattr(priors, "sample_fitted_graph", forbidden)
    monkeypatch.setattr(GraphConstruction, "task_record", forbidden)
    session = ExplorerSession(tmp_path)
    controls = _controls(session, "Fitted structural tree: dt")
    before = _construct(session, controls)
    receipts = set(tmp_path.glob(".attempts/**/*-result.json"))
    controls["fits"][0]["plan"]["input_width"] = 256
    with pytest.raises(ValueError, match="Generation failed"):
        _construct(session, controls)
    assert session.request("view", {})["view"]["episode_id"] == before["view"]["episode_id"]
    added = set(tmp_path.glob(".attempts/**/*-result.json")) - receipts
    assert len(added) == 1
    assert json.loads(added.pop().read_text())["status"] == "FAILED"


@pytest.mark.parametrize(
    "label",
    [
        "Explicit causal graph",
        "Numerical pipeline: readout-only",
        "Native mixed graph (sampled domains)",
    ],
)
def test_original_frozen_archive_text_supports_new_identities_without_rewriting_construction(
    tmp_path: Path, label: str
) -> None:
    session = ExplorerSession(tmp_path)
    result = _construct(session, _controls(session, label))
    original = json.loads(result["recipe_json"])
    replay = session.request(
        "generate",
        {
            "record_json": result["recipe_json"],
            "frozen_identity": {"seed": "27", "task_id": "8"},
        },
    )
    changed = json.loads(replay["recipe_json"])
    assert changed["recipe"] == original["recipe"]
    assert (changed["seed"], changed["task_id"]) == (27, 8)
    assert replay["view"]["episode_id"] != result["view"]["episode_id"]
    if label.startswith("Native"):
        assert replay["view"]["feature_schema"] == result["view"]["feature_schema"]
        assert replay["view"]["target_origin"] == "native-variable-v1"
