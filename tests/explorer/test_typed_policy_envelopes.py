"""Typed worker construction retains sampled/fitted sources and explicit hard lifecycles."""

import json
from pathlib import Path

import numpy as np
import pytest

from r0.modules.explorer import ExplorerSession
from r0.modules.generation import load_causal_pair


@pytest.mark.parametrize(
    "label",
    [
        "Native mixed graph (sampled domains)",
        "Native heterogeneous vector graph",
        "Fitted structural tree: dt",
        "Sampled fitted parameters: dt regression (r0)",
        "XGBoost shared-vector fit: reg:squarederror",
        "Typed categorical converter pipelines",
    ],
)
@pytest.mark.parametrize("lifecycle", ["independent-calibration-r0", "source-reference-population"])
def test_retained_envelopes_through_guarded_policy_worker(
    tmp_path: Path, label: str, lifecycle: str
) -> None:
    session = ExplorerSession(tmp_path)
    controls = session.request("construction_examples", {})["examples"][label]
    if "skeleton" in controls:
        controls["skeleton"]["task"].update(rows=16, context_rows=8)
    elif "request" in controls:
        controls["request"].update(rows=16, context_rows=8, calibration_rows=16)
        controls["request"]["converter_prior"]["modes"] = ["softmax_disc"]
    else:
        controls.update(rows=16, context_rows=8, calibration_rows=16)
    baseline = session.request(
        "construct", {"controls": controls, "construction_seed": 28, "seed": 19, "task_id": 7}
    )
    original = json.loads(baseline["recipe_json"])["recipe"]
    info = session.request("intervention_controls", {})
    address = {"node": 0, "channels": [0], "location": "state_and_readout"}
    if info["native"]:
        address = {"node": 0, "kind": "native-variable", "variable": 0, "location": "native-value"}
    elif info["converted"]:
        capture = info["nodes"][0]["converters"][0]
        address = {
            "node": 0,
            "kind": "capture",
            "capture": capture["name"],
            "location": "capture",
            "probabilities": [1.0] + [0.0] * (len(capture["vocabulary"]) - 1),
        }
    draft = {
        "profile": "explicit-typed-policy-r0-v1",
        "streams": [3, 5, 7],
        "kind": "hard",
        "eligibility": [0],
        "count": 1,
        "mode": "matched",
        "bounds_lifecycle": lifecycle,
        "reference_rows": 16,
        "locations": [address],
    }
    result = session.request("sample_intervention_policy", {"policy_json": json.dumps(draft)})
    pair = load_causal_pair(tmp_path / result["intervention"]["path"])
    record = json.loads(pair.after.recipe_json)["recipe"]
    assert record["request"]["base"] == original
    assert pair.before.recipe_json == baseline["recipe_json"]
    assert result["recipe_json"] == baseline["recipe_json"]
    # The selected root's first observed channel is a single retained environment constant.
    if info["converted"]:
        # Capture-only changes do not replace descendant representations.
        np.testing.assert_array_equal(pair.after.targets, pair.before.targets)
    else:
        np.testing.assert_array_equal(pair.after.inputs[:, 0], pair.after.inputs[0, 0])
    if not info["converted"]:
        assert record["bounds"]
        assert record["request"]["reference_rows"] == 16
        if lifecycle == "source-reference-population":
            assert record["request"]["reference_seed"] == 19
            assert record["request"]["reference_task_id"] == 7
    graph = session.request(
        "intervention_graph",
        {
            "indices": [15, 0, 15],
            "max_seconds": 60,
            "expected_intervention_id": result["intervention"]["after_id"],
        },
    )
    assert graph["after"]["rows"] == [15, 0, 15]
    reopened = ExplorerSession(tmp_path).request(
        "open_intervention", {"path": result["intervention"]["path"]}
    )
    assert reopened["intervention"] == result["intervention"]
