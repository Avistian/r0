"""Source-profile roots and neural row innovations compose without dropping custody."""

import json
from pathlib import Path

from r0.modules.explorer import ExplorerSession


def test_sampled_root_and_noise_compose_and_remain_frozen(tmp_path: Path):
    session = ExplorerSession(tmp_path)
    examples = session.request("construction_examples", {})["examples"]
    controls = examples["Noisy network: gated"]
    root = examples["Source root: heterogeneous / gated / independent standardize (r0)"]
    root["source"]["task"].update(rows=16, context_rows=8)
    root["calibration_rows"] = 32
    controls["source"] = root
    baseline = session.request(
        "construct",
        {
            "controls": controls,
            "construction_seed": "7",
            "seed": "11",
            "task_id": "18014398509481987",
        },
    )
    recipe = json.loads(baseline["recipe_json"])["recipe"]
    assert recipe["source"]["root"]["version"] == "sampled-source-root-base-v1"
    assert recipe["nodes"][0]["version"] == "functional-causal-node-v2"
    assert recipe["nodes"][0]["root"] == recipe["source"]["root"]
    assert recipe["sampling"] is not None
    after = session.request(
        "generate",
        {
            "record_json": baseline["recipe_json"],
            "frozen_identity": {"seed": "11", "task_id": "18014398509481988"},
        },
    )
    assert json.loads(after["recipe_json"])["recipe"] == recipe
    assert (
        session.request("generate", {"record_json": baseline["recipe_json"]})["recipe_json"]
        == baseline["recipe_json"]
    )
    editor = session.request("mechanism", {})
    assert editor["editable_recipe"]["root"] == recipe["source"]["root"]
    assert session.request("graph", {"indices": [15, 0, 15]})["graph"]["rows"] == [15, 0, 15]
    assert session.request("curve", {"feature": 0, "points": 7})["curve"]["reference"] == "zero"
