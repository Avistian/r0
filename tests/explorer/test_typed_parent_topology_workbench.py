"""Guarded parent composition retains explicit edges and original-row inspection."""

import json
from pathlib import Path

import pytest

from r0.modules.explorer import ExplorerSession


@pytest.mark.parametrize("family", ["functional", "native", "native-vector", "converted"])
@pytest.mark.parametrize("source", ["tabiclv2", "mitra2", "extension"])
def test_parent_topology_workbench_retains_composition(tmp_path: Path, family: str, source: str):
    session = ExplorerSession(tmp_path)
    controls = session.request("construction_examples", {})["examples"][
        f"Parent topology: {source} / {family}"
    ]
    caps = controls["request"]["limits"]
    result = session.request(
        "construct",
        dict(
            controls=controls,
            construction_seed=77,
            seed=79,
            task_id=83,
            causal_limits=caps,
            max_seconds=120,
        ),
    )
    recipe = json.loads(result["recipe_json"])["recipe"]
    assert recipe["version"] == "sampled-typed-parent-topology-r0-v1"
    assert recipe["request"] == json.loads(json.dumps(controls["request"]))
    assert recipe["initial"][0] == recipe["final"][0]
    assert recipe["graph"]["request"]["composition"]["source"] == source
    assert session.request("graph", dict(indices=[7, 2, 7], causal_limits=caps))["graph"][
        "rows"
    ] == [7, 2, 7]
    assert session.request("mechanism", dict(causal_limits=caps))["editable_recipe"]["nodes"]
