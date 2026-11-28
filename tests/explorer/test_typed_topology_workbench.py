"""Four-family typed topology is constructed through the guarded session."""

import json
from pathlib import Path

import pytest

from r0.modules.explorer import ExplorerSession


@pytest.mark.parametrize("family", ["functional", "native", "native-vector", "converted"])
def test_typed_topology_guarded_construct_and_trace(tmp_path: Path, family: str):
    session = ExplorerSession(tmp_path)
    controls = session.request("construction_examples", {})["examples"][
        f"Topology hyperprior: typed {family}"
    ]
    caps = controls["request"]["limits"]
    result = session.request(
        "construct",
        dict(
            controls=controls,
            construction_seed=28,
            seed=19,
            task_id=7,
            causal_limits=caps,
            max_seconds=120,
        ),
    )
    archive = json.loads(result["recipe_json"])["recipe"]
    assert archive["version"] == "sampled-typed-topology-r0-v1"
    assert archive["request"] == json.loads(json.dumps(controls["request"]))
    assert len(archive["initial"]) == 5
    assert session.request("graph", dict(indices=[7, 0, 7], causal_limits=caps))["graph"][
        "rows"
    ] == [7, 0, 7]
    editor = session.request("mechanism", dict(causal_limits=caps))
    assert editor["editable_recipe"]["nodes"]


@pytest.mark.parametrize("field", ["engine", "request", "caps"])
def test_invalid_typed_controls_reject_before_streams(tmp_path: Path, field: str):
    from unittest.mock import patch

    from r0.modules import priors as p
    from r0.modules.generation import GraphConstruction

    controls = ExplorerSession(tmp_path).request("construction_examples", {})["examples"][
        "Topology hyperprior: typed native"
    ]
    caps = p.CausalLimits(**controls["request"]["limits"])
    if field == "engine":
        controls["engine"] = "unknown"
    elif field == "request":
        controls["request"]["native"] = None
    else:
        caps = p.CausalLimits(working_bytes=1)
    with (
        patch("numpy.random.SeedSequence", side_effect=AssertionError("early stream")),
        pytest.raises(ValueError, match="engine|role|limit|budget"),
    ):
        GraphConstruction(json.dumps(controls), 28).build(caps)
