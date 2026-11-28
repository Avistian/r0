"""Frozen topology references retain source custody and literal downstream equations."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from r0.modules import generation as g
from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession


@pytest.mark.parametrize("family", ["authored linear", "joint linear"])
@pytest.mark.parametrize("engine", ["PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937"])
def test_guarded_topology_curve_literal_equations(tmp_path: Path, family: str, engine: str):
    controls = ExplorerSession(tmp_path / "catalog").request("construction_examples", {})[
        "examples"
    ][f"Topology hyperprior: {family}"]
    controls["engine"] = engine
    limits = p.CausalLimits(**controls["request"]["limits"])
    source = g.GraphConstruction(json.dumps(controls), 53).build(limits)
    assert isinstance(source, (p.HyperpriorGraph, p.JointHyperpriorGraph))
    seed, task_id = 2**55 + 33, 2**55 + 35
    result = g.run_guarded_document(
        dict(
            validation="network-curve",
            recipe=asdict(source),
            causal_limits=asdict(limits),
            seed=seed,
            task_id=task_id,
            anchor_row=5,
            anchor=[0.0],
            feature=0,
            points=7,
            bounds=[-1.3, 2.1],
            reference="original-row",
            max_seconds=60,
        ),
        tmp_path / "curve",
    )
    assert result.returncode == 0, result.stderr
    assert result.summary is not None
    curve = cast(dict[str, Any], result.summary)
    raw = json.loads(curve["source_json"])
    assert raw == json.loads(json.dumps(dict(recipe=asdict(source), seed=seed, task_id=task_id)))
    assert curve["original_task_id"] == str(task_id)
    assert curve["original_seed"] == str(seed)
    assert curve["source_sha256"] == hashlib.sha256(curve["source_json"].encode()).hexdigest()
    runtime = source.runtime
    states: list[NDArray[np.float64]] = []
    for index, node in enumerate(runtime.nodes):
        assert isinstance(node.function, p.FrozenLinearFunction)
        if node.root is not None:
            key = json.dumps(
                ("functional-causal-v1", seed, task_id, ("root-base", index)),
                sort_keys=True,
                separators=(",", ":"),
            )
            rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16))
            value = np.repeat(rng.normal(size=(16, 1))[5:6], 7, axis=0)
        else:
            value = np.concatenate([states[parent] for parent in node.parents], axis=1)
        value = value @ np.asarray(node.function.matrix).T
        for column, (n, channel) in enumerate(runtime.observed):
            if n == index:
                value[:, channel] = curve["values"] if column == 0 else 0.0
        states.append(value)
    expected = np.column_stack([states[n][:, channel] for n, channel in runtime.target])
    np.testing.assert_allclose(curve["scores"], expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize(("task_type", "classes"), [("binary", 2), ("multiclass", 3)])
def test_classification_reference_retains_raw_score_channels(
    tmp_path: Path, task_type: str, classes: int
):
    session = ExplorerSession(tmp_path)
    controls = session.request("construction_examples", {})["examples"][
        "Topology hyperprior: joint linear"
    ]
    controls["request"]["task"].update(task_type=task_type, classes=classes)
    controls["request"]["topology"]["nodes"] = 5
    session.request(
        "construct",
        dict(
            controls=controls,
            construction_seed=53,
            seed=19,
            task_id=23,
            causal_limits=controls["request"]["limits"],
            max_seconds=60,
        ),
    )
    assert session.task is not None
    original = session.task
    anchor = float(original.inputs[3, 0])
    response = session.request(
        "curve",
        dict(
            feature=0,
            anchor_row=3,
            points=2,
            bounds=[anchor, anchor],
            reference="original-row",
            causal_limits=controls["request"]["limits"],
            max_seconds=60,
        ),
    )
    response = response["curve"]
    np.testing.assert_allclose(
        response["scores"], np.repeat(original.scores[3:4].numpy(), 2, axis=0), rtol=0, atol=1e-12
    )
    assert "not a conditional mean" in response["score_semantics"]
    assert session.task.recipe_json == original.recipe_json
    with pytest.raises(ValueError, match="points|curve"):
        session.request("curve", dict(feature=0, points=1))
    assert session.task.recipe_json == original.recipe_json
