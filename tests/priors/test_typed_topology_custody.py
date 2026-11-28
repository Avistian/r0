"""Fail closed before restoring streams; retain all caller streams on a failed draw."""

import copy
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from typed_topology_fixture import request

from r0.modules import generation as g
from r0.modules import priors as p
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot


def sample(controls: p.TypedTopologyRequest, engine: type[Any] = np.random.PCG64):
    streams = [np.random.Generator(engine(510 + i)) for i in range(5)]
    return p.sample_typed_topology_graph(
        controls,
        topology_rng=streams[0],
        named_rng=streams[1],
        raw_rng=streams[2],
        calibration_rng=streams[3],
        fit_rng=streams[4],
    )


@pytest.mark.parametrize(
    "engine",
    [np.random.PCG64, np.random.PCG64DXSM, np.random.SFC64, np.random.Philox, np.random.MT19937],
)
@pytest.mark.parametrize("family", ["functional", "native", "native-vector", "converted"])
def test_all_engine_task_archives_preserve_topology_and_mechanism_custody(
    engine: type[Any], family: str, tmp_path: Path
):
    controls = request(family)
    sampled = sample(controls, engine)
    task = p.generate_task(sampled, seed=601, task_id=2**55 + 809)
    path = tmp_path / "task.zip"
    g.save_group(path, [(2**55 + 809, task)], max_bytes=32_000_000)
    restored = next(g.load_group(path))
    assert restored.recipe_json == task.recipe_json
    assert torch.equal(restored.inputs, task.inputs)
    assert torch.equal(restored.targets, task.targets)
    trace = p.explain_generation(restored, [7, 3, 7], limits=controls.limits)
    assert trace["rows"] == [7, 3, 7]


@pytest.mark.parametrize(
    "field", ["version", "geometry", "child", "initial", "final", "topology", "working"]
)
def test_known_outer_metadata_rejects_before_restoring(field: str, monkeypatch: pytest.MonkeyPatch):
    controls = request("native-vector")
    raw = asdict(sample(controls))
    if field == "version":
        raw["version"] = "wrong"
    elif field == "geometry":
        raw["geometry"]["widths"] = (False,) * 4
    elif field == "child":
        raw["graph"]["request"]["calibration_rows"] = 33
    elif field == "initial":
        raw["initial"] = raw["initial"][:4]
    elif field == "final":
        raw["graph"]["final"] = raw["graph"]["initial"]
    elif field == "topology":
        raw["topology"]["request"]["nodes"] = 5
    calls: list[int] = []

    def forbidden(*args: Any, **kwargs: Any):
        calls.append(1)
        raise AssertionError("restored before known rejection")

    monkeypatch.setattr(GeneratorSnapshot, "restore", forbidden)
    with pytest.raises(
        ValueError, match="version|geometry|child|stream|snapshot|topology|limit|budget"
    ):
        p.typed_topology_graph_from_dict(
            raw,
            limits=replace(controls.limits, working_bytes=1)
            if field == "working"
            else controls.limits,
        )
    assert calls == []


def test_runtime_failure_rolls_back_all_five_streams(monkeypatch: pytest.MonkeyPatch):
    controls = request("native")
    streams = [np.random.default_rng(70 + i) for i in range(5)]
    before = copy.deepcopy([s.bit_generator.state for s in streams])

    def fail(*args: Any, **kwargs: Any):
        raise ValueError("deliberate materialization failure")

    monkeypatch.setattr(p.FrozenLinearFunction, "__post_init__", fail)
    with pytest.raises(ValueError, match="deliberate"):
        p.sample_typed_topology_graph(
            controls,
            topology_rng=streams[0],
            named_rng=streams[1],
            raw_rng=streams[2],
            calibration_rng=streams[3],
            fit_rng=streams[4],
        )
    assert [s.bit_generator.state for s in streams] == before


@pytest.mark.parametrize("family", ["functional", "native", "native-vector", "converted"])
def test_random_size_roles_preserve_realized_size(family: str):
    controls = replace(
        request(family),
        topology=p.TopologyHyperprior(log_nodes_low=math.log(3), log_nodes_high=math.log(7)),
    )
    sampled = sample(controls)
    assert len(sampled.runtime.nodes) == len(sampled.topology.parents)
    assert 3 <= len(sampled.runtime.nodes) <= 7


@pytest.mark.parametrize("field", ["native", "converted"])
@pytest.mark.parametrize("value", [False, {}, "unused"])
def test_irrelevant_role_controls_are_rejected_explicitly(field: str, value: Any):
    with pytest.raises(ValueError, match="role|vocabulary"):
        replace(request("functional"), **{field: value})


@pytest.mark.parametrize("family", ["functional", "native", "native-vector", "converted"])
@pytest.mark.parametrize("classes", [2, 3])
def test_classification_roles_survive_random_size_replay(family: str, classes: int):
    controls = request(family)
    changes: dict[str, Any] = dict(
        topology=p.TopologyHyperprior(log_nodes_low=math.log(3), log_nodes_high=math.log(7)),
        task=replace(
            controls.task, task_type="binary" if classes == 2 else "multiclass", classes=classes
        ),
    )
    if controls.native is not None:
        changes["native"] = replace(
            controls.native,
            features=p.NativeDomainPrior(kinds=("binary",)),
            target=p.NativeDomainPrior(kinds=("binary",))
            if classes == 2
            else p.NativeDomainPrior(kinds=("categorical",), cardinalities=(classes,)),
        )
    if controls.converted is not None:
        changes["converted"] = replace(
            controls.converted,
            vocabularies=(("low", "high"), ()),
            target_vocabulary=tuple(str(i) for i in range(classes)),
        )
    controls = replace(controls, **changes)
    sampled = sample(controls)
    reopened = p.typed_topology_graph_from_dict(asdict(sampled), limits=controls.limits)
    task = p.generate_task(reopened, seed=801, task_id=19)
    assert task.targets.dtype == torch.int64
    assert torch.all((task.targets >= 0) & (task.targets < classes))
    assert len(reopened.runtime.nodes) == len(sampled.topology.parents)
    assert asdict(reopened) == asdict(sampled)
