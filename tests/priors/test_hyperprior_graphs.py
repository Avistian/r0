"""Topology construction remains attached to generated tasks and public archives."""

import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import numpy as np
import pytest
import torch

from r0.modules import generation as g
from r0.modules import priors as p


def test_hyperprior_graph_public_archive_and_original_row_trace():
    root = p.FunctionalCausalNode(
        (),
        p.FrozenTreeFunction(1, ((0,),), ((0.0,),), (((2.0,), (2.0,)),)),
        p.JointRootBase("gaussian", 1),
    )
    edge = p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((3.0,),)))
    request = p.HyperpriorGraphRequest(
        p.LinearRecipe(rows=16, features=1, signal_features=1, active_features=1, context_rows=8),
        p.FixedSizeTopologyHyperprior(5),
        root,
        edge,
    )
    recipe = p.sample_hyperprior_graph(request, rng=np.random.default_rng(73))
    restored = p.recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
    task = p.generate_task(restored, seed=2**53 + 17, task_id=2**53 + 29)
    values: list[float] = []
    for parents in recipe.topology.parents:
        values.append(2.0 if not parents else 3 * values[parents[0]])
    assert torch.equal(task.inputs, torch.full((16, 1), values[0], dtype=torch.float64))
    assert torch.equal(task.scores, torch.full((16, 1), values[-1], dtype=torch.float64))
    trace = p.explain_generation(task, [15, 0, 15])
    assert trace["signal_max_error"] == trace["target_max_error"] == 0
    assert json.loads(task.recipe_json)["recipe"] == json.loads(json.dumps(asdict(recipe)))


def _request(task_type: Literal["regression", "binary", "multiclass"] = "regression"):
    task = p.LinearRecipe(
        rows=16,
        features=2,
        signal_features=2,
        active_features=2,
        context_rows=8,
        task_type=task_type,
        classes={"regression": 0, "binary": 2, "multiclass": 3}[task_type],
    )
    root = p.FunctionalCausalNode(
        (), p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))), p.JointRootBase("gaussian", 2)
    )
    edge = p.FunctionalCausalNode(
        (0,), p.FrozenLinearFunction(((0.5, 0.25), (0.0, 2.0))), noise_scale=(0.1, 0.2)
    )
    return p.HyperpriorGraphRequest(
        task, p.TopologyHyperprior(math.log(4), math.log(7), components=2), root, edge
    )


@pytest.mark.parametrize(
    "engine",
    [np.random.PCG64, np.random.PCG64DXSM, np.random.Philox, np.random.SFC64, np.random.MT19937],
)
@pytest.mark.parametrize("task_type", ["regression", "binary", "multiclass"])
def test_five_engine_task_archive_noise_and_target_adapter(
    engine: type[np.random.BitGenerator],
    task_type: Literal["regression", "binary", "multiclass"],
    tmp_path: Path,
):
    recipe = p.sample_hyperprior_graph(_request(task_type), rng=np.random.Generator(engine(43)))
    task = p.generate_task(recipe, seed=2**53 + 1, task_id=2**53 + 3)
    explicit = p.generate_task(recipe.runtime, seed=2**53 + 1, task_id=2**53 + 3)
    assert torch.equal(task.inputs, explicit.inputs)
    assert torch.equal(task.targets, explicit.targets)
    assert torch.equal(task.scores, explicit.scores)
    path = tmp_path / "graph.zip"
    g.save_group(path, [(2**53 + 3, task)], max_bytes=8 * 1024**2)
    loaded = next(g.load_group(path))
    restored = p.recipe_from_dict(json.loads(loaded.recipe_json)["recipe"])
    assert restored == recipe
    replayed = p.generate_task(restored, seed=2**53 + 1, task_id=2**53 + 3)
    assert replayed.recipe_json == loaded.recipe_json
    assert torch.equal(loaded.targets, replayed.targets)
    trace = p.explain_generation(loaded, [15, 0, 15])
    assert trace["signal_max_error"] == trace["target_max_error"] == 0
    seen: set[str] = set()
    for node in trace["nodes"]:
        assert set(node["parents"]) <= seen
        seen.add(node["id"])
    next_recipe = recipe.with_task(replace(recipe.task, rows=23, context_rows=9))
    next_task = p.generate_task(next_recipe, seed=99, task_id=108)
    assert next_recipe.topology is recipe.topology
    assert next_task.source_task_id != task.source_task_id


def test_decode_and_construction_limits_precede_any_rng_restore_or_clone():
    request = _request()
    recipe = p.sample_hyperprior_graph(request, rng=np.random.default_rng(15))
    raw = json.loads(json.dumps(asdict(recipe)))
    raw["request"]["task"]["rows"] = 20000
    raw["request"]["limits"]["working_bytes"] = 80 * 1024**2
    raw["topology"]["limits"]["working_bytes"] = 80 * 1024**2
    with patch.object(
        type(recipe.topology.initial), "restore", side_effect=AssertionError("restore")
    ):
        with pytest.raises(ValueError, match="budget|memory|workspace|limit"):
            p.hyperprior_graph_from_dict(raw)
        with pytest.raises(ValueError, match="budget|memory|workspace|limit"):
            p.hyperprior_graph_from_dict(asdict(recipe), limits=p.CausalLimits(working_bytes=1024))
    # A very large permitted N is refused before opening the sampler's transaction.
    with (
        patch("copy.deepcopy", side_effect=AssertionError("allocation")),
        pytest.raises(ValueError, match="limit"),
    ):
        p.HyperpriorGraphRequest(
            request.task,
            p.FixedSizeTopologyHyperprior(100000),
            request.root_template,
            request.edge_template,
        )


def test_strict_archive_fields_and_derived_receipt_tampering():
    recipe = p.sample_hyperprior_graph(_request(), rng=np.random.default_rng(1))
    raw = json.loads(json.dumps(asdict(recipe)))
    for key in raw:
        with pytest.raises(ValueError, match="field|record|archive|construction|parent|topology"):
            p.hyperprior_graph_from_dict({k: v for k, v in raw.items() if k != key})
    for key in raw["request"]:
        changed = json.loads(json.dumps(raw))
        del changed["request"][key]
        with pytest.raises(ValueError, match="field|record|archive|construction|parent|topology"):
            p.hyperprior_graph_from_dict(changed)
    changed = json.loads(json.dumps(raw))
    changed["topology"]["parents"][-1] = []
    with pytest.raises(ValueError, match="field|record|archive|construction|parent|topology"):
        p.hyperprior_graph_from_dict(changed)
    changed = json.loads(json.dumps(raw))
    changed["request"]["root_template"]["function"]["unused"] = 1
    with pytest.raises(ValueError, match="field|record|archive|construction|parent|topology"):
        p.hyperprior_graph_from_dict(changed)


def test_sampled_convolution_template_and_parent_composition(tmp_path: Path):
    layer = p.Conv1DLayerPrior((p.Conv1DGeometry(1, 1),), bias_law="rademacher")
    sample = p.sample_feature_conv1d(
        p.FeatureConv1DRequest(2, 1, (layer,)),
        named_rng=np.random.default_rng(3),
        raw_rng=np.random.default_rng(4),
    )
    request = _request()
    edge = p.FunctionalCausalNode((0,), p.FrozenParentFunction((sample,), aggregation="mean"))
    recipe = p.sample_hyperprior_graph(
        replace(request, edge_template=edge), rng=np.random.default_rng(7)
    )
    task = p.generate_task(recipe, seed=13, task_id=18)
    decoded = p.hyperprior_graph_from_dict(json.loads(task.recipe_json)["recipe"])
    assert decoded == recipe
    assert p.explain_generation(task, [15, 0, 15])["target_max_error"] == 0
    run = g.run_generation(
        g.RunRequest(recipe, seed=13, start=18, count=1, max_seconds=45, max_bytes=16 * 1024**2),
        tmp_path,
    )
    assert run.completed == 1
    assert run.failed == 0
    replay = g.run_generation(
        g.RunRequest(recipe, seed=13, start=18, count=1, max_seconds=45, max_bytes=16 * 1024**2),
        tmp_path,
    )
    assert replay.completed == 1
    assert replay.attempted == 0


def test_inspection_rejects_forged_topology_provenance():
    recipe = p.sample_hyperprior_graph(_request(), rng=np.random.default_rng(14))
    task = p.generate_task(recipe, seed=19, task_id=20)
    card = json.loads(task.mechanism_json)
    card["topology_receipt_fingerprint"] = "unrelated-construction"
    with pytest.raises(ValueError, match="topology.*metadata"):
        p.explain_generation(replace(task, mechanism_json=json.dumps(card)), [0])


def test_sampler_rolls_back_late_runtime_assembly_failure():
    request = _request()
    rng = np.random.default_rng(3)
    before = json.dumps(rng.bit_generator.state, sort_keys=True)
    with (
        patch.object(p.HyperpriorGraphRequest, "runtime", side_effect=ValueError("late assembly")),
        pytest.raises(ValueError, match="late assembly"),
    ):
        p.sample_hyperprior_graph(request, rng=rng)
    assert json.dumps(rng.bit_generator.state, sort_keys=True) == before
