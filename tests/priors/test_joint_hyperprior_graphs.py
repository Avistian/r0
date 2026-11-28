"""Joint mechanisms retain independent topology and all construction stream custody."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from r0.modules import generation as g
from r0.modules import priors as p


def test_joint_hyperprior_public_generation_and_exact_archive():
    prior = p.JointFunctionPrior("lin", matrix=p.JointMatrixPrior(families=("gaussian",)))
    request = p.JointHyperpriorGraphRequest(
        p.LinearRecipe(rows=16, features=1, signal_features=1, active_features=1, context_rows=8),
        p.FixedSizeTopologyHyperprior(3),
        calibration_rows=16,
        root_prior=p.JointRootPrior(("gaussian",), prior),
        function_prior=prior,
        limits=p.CausalLimits(parameters=8_000_000, working_bytes=2 * 1024**3),
    )
    streams = [np.random.default_rng(i) for i in range(5)]
    recipe = p.sample_joint_hyperprior_graph(
        request,
        topology_rng=streams[0],
        named_rng=streams[1],
        raw_rng=streams[2],
        calibration_rng=streams[3],
        fit_rng=streams[4],
    )
    task = p.generate_task(recipe, seed=2**53 + 1, task_id=2**53 + 7)
    decoded = p.recipe_from_dict(json.loads(json.dumps(asdict(recipe))), limits=request.limits)
    replayed = p.generate_task(decoded, seed=2**53 + 1, task_id=2**53 + 7)
    assert torch.equal(task.targets, replayed.targets)
    assert task.recipe_json == replayed.recipe_json
    assert p.explain_generation(task, [15, 0, 15], limits=request.limits)["target_max_error"] == 0


def _request(family: str = "lin", *, matrix: str = "gaussian"):
    prior = p.JointFunctionPrior(
        family, matrix=p.JointMatrixPrior(families=(matrix,)), frequencies=8
    )
    root_prior = p.JointFunctionPrior("r0-linear-projection-v1", matrix=prior.matrix)
    return p.JointHyperpriorGraphRequest(
        p.LinearRecipe(rows=16, features=1, signal_features=1, active_features=1, context_rows=8),
        p.FixedSizeTopologyHyperprior(3),
        calibration_rows=32,
        root_prior=p.JointRootPrior(("gaussian",), root_prior),
        function_prior=prior,
        limits=p.CausalLimits(parameters=64_000_000, working_bytes=16 * 1024**3),
        raw_limits=p.RawDrawLimits(numbers=16_000_000, working_bytes=4 * 1024**3),
    )


def _sample(
    request: p.JointHyperpriorGraphRequest, engine: type[np.random.BitGenerator] = np.random.PCG64
):
    streams = [np.random.Generator(engine(i + 31)) for i in range(5)]
    return p.sample_joint_hyperprior_graph(
        request,
        topology_rng=streams[0],
        named_rng=streams[1],
        raw_rng=streams[2],
        calibration_rng=streams[3],
        fit_rng=streams[4],
    )


@pytest.mark.parametrize(
    "family",
    [
        "lin",
        "quad",
        "gp",
        "tree",
        "disc",
        "em",
        "mlp",
        "prod",
        "kumo-prod",
        "default",
        "default,kumo-prod",
    ],
)
def test_every_source_catalog_has_retained_mechanisms_and_support_bounds(family: str):
    request = _request(family)
    recipe = _sample(request)
    budget = request.admit()
    assert sum(len(draw.values) for draw in recipe.graph.raw) <= budget["raw_numbers"]
    assert len(recipe.graph.raw) <= budget["raw_draws"]
    assert recipe.graph.registry.decisions.draw_count <= budget["named_calls"]
    assert recipe.graph.admission.planned_parameters <= budget["parameters"]
    assert recipe.graph.admission.planned_working_bytes <= budget["estimated_working_bytes"]
    explicit = p.generate_task(recipe.graph, seed=41, task_id=42)
    task = p.generate_task(recipe, seed=41, task_id=42)
    assert torch.equal(task.inputs, explicit.inputs)
    assert torch.equal(task.targets, explicit.targets)
    assert p.joint_hyperprior_graph_from_dict(asdict(recipe), limits=request.limits) == recipe
    assert p.explain_generation(task, [15, 0, 15], limits=request.limits)["signal_max_error"] == 0


@pytest.mark.parametrize(
    "matrix", ["gaussian", "importance", "factorized", "kernel", "transformed"]
)
def test_all_source_matrix_families_without_filtering(matrix: str):
    request = _request(matrix=matrix)
    recipe = _sample(request)
    mechanism = recipe.graph.plan.nodes[1].mechanism
    assert isinstance(mechanism, p.JointFunctionPlan)
    assert mechanism.matrices[0].family == matrix
    recipe.validate()


@pytest.mark.parametrize(
    "engine",
    [np.random.PCG64, np.random.PCG64DXSM, np.random.Philox, np.random.SFC64, np.random.MT19937],
)
def test_five_stream_engines_match_separately_called_public_samplers(
    engine: type[np.random.BitGenerator], tmp_path: Path
):
    request = _request()
    recipe = _sample(request, engine)
    topology = p.sample_topology_hyperprior(
        request.topology,
        rng=np.random.Generator(engine(31)),
        limits=request.limits,
        raw_limits=request.raw_limits,
    )
    child = p.sample_joint_graph(
        request.for_parents(topology.parents),
        named_rng=np.random.Generator(engine(32)),
        raw_rng=np.random.Generator(engine(33)),
        calibration_rng=np.random.Generator(engine(34)),
        fit_rng=np.random.Generator(engine(35)),
    )
    assert recipe.topology == topology
    assert recipe.graph == child
    task = p.generate_task(recipe, seed=2**53 + 17, task_id=2**53 + 19)
    archive = tmp_path / "task.zip"
    g.save_group(archive, [(2**53 + 19, task)], max_bytes=16 * 1024**2)
    loaded = next(g.load_group(archive))
    assert torch.equal(loaded.targets, task.targets)
    assert loaded.recipe_json == task.recipe_json


def test_known_support_limits_and_invalid_outer_fields_before_any_restore():
    request = _request()
    recipe = _sample(request)
    raw = json.loads(json.dumps(asdict(recipe)))
    for key, value in (("layout", "unknown"), ("mode", "invalid"), ("width", 0)):
        changed = json.loads(json.dumps(raw))
        changed["request"][key] = value
        with (
            patch.object(
                type(recipe.topology.initial), "restore", side_effect=AssertionError("restore")
            ),
            pytest.raises(ValueError, match="request|width|version|control|geometry"),
        ):
            p.joint_hyperprior_graph_from_dict(changed, limits=request.limits)
    with patch("copy.deepcopy", side_effect=AssertionError("clone")):
        with pytest.raises(ValueError, match="limit|budget"):
            replace(request, raw_limits=p.RawDrawLimits(numbers=2))
        with pytest.raises(ValueError, match="limit|budget"):
            replace(request, limits=p.CausalLimits(working_bytes=1024))


@pytest.mark.parametrize("mode", ["meta", "local", "global"])
@pytest.mark.parametrize("policy", ["shared", "refresh"])
def test_named_source_modes_and_scope_policy_remain_explicit(mode: str, policy: str):
    request = replace(_request(), mode=mode, categorical_policy=policy)
    recipe = _sample(request)
    assert recipe.graph.registry.decisions.numerical.default_mode == mode
    assert recipe.graph.registry.decisions.categorical.policy == policy
    recipe.validate()


def test_five_stream_rollback_and_declared_state_caps_before_cloning():
    request = _request()
    streams = [np.random.default_rng(i) for i in range(5)]
    before = [json.dumps(s.bit_generator.state, sort_keys=True) for s in streams]
    kwargs = dict(
        topology_rng=streams[0],
        named_rng=streams[1],
        raw_rng=streams[2],
        calibration_rng=streams[3],
        fit_rng=streams[4],
    )
    with (
        patch.object(
            p.JointHyperpriorGraph, "validate_structure", side_effect=ValueError("late bind")
        ),
        pytest.raises(ValueError, match="late bind"),
    ):
        p.sample_joint_hyperprior_graph(request, **kwargs)
    assert [json.dumps(s.bit_generator.state, sort_keys=True) for s in streams] == before
    small = replace(request, raw_limits=replace(request.raw_limits, state_bytes=1))
    with (
        patch("copy.deepcopy", side_effect=AssertionError("clone")),
        pytest.raises(ValueError, match="state byte|snapshot"),
    ):
        p.sample_joint_hyperprior_graph(small, **kwargs)
    assert [json.dumps(s.bit_generator.state, sort_keys=True) for s in streams] == before


def test_full_support_boundaries_do_not_draw_or_filter_the_catalog():
    request = _request("default,kumo-prod")
    budget = request.admit()
    with patch("copy.deepcopy", side_effect=AssertionError("clone")):
        for caps in (
            replace(request.raw_limits, numbers=budget["raw_numbers"] - 1),
            replace(request.raw_limits, draws=budget["raw_draws"] - 1),
        ):
            with pytest.raises(ValueError, match="full-support raw"):
                replace(request, raw_limits=caps)
        with pytest.raises(ValueError, match="budget"):
            replace(request, limits=replace(request.limits, parameters=budget["parameters"] - 1))
    exact = replace(
        request,
        raw_limits=replace(
            request.raw_limits, numbers=budget["raw_numbers"], draws=budget["raw_draws"]
        ),
    )
    assert exact.function_prior.functions == "default,kumo-prod"
    exact.admit()


@pytest.mark.parametrize("family", ["uniform", "gaussian", "ball", "covariance"])
def test_root_laws_and_structural_noise_share_the_existing_construction(family: str):
    original = _request()
    request = replace(
        original,
        root_prior=replace(original.root_prior, families=(family,)),
        noise=p.JointGraphNoise(scales=(0.2,)),
    )
    recipe = _sample(request)
    root = recipe.graph.runtime.nodes[0].root
    assert isinstance(root, p.JointRootBase)
    assert root.family == family
    task = p.generate_task(recipe, seed=7, task_id=8)
    assert p.explain_generation(task, [15, 1, 15], limits=request.limits)["target_max_error"] == 0


def test_unchanged_full_default_catalog_and_exact_nested_tamper_rejection():
    original = _request()
    prior = p.JointFunctionPrior("default,kumo-prod", frequencies=8)
    request = replace(original, root_prior=p.JointRootPrior(function=prior), function_prior=prior)
    recipe = _sample(request)
    raw = json.loads(json.dumps(asdict(recipe)))
    for section in ("request", "topology", "graph"):
        changed = json.loads(json.dumps(raw))
        changed[section]["unexpected"] = 1
        with pytest.raises(ValueError, match="field|record"):
            p.joint_hyperprior_graph_from_dict(changed, limits=request.limits)
    changed = json.loads(json.dumps(raw))
    changed["graph"]["raw"][0]["values"][0] += 0.1
    with pytest.raises(ValueError, match="replay"):
        p.joint_hyperprior_graph_from_dict(changed, limits=request.limits)
    request.admit()


@pytest.mark.parametrize(("kind", "classes"), [("binary", 2), ("multiclass", 3)])
def test_joint_classification_adapter_and_guarded_run(kind: str, classes: int, tmp_path: Path):
    from typing import Literal, cast

    base = _request()
    task = replace(
        base.task, task_type=cast(Literal["binary", "multiclass"], kind), classes=classes
    )
    request = replace(base, task=task, topology=p.FixedSizeTopologyHyperprior(4))
    recipe = _sample(request)
    generated = p.generate_task(recipe, seed=41, task_id=42)
    assert generated.probabilities is not None
    assert generated.probabilities.shape == (16, classes)
    trace = p.explain_generation(generated, [15, 0, 15], limits=request.limits)
    assert trace["target_max_error"] == 0
    run = g.run_generation(
        g.RunRequest(
            recipe,
            seed=41,
            start=42,
            count=1,
            max_seconds=45,
            max_bytes=16 * 1024**2,
            causal_limits=request.limits,
        ),
        tmp_path,
    )
    assert run.completed == 1
    assert run.failed == 0


def test_real_prefix_overflow_rolls_back_all_five_streams_without_retry():
    request = replace(_request("tree"), noise=p.JointGraphNoise(scales=(1e308,)))
    streams = [np.random.default_rng(i + 31) for i in range(5)]
    before = [json.dumps(s.bit_generator.state, sort_keys=True) for s in streams]
    with pytest.raises(ValueError, match="nonfinite|finite"):
        p.sample_joint_hyperprior_graph(
            request,
            topology_rng=streams[0],
            named_rng=streams[1],
            raw_rng=streams[2],
            calibration_rng=streams[3],
            fit_rng=streams[4],
        )
    assert [json.dumps(s.bit_generator.state, sort_keys=True) for s in streams] == before


def test_task_row_changes_preserve_topology_and_independent_fit_receipts():
    recipe = _sample(_request("tree"))
    changed = recipe.with_task(replace(recipe.task, rows=23, context_rows=9))
    assert changed.topology is recipe.topology
    assert changed.graph.raw == recipe.graph.raw
    assert changed.graph.initial == recipe.graph.initial
    assert changed.graph.final == recipe.graph.final
    assert changed.graph.realizations == recipe.graph.realizations
    assert changed.graph.calibrations == recipe.graph.calibrations
    assert changed.graph.registry == recipe.graph.registry
    assert changed.graph.request.calibration_rows == recipe.graph.request.calibration_rows
