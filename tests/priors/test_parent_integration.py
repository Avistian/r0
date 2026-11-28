"""Parent equations compose with sampled geometry, numerical pipelines and native states."""

import json
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from r0.modules import priors
from r0.modules.generation import load_group, save_group


def _streams(seed: int = 100) -> dict[str, np.random.Generator]:
    return dict(
        zip(
            ("named_rng", "raw_rng", "calibration_rng", "fit_rng"),
            (np.random.default_rng(seed + i) for i in range(4)),
            strict=True,
        )
    )


def _geometry(pipeline: bool, source: str, family: str = "lin") -> priors.GeometryParentRequest:
    f = priors.JointFunctionPrior(
        functions=family, frequencies=4, matrix=priors.JointMatrixPrior(families=("gaussian",))
    )
    return priors.GeometryParentRequest(
        topology=priors.AttachmentGraphPrior(nodes=4, edges_per_node=2),
        task=priors.LinearRecipe(
            rows=16, context_rows=8, features=2, signal_features=2, active_features=2
        ),
        subsample=False,
        width_prior=priors.JointWidthPrior(low=1, high=3),
        calibration_rows=8,
        root_prior=priors.JointRootPrior(families=("gaussian",), function=f),
        function_prior=f,
        composition=priors.ParentCompositionPrior(
            source=source,
            mode="sample" if source == "mitra2" else "aggregate",
            aggregation="weighted" if source == "mitra2" else "sum",
        ),
        pipeline_prior=priors.JointPipelinePrior() if pipeline else None,
    )


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("source", ["tabiclv2", "mitra2"])
def test_geometry_parent_archive_pipeline_scope_and_group_roundtrip(
    pipeline: bool, source: str, tmp_path: Path
) -> None:
    graph = priors.sample_geometry_graph(_geometry(pipeline, source), **_streams())
    assert isinstance(graph, priors.SampledParentGeometry)
    assert bool(graph.pipelines) == pipeline
    assert graph.replay() == graph
    assert priors.recipe_from_dict(json.loads(json.dumps(asdict(graph)))) == graph
    assert any(isinstance(n.function, priors.FrozenParentFunction) for n in graph.runtime.nodes)
    task = priors.generate_task(graph, seed=17, task_id=2)
    assert priors.explain_generation(task, [3, 0, 3])["rows"] == [3, 0, 3]
    resplit = graph.with_task(replace(graph.task, context_rows=4, split=1))
    other = priors.generate_task(resplit, seed=17, task_id=2)
    assert task.source_task_id == other.source_task_id
    torch.testing.assert_close(task.inputs, other.inputs, rtol=0, atol=0)
    pair = priors.generate_pair(graph, intervention="noise", value=0.2, seed=17, task_id=2)
    assert all(json.loads(pair.card_json)["required_equalities"].values())
    save_group(tmp_path / "parents.zip", [(2, task)], max_bytes=10_000_000)
    assert next(load_group(tmp_path / "parents.zip")).episode() == task.episode()
    if pipeline:
        for fitted in graph.pipelines:
            assert (
                len(fitted.plan.converters)
                == graph.layout.converter_channels[int(fitted.plan.operation[0].split(":")[1])]
            )


def test_fitted_geometry_children_use_finalized_parent_pipelines_and_rollback() -> None:
    request = _geometry(True, "tabiclv2", "tree")
    graph = priors.sample_geometry_graph(request, **_streams())
    assert graph.replay() == graph
    assert graph.graph.calibrations
    total = len(graph.topology.raw) + len(graph.observations.raw) + len(graph.graph.raw)
    streams = _streams()
    before = [json.dumps(r.bit_generator.state) for r in streams.values()]
    with pytest.raises(ValueError, match="limit"):
        priors.sample_geometry_graph(
            replace(request, raw_limits=priors.RawDrawLimits(draws=total - 1)), **streams
        )
    assert [json.dumps(r.bit_generator.state) for r in streams.values()] == before
    raw = json.loads(json.dumps(asdict(graph)))
    raw["graph"]["calibrations"][-1]["role"] = "wrong"
    with pytest.raises(ValueError, match="differs from complete replay"):
        priors.recipe_from_dict(raw)


def _native(kind: str, family: str, aggregation: str) -> priors.NativeParentGraphRequest:
    domain = priors.NativeDomainPrior(kinds=(kind,), cardinalities=(3, 4))
    f = priors.JointFunctionPrior(
        functions=family, frequencies=4, matrix=priors.JointMatrixPrior(families=("gaussian",))
    )
    return priors.NativeParentGraphRequest(
        parents=((), (), (0, 1), (1, 2)),
        domains=(domain,) * 4,
        observed=((0, 0), (1, 0)),
        target=(3, 0),
        rows=16,
        context_rows=8,
        calibration_rows=16,
        root_prior=priors.JointRootPrior(families=("gaussian",), function=f),
        function_prior=f,
        composition=priors.ParentCompositionPrior(
            source="extension", mode="aggregate", aggregation=aggregation
        ),
    )


@pytest.mark.parametrize("kind", ["continuous", "binary", "categorical"])
@pytest.mark.parametrize(
    "aggregation", ["sum", "product", "mean", "max", "logsumexp", "weighted", "mlp"]
)
def test_native_parent_encoded_widths_replay_and_inspection(kind: str, aggregation: str) -> None:
    graph = priors.sample_native_graph(_native(kind, "lin", aggregation), **_streams(1))
    assert isinstance(graph, priors.SampledNativeParentGraph)
    graph.validate_replay()
    assert priors.recipe_from_dict(json.loads(json.dumps(asdict(graph)))) == graph
    function = graph.recipe.nodes[2].equation.function
    assert isinstance(function, priors.FrozenParentFunction)
    assert tuple(c.input_width for c in function.children) == tuple(
        graph.recipe.nodes[p].state_width for p in (0, 1)
    )
    task = priors.generate_task(graph, seed=8, task_id=9)
    assert priors.explain_generation(task, [2, 0])["rows"] == [2, 0]
    other = priors.generate_task(
        graph.with_task(replace(graph.task, context_rows=4, split=1)), seed=8, task_id=9
    )
    assert task.source_task_id == other.source_task_id
    torch.testing.assert_close(task.targets, other.targets, rtol=0, atol=0)


def test_native_parent_fitted_slices_retain_raw_ancestor_receipts(tmp_path: Path) -> None:
    graph = priors.sample_native_graph(_native("categorical", "tree", "sum"), **_streams(114))
    assert isinstance(graph, priors.SampledNativeParentGraph)
    assert graph.calibrations
    graph.validate_replay()
    for receipt in graph.calibrations:
        assert all(
            r.stream == "calibration" for r in graph.raw[receipt.raw_start : receipt.raw_stop]
        )
    task = priors.generate_task(graph, seed=8, task_id=9)
    save_group(tmp_path / "native.zip", [(9, task)], max_bytes=10_000_000)
    assert next(load_group(tmp_path / "native.zip")).episode() == task.episode()
    raw = deepcopy(asdict(graph))
    raw["calibrations"] = ()
    with pytest.raises(ValueError, match="differs from complete replay"):
        priors.recipe_from_dict(raw)


@pytest.mark.parametrize("family", ["quad", "gp", "disc", "em", "mlp", "prod"])
def test_native_parent_accepts_remaining_sampled_function_families(family: str) -> None:
    graph = priors.sample_native_graph(_native("categorical", family, "sum"), **_streams(114))
    graph.validate_replay()
    task = priors.generate_task(graph, seed=8, task_id=9)
    assert torch.isfinite(task.inputs).all()
    assert priors.recipe_from_dict(asdict(graph)) == graph


@pytest.mark.parametrize("aggregation", ["weighted", "mlp"])
def test_native_mitra_joint_member_uses_all_encoded_parent_states(aggregation: str) -> None:
    request = replace(
        _native("categorical", "lin", aggregation),
        composition=priors.ParentCompositionPrior(source="mitra2", aggregation=aggregation),
    )
    graph = priors.sample_native_graph(request, **_streams(1))
    function = graph.recipe.nodes[2].equation.function
    assert isinstance(function, priors.FrozenParentFunction)
    assert function.joint is not None
    assert function.joint.input_width == sum(graph.recipe.nodes[p].state_width for p in (0, 1))
    graph.validate_replay()


def test_native_parent_rejected_large_tree_draw_rolls_back_every_stream() -> None:
    # This explicit source draw exceeds the default complete-construction budget.
    # Retain its rejection rather than silently resampling another tree geometry.
    streams = _streams(1)
    before = [json.dumps(r.bit_generator.state) for r in streams.values()]
    with pytest.raises(ValueError, match="aggregate native construction"):
        priors.sample_native_graph(_native("categorical", "tree", "sum"), **streams)
    assert [json.dumps(r.bit_generator.state) for r in streams.values()] == before


def test_geometry_neural_reducer_is_fitted_before_its_sampled_node_pipeline() -> None:
    request = replace(
        _geometry(True, "tabiclv2"),
        composition=priors.ParentCompositionPrior(
            source="extension", mode="aggregate", aggregation="mlp"
        ),
    )
    request = replace(
        request,
        pipeline_prior=priors.JointPipelinePrior(
            converter=priors.JointNumericalConverterPrior(
                allow_activation=False, allow_kumaraswamy=False
            )
        ),
    )
    graph = priors.sample_geometry_graph(request, **_streams())
    assert isinstance(graph, priors.SampledParentGeometry)
    assert graph.graph.calibrations
    for receipt in graph.graph.calibrations:
        assert isinstance(receipt, priors.ParentCalibrationReceipt)
        assert receipt.role == "reducer-fit"
        assert receipt.raw_stop <= graph.pipelines[receipt.node].raw_start
    graph.validate()
    assert priors.recipe_from_dict(asdict(graph)) == graph
    task = priors.generate_task(graph, seed=17, task_id=2)
    assert priors.explain_generation(task, [0, 2])["rows"] == [0, 2]


def test_neural_parent_pipeline_rejects_uncertified_warp_without_retry() -> None:
    request = replace(
        _geometry(True, "tabiclv2"),
        composition=priors.ParentCompositionPrior(
            source="extension", mode="aggregate", aggregation="mlp"
        ),
    )
    streams = _streams()
    before = [json.dumps(r.bit_generator.state) for r in streams.values()]
    with pytest.raises(ValueError, match="finite output second moments"):
        priors.sample_geometry_graph(request, **streams)
    assert [json.dumps(r.bit_generator.state) for r in streams.values()] == before
