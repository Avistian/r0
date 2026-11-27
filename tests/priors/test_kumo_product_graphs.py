"""Broad products travel through every existing sampled construction boundary."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from r0.modules import priors
from r0.modules.generation import load_group, save_group


def _functions(token: str) -> priors.JointFunctionPrior:
    return priors.JointFunctionPrior(
        functions=token, frequencies=4, matrix=priors.JointMatrixPrior(families=("gaussian",))
    )


def _streams() -> dict[str, np.random.Generator]:
    return dict(
        zip(
            ("named_rng", "raw_rng", "calibration_rng", "fit_rng"),
            (np.random.default_rng(s) for s in (72, 10, 11, 12)),
            strict=True,
        )
    )


def _task() -> priors.LinearRecipe:
    return priors.LinearRecipe(
        rows=16, context_rows=8, features=1, signal_features=1, active_features=1
    )


def _build(
    path: str,
) -> (
    priors.SampledFunctionalGraph
    | priors.SampledParentGraph
    | priors.SampledGeometryGraph
    | priors.SampledNativeGraph
):
    root = priors.JointRootPrior(families=("gaussian",), function=_functions("lin"))
    f = _functions("kumo-prod")
    composition = priors.ParentCompositionPrior(
        source="extension", mode="aggregate", aggregation="sum"
    )
    if path in ("explicit", "parent", "root"):
        request = priors.JointGraphRequest(
            _task(),
            ((), (0,), (0, 1)),
            (2, 2, 2),
            ((0, 0),),
            ((2, 1),),
            calibration_rows=8,
            root_prior=root,
            function_prior=f,
            mode="local",
        )
        if path == "root":
            request = replace(
                request,
                parents=((),),
                widths=(2,),
                target=((0, 1),),
                root_prior=replace(root, function=f),
            )
        return (
            priors.sample_parent_graph(request, composition=composition, **_streams())
            if path == "parent"
            else priors.sample_joint_graph(request, **_streams())
        )
    if path in ("native", "native-parent", "native-root"):
        domain = priors.NativeDomainPrior(
            kinds=("categorical",), cardinalities=(3,), emissions=("softmax",)
        )
        kind = (
            priors.NativeParentGraphRequest
            if path == "native-parent"
            else priors.NativeGraphRequest
        )
        request = kind(
            parents=((), (0,), (0, 1)),
            domains=(domain,) * 3,
            observed=((0, 0),),
            target=(2, 0),
            rows=16,
            context_rows=8,
            calibration_rows=8,
            root_prior=root,
            function_prior=f,
            mode="local",
        )
        if isinstance(request, priors.NativeParentGraphRequest):
            request = replace(request, composition=composition)
        if path == "native-root":
            request = replace(
                request,
                parents=((), ()),
                domains=(domain,) * 2,
                target=(1, 0),
                root_prior=replace(root, function=f),
            )
        return priors.sample_native_graph(request, **_streams())
    kind = (
        priors.GeometryParentRequest
        if path == "geometry-parent"
        else priors.GeometryPipelineRequest
        if path == "pipeline"
        else priors.GeometryGraphRequest
    )
    request = kind(
        topology=priors.AttachmentGraphPrior(nodes=3, edges_per_node=2),
        task=_task(),
        width_prior=priors.JointWidthPrior(low=1, high=2),
        subsample=False,
        calibration_rows=8,
        root_prior=root,
        function_prior=f,
        mode="local",
    )
    pipeline = priors.JointPipelinePrior(
        converter=priors.JointNumericalConverterPrior(
            allow_activation=False, allow_kumaraswamy=False
        )
    )
    if isinstance(request, priors.GeometryParentRequest):
        request = replace(request, composition=composition, pipeline_prior=pipeline)
    elif isinstance(request, priors.GeometryPipelineRequest):
        request = replace(request, pipeline_prior=pipeline)
    return priors.sample_geometry_graph(request, **_streams())


@pytest.mark.parametrize(
    "path",
    [
        "explicit",
        "parent",
        "root",
        "geometry",
        "pipeline",
        "geometry-parent",
        "native",
        "native-parent",
        "native-root",
    ],
)
def test_broad_products_generate_decode_inspect_resplit_and_save(path: str, tmp_path: Path) -> None:
    graph = _build(path)
    assert priors.recipe_from_dict(json.loads(json.dumps(asdict(graph)))) == graph
    task = priors.generate_task(graph, seed=21, task_id=4)
    inspection = priors.explain_generation(task, [3, 0, 3])
    assert inspection["rows"] == [3, 0, 3]
    assert any(n["label"] == "Kumo product left child" for n in inspection["nodes"])
    assert any(n["label"] == "Kumo product right child" for n in inspection["nodes"])
    assert hasattr(graph, "with_task")
    resplit = graph.with_task(replace(graph.task, context_rows=4, split=1))
    other = priors.generate_task(resplit, seed=21, task_id=4)
    assert task.source_task_id == other.source_task_id
    torch.testing.assert_close(task.inputs, other.inputs, rtol=0, atol=0)
    torch.testing.assert_close(task.targets, other.targets, rtol=0, atol=0)
    save_group(tmp_path / "product.zip", [(4, task)], max_bytes=10_000_000)
    assert next(load_group(tmp_path / "product.zip")).episode() == task.episode()
    if not path.startswith("native"):
        pair = priors.generate_pair(graph, intervention="noise", value=0.2, seed=21, task_id=4)
        assert all(json.loads(pair.card_json)["required_equalities"].values())
