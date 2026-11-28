"""Joint heterogeneous nodes preserve variable, score and one-hot state semantics."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from r0.modules import priors as p
from r0.modules.generation import load_group, save_group


def authored() -> p.NativeVectorRecipe:
    continuous = p.VariableDomain()
    binary = p.VariableDomain("binary", ("no", "yes"), "threshold", (0.0,))
    categorical = p.VariableDomain("categorical", ("a", "b", "c"), "softmax")
    root = p.NativeVectorNode(
        p.FunctionalCausalNode(
            (),
            p.FrozenLinearFunction(((1.0,), (1.0,), (-1.0,), (0.0,), (1.0,))),
            p.JointRootBase("gaussian", 1),
        ),
        (continuous, binary, categorical),
    )
    child = p.NativeVectorNode(
        p.FunctionalCausalNode(
            (0,),
            p.FrozenLinearFunction(
                ((1.0, 0.0, 10.0, 0.0, 1.0, 2.0), (1.0, 0.0, 10.0, 0.0, 1.0, 2.0))
            ),
        ),
        (continuous, binary),
    )
    return p.NativeVectorRecipe(
        p.LinearRecipe(rows=32, context_rows=16, features=4, signal_features=4, active_features=4),
        (root, child),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
        (1, 0),
    )


def test_joint_equation_uses_ordered_one_hot_blocks_and_typed_sibling_readouts() -> None:
    recipe = authored()
    task = p.generate_task(recipe, seed=17, task_id=5)
    x = task.inputs.numpy()
    expected = x[:, 0] + 10 * x[:, 1] + x[:, 2]
    np.testing.assert_array_equal(task.targets.numpy(), expected)
    np.testing.assert_array_equal(x[:, 1], (x[:, 0] >= 0).astype(float))
    np.testing.assert_array_equal(x[:, 3], (expected >= 0).astype(float))
    assert [f.kind for f in task.feature_schema] == [
        "continuous",
        "binary",
        "categorical",
        "binary",
    ]
    assert [
        (s.score_start, s.score_stop, s.state_start, s.state_stop) for s in recipe.nodes[0].spans
    ] == [(0, 1, 0, 1), (1, 2, 1, 3), (2, 5, 3, 6)]
    with pytest.raises(ValueError, match="cannot be observed"):
        replace(recipe, target=(1, 1))
    with pytest.raises(ValueError, match="encoded parent widths"):
        replace(
            recipe,
            nodes=(
                recipe.nodes[0],
                replace(
                    recipe.nodes[1],
                    equation=p.FunctionalCausalNode(
                        (0,), p.FrozenLinearFunction(((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)))
                    ),
                ),
            ),
        )


def test_softmax_probability_and_sample_oracle_for_target_sibling() -> None:
    base = authored()
    recipe = replace(
        base,
        task=replace(
            base.task,
            features=2,
            signal_features=2,
            active_features=2,
            task_type="multiclass",
            classes=3,
        ),
        nodes=(base.nodes[0],),
        observed=((0, 0), (0, 1)),
        target=(0, 2),
    )
    task = p.generate_task(recipe, seed=17, task_id=5)
    x = task.inputs[:, 0].numpy()
    logits = np.stack((-x, np.zeros_like(x), x), axis=1)
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    probability = weights / weights.sum(axis=1, keepdims=True)
    assert task.probabilities is not None
    np.testing.assert_allclose(task.probabilities.numpy(), probability, rtol=1e-14)
    graph = p.explain_generation(task, list(range(16)))
    uniform = np.array(next(n for n in graph["nodes"] if n["id"] == "uniform-0-2")["values"])[:, 0]
    cdf = probability[:16].cumsum(axis=1)
    cdf[:, -1] = 1
    np.testing.assert_array_equal(task.targets[:16].numpy(), (uniform[:, None] >= cdf).sum(axis=1))
    assert task.targets.dtype == torch.int64


def test_correlated_joint_innovations_are_applied_once_before_sibling_emissions() -> None:
    base = authored()
    child = base.nodes[1]
    equation = replace(
        child.equation,
        noise_scale=(0.5, 0.5),
        exogenous=p.ExogenousDistribution(factor=((1.0, 1.0), (0.0, 0.0))),
    )
    recipe = replace(base, nodes=(base.nodes[0], replace(child, equation=equation)))
    task = p.generate_task(recipe, seed=17, task_id=5)
    values = np.array(
        next(
            n
            for n in p.explain_generation(task, list(range(16)))["nodes"]
            if n["id"] == "function-1"
        )["values"]
    )
    np.testing.assert_array_equal(values[:, 0], values[:, 1])
    np.testing.assert_array_equal(
        task.inputs[:, -1].numpy(), (task.targets.numpy() >= 0).astype(float)
    )


def sampled_request(parent: bool = False, function: str = "lin") -> p.NativeVectorGraphRequest:
    c = p.NativeDomainPrior(kinds=("continuous",))
    b = p.NativeDomainPrior(
        kinds=("binary",),
        emissions=("threshold",),
        frequency_laws=("uniform",),
        threshold_laws=("uniform-quantile",),
    )
    k = p.AsymmetricNativeDomainPrior(
        kinds=("categorical",),
        cardinalities=(3,),
        emissions=("softmax",),
        concentration_vectors=((1.0, 2.0, 4.0),),
    )
    k5 = p.AsymmetricNativeDomainPrior(
        kinds=("categorical",),
        cardinalities=(5,),
        emissions=("softmax",),
        concentration_vectors=((1.0, 2.0, 3.0, 4.0, 5.0),),
    )
    f = p.JointFunctionPrior(functions=function, matrix=p.JointMatrixPrior(families=("gaussian",)))
    return p.NativeVectorGraphRequest(
        parents=((), (0,)),
        domains=((c, b, k), (k5, c, b)),
        observed=((0, 0), (0, 1), (0, 2), (1, 1)),
        target=(1, 0),
        rows=16,
        context_rows=8,
        calibration_rows=8,
        root_prior=p.JointRootPrior(families=("gaussian",), function=f),
        function_prior=f,
        composition=p.ParentCompositionPrior(
            source="extension", mode="aggregate", aggregation="sum"
        )
        if parent
        else None,
    )


def sample(request: p.NativeVectorGraphRequest):
    return p.sample_native_vector_graph(
        request,
        **{
            s: np.random.default_rng(i)
            for i, s in enumerate(("named_rng", "raw_rng", "calibration_rng", "fit_rng"))
        },
    )


@pytest.mark.parametrize("parent", [False, True])
@pytest.mark.parametrize("function", ["lin", "tree"])
def test_sampled_vector_archive_prefix_custody_and_typed_save(
    parent: bool, function: str, tmp_path: Path
) -> None:
    graph = sample(sampled_request(parent, function))
    assert [
        (n.equation.function.output_width, n.state_width, n.value_width) for n in graph.recipe.nodes
    ] == [(5, 6, 3), (7, 8, 3)]
    assert [v.operation[-1] for v in graph.domains[0].variables] == [
        "domain:0",
        "domain:1",
        "domain:2",
    ]
    assert p.recipe_from_dict(json.loads(json.dumps(asdict(graph)))) == graph
    task = p.generate_task(graph, seed=8, task_id=9)
    assert len(task.classes) == 5
    assert task.feature_schema[-1].kind == "continuous"
    assert p.explain_generation(task, [3, 1, 3])["rows"] == [3, 1, 3]
    other = p.generate_task(
        graph.with_task(replace(graph.task, split=2, context_rows=4)), seed=8, task_id=9
    )
    assert task.source_task_id == other.source_task_id
    torch.testing.assert_close(task.targets, other.targets, rtol=0, atol=0)
    save_group(tmp_path / "vector.zip", [(9, task)], max_bytes=10_000_000)
    assert next(load_group(tmp_path / "vector.zip")).episode() == task.episode()
    if function == "tree" and not parent:
        calibration = np.asarray(asdict(graph.realizations[1])["calibration"]["sample"])
        assert calibration.shape == (8, 6)
        np.testing.assert_array_equal(calibration[:, 1:3].sum(axis=1), 1)
        np.testing.assert_array_equal(calibration[:, 3:6].sum(axis=1), 1)
    data = json.loads(json.dumps(asdict(graph)))
    data["domains"][0]["variables"][2]["concentrations"][0] += 1
    with pytest.raises(ValueError, match="replay"):
        p.recipe_from_dict(data)


def test_vector_geometry_admission_and_four_stream_atomicity() -> None:
    r = sampled_request()
    with pytest.raises(ValueError, match="scalar variable"):
        replace(r, domains=((replace(r.domains[0][0], continuous_width=2),), r.domains[1]))
    with pytest.raises(ValueError, match="channel"):
        replace(r, limits=replace(r.limits, node_channels=4))
    streams = {
        s: np.random.default_rng(i)
        for i, s in enumerate(("named_rng", "raw_rng", "calibration_rng", "fit_rng"))
    }
    before = json.dumps([x.bit_generator.state for x in streams.values()])
    with pytest.raises(ValueError, match="limit"):
        p.sample_native_vector_graph(replace(r, raw_limits=p.RawDrawLimits(draws=1)), **streams)
    assert json.dumps([x.bit_generator.state for x in streams.values()]) == before


def test_inspection_nodes_are_in_dependency_order_for_graph_rendering() -> None:
    task = p.generate_task(authored(), seed=17, task_id=5)
    graph = p.explain_generation(task, [3, 1, 3])
    seen: set[str] = set()
    for node in graph["nodes"]:
        assert set(node["parents"]) <= seen, node["id"]
        seen.add(node["id"])


def test_unequal_categorical_blocks_in_one_node_keep_separate_score_and_state_spans() -> None:
    request = sampled_request()
    # Append five categories after the same node's three-category variable.
    domains = (request.domains[0] + (request.domains[1][0],),)
    request = replace(
        request, parents=((),), domains=domains, observed=((0, 0), (0, 1), (0, 2)), target=(0, 3)
    )
    graph = sample(request)
    node = graph.recipe.nodes[0]
    assert (node.equation.function.output_width, node.state_width, node.value_width) == (10, 11, 4)
    assert (node.spans[2].state_start, node.spans[2].state_stop) == (3, 6)
    assert (node.spans[3].state_start, node.spans[3].state_stop) == (6, 11)
    task = p.generate_task(graph, seed=8, task_id=9)
    assert len(task.classes) == 5
    assert len(task.feature_schema[2].vocabulary) == 3
    assert task.inputs.shape == (16, 3)
    assert p.recipe_from_dict(json.loads(json.dumps(asdict(graph)))) == graph
