"""Independent arithmetic and complete archive checks for parent-specific mechanisms."""

import json
from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from r0.modules import priors


def request(family: str = "lin") -> priors.JointGraphRequest:
    functions = priors.JointFunctionPrior(
        functions=family, frequencies=4, matrix=priors.JointMatrixPrior(families=("gaussian",))
    )
    return priors.JointGraphRequest(
        priors.LinearRecipe(
            rows=16, context_rows=8, features=1, signal_features=1, active_features=1
        ),
        ((), (), (0, 1)),
        (1, 2, 1),
        ((0, 0),),
        ((2, 0),),
        calibration_rows=8,
        root_prior=priors.JointRootPrior(families=("gaussian",), function=functions),
        function_prior=functions,
    )


def sample(prior: priors.ParentCompositionPrior, family: str = "lin") -> priors.SampledParentGraph:
    return priors.sample_parent_graph(
        request(family),
        composition=prior,
        named_rng=np.random.default_rng(114),
        raw_rng=np.random.default_rng(115),
        calibration_rng=np.random.default_rng(116),
        fit_rng=np.random.default_rng(117),
    )


@pytest.mark.parametrize(
    "aggregation", ["sum", "product", "mean", "max", "logsumexp", "weighted", "mlp"]
)
def test_distinct_slices_and_joint_member_equation(aggregation: str) -> None:
    first = priors.FrozenLinearFunction(((2.0,),))
    second = priors.FrozenLinearFunction(((3.0, -1.0),))
    joint = priors.FrozenLinearFunction(((1.0, 1.0, 1.0),))
    reducer = priors.FrozenMLPFunction((priors.FrozenLinearFunction(((1.0, 2.0, 3.0),)),))
    function = priors.FrozenParentFunction(
        (first, second),
        aggregation,
        joint,
        (0.2, 0.3, 0.5) if aggregation == "weighted" else (),
        reducer if aggregation == "mlp" else None,
    )
    x = np.array([[1.0, 2.0, 4.0], [-2.0, 1.0, 3.0]])
    members = np.column_stack((x.sum(axis=1), 2 * x[:, 0], 3 * x[:, 1] - x[:, 2]))
    expected = {
        "sum": members.sum(1),
        "product": members.prod(1),
        "mean": members.mean(1),
        "max": members.max(1),
        "logsumexp": np.logaddexp.reduce(members, axis=1),
        "weighted": members @ np.array([0.2, 0.3, 0.5]),
        "mlp": members @ np.array([1.0, 2.0, 3.0]),
    }[aggregation]
    actual = priors.evaluate_frozen_function(function, torch.tensor(x))
    np.testing.assert_allclose(actual[:, 0].numpy(), expected)
    assert priors.frozen_function_from_dict(json.loads(json.dumps(asdict(function)))) == function


@pytest.mark.parametrize(
    ("source", "aggregation"),
    [("tabiclv2", a) for a in ("sum", "product", "max", "logsumexp")]
    + [("mitra2", a) for a in ("mean", "weighted", "mlp", "product", "max")]
    + [("extension", a) for a in ("sum", "product", "mean", "max", "logsumexp", "weighted", "mlp")],
)
def test_source_modes_replay_generate_inspect_and_resplit(source: str, aggregation: str) -> None:
    prior = priors.ParentCompositionPrior(
        source=source,
        mode="sample" if source == "mitra2" else "aggregate",
        aggregation=aggregation,
    )
    graph = sample(prior)
    function = graph.runtime.nodes[-1].function
    assert isinstance(function, priors.FrozenParentFunction)
    assert (function.joint is not None) == (source == "mitra2")
    raw = json.loads(json.dumps(asdict(graph), allow_nan=False))
    assert priors.recipe_from_dict(raw) == graph
    graph.validate()
    task = priors.generate_task(graph, seed=19, task_id=7)
    other = priors.generate_task(
        graph.with_task(replace(graph.task, context_rows=4, split=1)), seed=19, task_id=7
    )
    assert task.source_task_id == other.source_task_id
    assert task.episode_id != other.episode_id
    np.testing.assert_array_equal(task.inputs.numpy(), other.inputs.numpy())
    explanation = priors.explain_generation(task, [7, 0, 7])
    assert len([n for n in explanation["nodes"] if n["id"].startswith("parent-member-")]) == 2
    corrupted = deepcopy(raw)
    corrupted["raw"][0]["role"] = "wrong"
    with pytest.raises(ValueError, match="differs from complete replay"):
        priors.parent_graph_from_dict(corrupted)


@pytest.mark.parametrize("family", ["lin", "quad", "gp", "tree", "disc", "em", "mlp", "prod"])
def test_all_function_families_fit_distinct_parent_populations(family: str) -> None:
    graph = sample(priors.ParentCompositionPrior(mode="aggregate", aggregation="sum"), family)
    assert graph.replay() == graph
    function = graph.runtime.nodes[-1].function
    assert isinstance(function, priors.FrozenParentFunction)
    assert tuple(c.input_width for c in function.children) == (1, 2)
    for receipt in graph.calibrations:
        assert receipt.raw_stop > receipt.raw_start
        records = graph.raw[receipt.raw_start : receipt.raw_stop]
        assert all(
            r.stream == "calibration"
            and r.owner[:4] == ("graph", "construction", "node:2", receipt.role)
            for r in records
        )


def test_catalogs_limits_and_failed_transaction() -> None:
    with pytest.raises(ValueError, match="source-specific"):
        priors.ParentCompositionPrior(aggregation="mean")
    streams = [np.random.default_rng(i) for i in range(4)]
    before = [json.dumps(s.bit_generator.state) for s in streams]
    with pytest.raises(ValueError, match="limit"):
        priors.sample_parent_graph(
            replace(request(), limits=priors.CausalLimits(parameters=100)),
            named_rng=streams[0],
            raw_rng=streams[1],
            calibration_rng=streams[2],
            fit_rng=streams[3],
        )
    assert [json.dumps(s.bit_generator.state) for s in streams] == before
    function = priors.FrozenParentFunction(
        (priors.FrozenLinearFunction(((1e200,),)),) * 2, "product"
    )
    with pytest.raises(ValueError, match="not retried"):
        priors.evaluate_frozen_function(function, torch.ones((1, 2)))


def test_parent_product_propagates_dependent_moment_bound_and_noise_pairs() -> None:
    linear = priors.FrozenLinearFunction(((1.0,),))
    root = priors.FunctionalCausalNode(
        (),
        linear,
        priors.JointRootBase("uniform", 1),
        (1.0,),
        priors.ExogenousDistribution(priors.InputDistribution("student_t", df=4.0)),
    )
    alias = priors.FunctionalCausalNode((0,), linear)
    product = priors.FunctionalCausalNode(
        (0, 1), priors.FrozenParentFunction((linear, linear), "product")
    )
    graph = priors.FunctionalCausalRecipe(
        request().task, (root, alias, product), ((0, 0),), ((2, 0),)
    )
    task = priors.generate_task(graph, seed=3, task_id=4)
    assert json.loads(task.mechanism_json)["moment_contract"]["target_orders"] == [2.0]
    torch.testing.assert_close(task.scores, task.inputs**2)
    sampled = sample(priors.ParentCompositionPrior(source="mitra2", aggregation="weighted"))
    pair = priors.generate_pair(sampled, intervention="noise", value=0.25, seed=19, task_id=7)
    assert all(json.loads(pair.card_json)["required_equalities"].values())
    reopened = priors.recipe_from_dict(json.loads(pair.before.recipe_json)["recipe"])
    assert reopened == sampled


def test_parent_slices_must_match_each_graph_parent_and_decoder_limits() -> None:
    one = priors.FrozenLinearFunction(((1.0,),))
    two = priors.FrozenLinearFunction(((1.0, 1.0),))
    roots = (
        priors.FunctionalCausalNode((), one, priors.JointRootBase("uniform", 1)),
        priors.FunctionalCausalNode(
            (),
            priors.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))),
            priors.JointRootBase("uniform", 2),
        ),
    )
    bad = priors.FunctionalCausalNode((0, 1), priors.FrozenParentFunction((two, one)))
    with pytest.raises(ValueError, match="slices"):
        priors.FunctionalCausalRecipe(request().task, (*roots, bad), ((0, 0),), ((2, 0),))
    graph = sample(priors.ParentCompositionPrior(mode="concat"))
    assert not isinstance(graph.runtime.nodes[-1].function, priors.FrozenParentFunction)
    with pytest.raises(ValueError, match="limit"):
        priors.recipe_from_dict(asdict(graph), limits=priors.CausalLimits(parameters=10))


def test_shared_named_selectors_and_finalized_ancestral_pipelines() -> None:
    original = request("tree")
    configured = replace(
        original,
        parents=((), (), (0, 1), (1, 2)),
        widths=(1, 2, 1, 1),
        target=((3, 0),),
        authored_pipelines=(
            priors.CausalNodePipeline(1, channel_importance=(2.0,)),
            None,
            priors.CausalNodePipeline(1, node_importance=3.0),
            None,
        ),
        noise=(
            priors.JointGraphNoise((0.1,)),
            priors.JointGraphNoise((0.2, 0.3)),
            priors.JointGraphNoise((0.4,)),
            priors.JointGraphNoise(),
        ),
    )
    graph = priors.sample_parent_graph(
        configured,
        composition=priors.ParentCompositionPrior(),
        named_rng=np.random.default_rng(114),
        raw_rng=np.random.default_rng(115),
        calibration_rng=np.random.default_rng(116),
        fit_rng=np.random.default_rng(117),
    )
    names = [draw.request.name for draw in graph.registry.decisions.categorical.draws]
    assert names.count("multi_function_func_type") == 2
    assert graph.replay() == graph
    assert graph.calibrations[-1].node == 3
    raw = json.loads(json.dumps(asdict(graph)))
    raw["calibrations"][-1]["parents"] = [0, 2]
    with pytest.raises(ValueError, match="differs from complete replay"):
        priors.parent_graph_from_dict(raw)
