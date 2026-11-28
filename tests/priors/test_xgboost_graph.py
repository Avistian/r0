"""Public frozen XGBoost graph integration and archive boundaries."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from r0.modules import priors as p


def model():
    return p.FrozenXGBoostVector(
        1,
        (0.25, -0.5),
        (
            p.XGBoostVectorTree(
                (0, -1, -1),
                (0.5, 0.0, 0.0),
                (1, -1, -1),
                (2, -1, -1),
                (True, False, False),
                ((0.0, 0.0), (1.0, -2.0), (-3.0, 4.0)),
            ),
        ),
    )


def test_public_xgboost_routes_and_roundtrips_without_backend():
    function = model()
    reopened = p.frozen_function_from_dict(json.loads(json.dumps(asdict(function))))
    assert reopened == function
    inputs = torch.tensor([[0.49], [0.5], [0.51]], dtype=torch.float64)
    expected = torch.tensor([[1.25, -2.5], [-2.75, 3.5], [-2.75, 3.5]], dtype=torch.float64)
    torch.testing.assert_close(
        p.evaluate_frozen_function(reopened, inputs), expected, rtol=0, atol=0
    )


def test_xgboost_function_works_as_parent_member():
    function = p.FrozenParentFunction((model(), model()), aggregation="sum")
    reopened = p.frozen_function_from_dict(asdict(function))
    inputs = torch.tensor([[0.2, 0.6]], dtype=torch.float64)
    torch.testing.assert_close(
        p.evaluate_frozen_function(reopened, inputs),
        torch.tensor([[-1.5, 1.0]], dtype=torch.float64),
    )


@pytest.mark.parametrize("field", ["initial", "trees", "backend_version", "objective"])
def test_xgboost_archive_requires_complete_fields(field: str):
    data = asdict(model())
    del data[field]
    with pytest.raises(ValueError, match="fields"):
        p.frozen_function_from_dict(data)


def test_xgboost_caller_admission_rejects_before_evaluation():
    with pytest.raises(ValueError, match="parameter|limit|budget"):
        p.evaluate_frozen_function(
            model(), torch.zeros((2, 1)), limits=p.CausalLimits(parameters=1)
        )


def test_xgboost_graph_generation_trace_and_archive(tmp_path: Path):
    from r0.modules import generation as g

    root = p.FunctionalCausalNode((), model(), p.JointRootBase("uniform", 1))
    child = p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0, -3.0),)))
    recipe = p.FunctionalCausalRecipe(
        p.LinearRecipe(rows=16, features=2, signal_features=2, active_features=2, context_rows=8),
        (root, child),
        ((0, 0), (0, 1)),
        ((1, 0),),
    )
    assert p.recipe_from_dict(asdict(recipe)) == recipe
    identity = 2**54 + 7
    task = p.generate_task(recipe, seed=19, task_id=identity)
    torch.testing.assert_close(task.scores[:, 0], 2 * task.inputs[:, 0] - 3 * task.inputs[:, 1])
    trace = p.explain_generation(task, [15, 1, 15])
    assert trace["signal_max_error"] == 0
    path = tmp_path / "xgboost.zip"
    g.save_group(path, [(identity, task)], max_bytes=4 * 1024**2)
    restored = next(g.load_group(path))
    assert restored.recipe_json == task.recipe_json
    torch.testing.assert_close(restored.inputs, task.inputs, rtol=0, atol=0)
    assert p.recipe_from_dict(json.loads(restored.recipe_json)["recipe"]) == recipe


@pytest.mark.parametrize("kind", ["native", "vector", "converted"])
def test_xgboost_typed_graphs_preserve_vector_leaf_semantics(kind: str):
    nodes = (
        p.FunctionalCausalNode((), model(), p.JointRootBase("uniform", 1)),
        p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0, -3.0),))),
    )
    task = p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2)
    if kind == "native":
        recipe = p.NativeCausalRecipe(
            task, tuple(p.NativeCausalNode(n) for n in nodes), ((0, 0), (0, 1)), (1, 0)
        )
    elif kind == "vector":
        recipe = p.NativeVectorRecipe(
            task,
            tuple(
                p.NativeVectorNode(n, (p.VariableDomain(),) * n.function.output_width)
                for n in nodes
            ),
            ((0, 0), (0, 1)),
            (1, 0),
        )
    else:
        recipe = p.ConvertedCausalRecipe(
            task,
            tuple(p.ConvertedCausalNode(n) for n in nodes),
            (p.ConvertedReadout(0, channel=0), p.ConvertedReadout(0, channel=1)),
            p.ConvertedReadout(1, channel=0),
        )
    assert p.recipe_from_dict(asdict(recipe)) == recipe
    result = p.generate_task(recipe, seed=13, task_id=2**54 + 9)
    torch.testing.assert_close(result.targets, result.inputs[:, 0] * 2 - result.inputs[:, 1] * 3)
    trace = p.explain_generation(result, [15, 0, 15])
    assert trace["rows"] == [15, 0, 15]


@pytest.mark.parametrize(
    "field", ["features", "thresholds", "left", "right", "default_left", "values"]
)
def test_xgboost_tree_archive_rejects_missing_fields(field: str):
    data = asdict(model())
    del data["trees"][0][field]
    with pytest.raises(ValueError, match="fields"):
        p.frozen_function_from_dict(data)
