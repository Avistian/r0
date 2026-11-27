"""Independent fixed-latent causal slices, including nonlinear observed roots."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from r0.modules import priors as p


def graph() -> p.FunctionalCausalRecipe:
    root = p.FrozenXGBoostVector(
        1,
        (0.0, 0.0),
        (
            p.XGBoostVectorTree(
                (0, -1, -1),
                (0.0, 0.0, 0.0),
                (1, -1, -1),
                (2, -1, -1),
                (True, False, False),
                ((0.0, 0.0), (-2.0, 3.0), (2.0, 5.0)),
            ),
        ),
    )
    return p.FunctionalCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1),
        (
            p.FunctionalCausalNode((), root, p.JointRootBase("gaussian", 1)),
            p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0, 3.0),)), noise_scale=(0.5,)),
        ),
        ((0, 0),),
        ((1, 0),),
    )


def draw(keys: tuple[object, ...]) -> np.ndarray:
    encoded = json.dumps(
        ("functional-causal-v1", 17, 29, keys), sort_keys=True, separators=(",", ":")
    )
    seed = int(hashlib.sha256(encoded.encode()).hexdigest(), 16)
    return np.random.default_rng(seed).normal(size=16)


@pytest.mark.parametrize("reference", ["zero", "original-row"])
def test_literal_hidden_root_and_noise_reference(reference: str):
    recipe = graph()
    inputs = torch.tensor([[-1.5], [0.0], [1.5]], dtype=torch.float64)
    actual = p.evaluate_functional_reference(
        recipe, inputs, seed=17, task_id=29, row=3, reference=reference
    )
    hidden = 3.0 if np.float32(draw(("root-base", 0))[3]) < 0 else 5.0
    noise = 0.5 * draw(("innovation", 1, 0))[3] if reference == "original-row" else 0.0
    expected = inputs.numpy() * 2 + hidden * 3 + noise
    np.testing.assert_allclose(actual.numpy(), expected, rtol=0, atol=1e-12)
    # The curve consumes observed root outputs, not the original root input again.
    assert float(actual[1, 0]) > float(actual[0, 0])
    assert float(actual[2, 0]) > float(actual[1, 0])


def test_pipeline_and_internal_observed_clamps():
    recipe = graph()
    # A frozen pipeline is applied before explicit state/readout clamps.
    root = replace(recipe.nodes[0], pipeline=p.CausalNodePipeline(channels=2, node_importance=2.0))
    recipe = replace(recipe, nodes=(root, recipe.nodes[1]))
    inputs = torch.tensor([[-1.0], [2.0]], dtype=torch.float64)
    actual = p.evaluate_functional_reference(
        recipe, inputs, seed=17, task_id=29, row=3, reference="zero"
    )
    hidden = (3.0 if np.float32(draw(("root-base", 0))[3]) < 0 else 5.0) * 2
    np.testing.assert_array_equal(actual.numpy(), inputs.numpy() * 2 + hidden * 3)


def test_reference_rejects_geometry_and_nonfinite():
    for bad in (
        torch.zeros((0, 1), dtype=torch.float64),
        torch.tensor([[float("nan")]], dtype=torch.float64),
    ):
        with pytest.raises(ValueError, match="reference"):
            p.evaluate_functional_reference(graph(), bad, seed=17, task_id=29, row=3)
    with pytest.raises(ValueError, match="reference"):
        p.evaluate_functional_reference(
            graph(), torch.zeros((2, 1), dtype=torch.float64), seed=True, task_id=29, row=3
        )


def test_query_budget_precedes_finite_mask(monkeypatch: pytest.MonkeyPatch):
    source = graph()
    boundary = source.admit()["estimated_working_bytes"]
    bounded = replace(source, limits=p.CausalLimits(working_bytes=boundary + 1000))
    inputs = torch.zeros((512, 1), dtype=torch.float64)

    def forbidden(*args: object, **kwargs: object):
        raise AssertionError("finite mask allocated before query admission")

    monkeypatch.setattr(torch, "isfinite", forbidden)
    with pytest.raises(ValueError, match="working"):
        p.evaluate_functional_reference(bounded, inputs, seed=17, task_id=29, row=3)
