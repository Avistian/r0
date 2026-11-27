"""Ancestral calibration evaluates admitted prefixes without artificial task readouts."""

import numpy as np
import pytest
import torch

from r0.modules import priors


def test_prefix_dimension_changes_and_parent_order_without_task_controls() -> None:
    first = priors.FunctionalCausalNode(
        (),
        priors.FrozenTreeFunction(1, ((0,),), ((0.0,),), (((2.0, 3.0), (2.0, 3.0)),)),
        root=priors.JointRootBase("uniform", 1),
    )
    second = priors.FunctionalCausalNode(
        (0,),
        priors.FrozenQuadraticFunction(
            2,
            (0, 1),
            (((1.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 0.0)),),
        ),
    )
    third = priors.FunctionalCausalNode((1, 0), priors.FrozenLinearFunction(((10.0, 1.0, 1.0),)))
    prefix = priors.FunctionalPrefix((first, second, third), rows=7)
    result = priors.evaluate_functional_prefix(prefix, rng=lambda *keys: np.random.default_rng(28))
    assert torch.equal(result.states[1], torch.full((7, 1), 31.0, dtype=torch.float64))
    assert torch.equal(result.states[2], torch.full((7, 1), 315.0, dtype=torch.float64))
    assert torch.equal(result.readouts[2], result.states[2])


def test_prefix_preserves_converter_capture_and_noisy_descendant_state() -> None:
    root = priors.FunctionalCausalNode(
        (),
        priors.FrozenTreeFunction(1, ((0,),), ((0.0,),), (((2.0,), (2.0,)),)),
        root=priors.JointRootBase("gaussian", 1),
        noise_scale=(3.0,),
        pipeline=priors.CausalNodePipeline(
            1,
            channel_importance=(2.0,),
            node_importance=5.0,
            converters=(priors.NumericalConverter(0),),
        ),
    )
    descendant = priors.FunctionalCausalNode((0,), priors.FrozenLinearFunction(((2.0,),)))
    prefix = priors.FunctionalPrefix((root, descendant), rows=9)
    raw = np.random.default_rng(23).standard_normal(9)
    expected = torch.tensor((4.0 + 3.0 * raw) * 5.0, dtype=torch.float64)[:, None]
    result = priors.evaluate_functional_prefix(prefix, rng=lambda *keys: np.random.default_rng(23))
    assert torch.equal(result.readouts[0], torch.full((9, 1), 4.0, dtype=torch.float64))
    torch.testing.assert_close(result.states[0], expected)
    torch.testing.assert_close(result.states[1], 2.0 * expected)


def test_prefix_rejects_invalid_ancestry_and_resources_before_evaluation() -> None:
    root = priors.FunctionalCausalNode(
        (),
        priors.FrozenLinearFunction(((1.0,),)),
        root=priors.JointRootBase("gaussian", 1),
    )
    with pytest.raises(ValueError, match="working memory|storage limit"):
        priors.FunctionalPrefix((root,), rows=32, limits=priors.CausalLimits(working_bytes=1))
    child = priors.FunctionalCausalNode((1,), priors.FrozenLinearFunction(((1.0,),)))
    with pytest.raises(ValueError, match="parent-before-child"):
        priors.FunctionalPrefix((root, child), rows=32)
    with pytest.raises(ValueError, match="rows"):
        priors.FunctionalPrefix((root,), rows=True)
