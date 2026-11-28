"""Source roots participate in ordinary graph generation and original-row traces."""

from dataclasses import asdict

import numpy as np
import pytest
import torch

from r0.modules import priors as p


def test_source_root_graph_roundtrip_generation_and_trace():
    law = p.SourceRootLaw(
        (
            p.SourceRootCoordinate("categorical", probabilities=(0.2, 0.8)),
            p.SourceRootCoordinate("zipf", exponent=2.5, clip=10),
        )
    )
    base = p.SourceRootBase(law)
    root = p.FunctionalCausalNode(
        (),
        p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))),
        root=base,
        version="functional-causal-node-v2",
    )
    child = p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0, 3.0),)))
    recipe = p.FunctionalCausalRecipe(
        p.LinearRecipe(rows=32, features=2, signal_features=2, active_features=2, context_rows=16),
        (root, child),
        ((0, 0), (0, 1)),
        ((1, 0),),
    )
    assert p.recipe_from_dict(asdict(recipe)) == recipe
    task = p.generate_task(recipe, seed=7, task_id=3)
    torch.testing.assert_close(task.scores[:, 0], 2 * task.inputs[:, 0] + 3 * task.inputs[:, 1])
    trace = p.explain_generation(task, [31, 2, 31])
    assert trace["signal_max_error"] == 0
    assert trace["target_max_error"] == 0
    with pytest.raises(ValueError, match="version"):
        p.FunctionalCausalNode((), root.function, root=base)


def test_source_root_base_frozen_transform_matches_calibration():
    law = p.SourceRootLaw((p.SourceRootCoordinate("gaussian"),))
    fit = p.calibrate_source_root(
        law, modes=("standardize",), rows=32, rng=np.random.default_rng(8)
    )
    base = p.SourceRootBase(law, fit)
    assert base.width == 1
    assert base.moment_order == float("inf")


def test_source_root_network_views_and_variance_matching():
    from dataclasses import replace

    law = p.SourceRootLaw(
        (
            p.SourceRootCoordinate("categorical", probabilities=(0.2, 0.8)),
            p.SourceRootCoordinate("zipf", exponent=2.5, clip=10),
        )
    )
    fit = p.calibrate_source_root(
        law, modes=("standardize", "center"), rows=128, rng=np.random.default_rng(8)
    )
    recipe = p.NetworkRecipe(
        p.LinearRecipe(rows=32, features=2, signal_features=2, active_features=2, context_rows=16),
        p.FrozenNetworkFunction(
            2, (p.NetworkBlock((0,), (p.NetworkAffine(((2.0, 3.0),), (1.0,)),)),)
        ),
        p.SourceRootBase(law, fit),
    )
    assert p.recipe_from_dict(asdict(recipe)) == recipe
    task = p.generate_task(recipe, seed=7, task_id=3)
    torch.testing.assert_close(task.scores[:, 0], 2 * task.inputs[:, 0] + 3 * task.inputs[:, 1] + 1)
    matching = p.fit_network_variance_match(
        recipe, before_inputs=(0, 1), after_inputs=(1,), rows=128, rng=np.random.default_rng(8)
    )
    view = p.MatchedNetworkInputView(
        recipe, replace(recipe.task, features=3, active_features=1), (1,), matching=matching
    )
    assert p.recipe_from_dict(asdict(view)) == view
    emitted = p.generate_task(view, seed=7, task_id=3)
    trace = p.explain_generation(emitted, [31, 2, 31])
    assert trace["signal_max_error"] == 0
    replay = p.generate_task(p.recipe_from_dict(asdict(view)), seed=7, task_id=3)
    assert torch.equal(replay.targets, emitted.targets)


def test_unbounded_zipf_cannot_claim_finite_variance_for_target_calibration():
    base = p.SourceRootBase(p.SourceRootLaw((p.SourceRootCoordinate("zipf", exponent=2.5),)))
    root = p.FunctionalCausalNode(
        (), p.FrozenLinearFunction(((1.0,),)), root=base, version="functional-causal-node-v2"
    )
    child = p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((1.0,),)))
    with pytest.raises(ValueError, match="finite second"):
        p.CalibratedFunctionalCausalRecipe(
            p.LinearRecipe(
                rows=32, features=1, signal_features=1, active_features=1, context_rows=16
            ),
            (root, child),
            ((0, 0),),
            ((1, 0),),
        )


def test_source_root_prefix_late_transform_failure_preserves_rng():
    import copy

    law = p.SourceRootLaw((p.SourceRootCoordinate("gaussian"),))
    fit = p.SourceRootCalibration(law, ("standardize",), 32, (0.0,), (1e-320,))
    root = p.FunctionalCausalNode(
        (),
        p.FrozenLinearFunction(((1.0,),)),
        root=p.SourceRootBase(law, fit),
        version="functional-causal-node-v2",
    )
    prefix = p.FunctionalPrefix((root,), rows=8)
    rng = np.random.default_rng(7)
    oracle = copy.deepcopy(rng)
    ledger = p.RawDrawLedger()
    with pytest.raises(ValueError, match="nonfinite"):
        p.sample_functional_prefix(prefix, rng=rng, owner=("root-probe",), capture=ledger)
    assert ledger.freeze() == ()
    np.testing.assert_array_equal(rng.random(5), oracle.random(5))
