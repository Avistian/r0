"""Fixed-row reference includes all channels, not truncated inspection previews."""

import torch

from r0.modules import priors as p


def test_original_row_reference_over_32_channels_matches_generation():
    width = 40
    source = p.NetworkRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1),
        p.FrozenNetworkFunction(
            1,
            (
                p.NetworkBlock(
                    (0,),
                    (
                        p.NetworkAffine(
                            tuple((i / 20,) for i in range(width)),
                            (0.1,) * width,
                            activation=p.CausalActivation("tanh"),
                        ),
                    ),
                ),
                p.NetworkBlock(
                    (1,), (p.NetworkAffine((tuple(0.1 * (i + 1) for i in range(width)),), (0.2,)),)
                ),
            ),
        ),
    )
    recipe = p.with_network_innovations(
        source,
        p.NetworkInnovationPlan(
            (
                p.NetworkInnovationStage((0.0,) * width, (0.2,) * width),
                p.NetworkInnovationStage((0.1,), (0.3,)),
            )
        ),
    )
    task = p.generate_task(recipe, seed=71, task_id=2**54 + 3)
    draws = p.network_innovation_row(recipe, seed=71, task_id=2**54 + 3, row=15)
    assert draws[2].shape == (1, 40)
    actual = p.evaluate_network_innovation_reference(recipe, task.inputs[15:16], innovations=draws)
    torch.testing.assert_close(actual, task.scores[15:16], rtol=0, atol=0)


def test_task_only_pair_preserves_compiled_architecture_and_statistics():
    import numpy as np

    source = p.NetworkRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1),
        p.FrozenNetworkFunction(1, (p.NetworkBlock((0,), (p.NetworkAffine(((1.0,),), (0.2,)),)),)),
    )
    recipe = p.sample_network_innovations(
        source,
        p.NetworkInnovationRequest(p.NetworkScalarPrior("uniform", 0.1, 0.3)),
        statistics_rng=np.random.default_rng(5),
    )
    changed = p.paired_recipe(recipe, intervention="noise", value=0.5)
    assert isinstance(changed, p.NetworkInnovationRecipe)
    assert changed.nodes == recipe.nodes
    assert changed.sampling == recipe.sampling
    assert changed.source is not None
    assert changed.source.task == changed.task
    pair = p.generate_pair(recipe, intervention="noise", value=0.5, seed=7, task_id=3)
    torch.testing.assert_close(pair.before.scores, pair.after.scores, rtol=0, atol=0)
    torch.testing.assert_close(pair.before.inputs, pair.after.inputs, rtol=0, atol=0)
