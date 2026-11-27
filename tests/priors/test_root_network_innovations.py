"""Source-root custody survives all architectures and explicit noise references."""

from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from r0.modules import priors as p


def root() -> p.SampledSourceRootBase:
    profile = p.sample_source_root_profile(
        p.SourceRootProfile(
            2, selector="forced-r0", families=("presampled_gaussian", "zipf_clipped")
        ),
        model_rng=np.random.default_rng(17),
        selection_rng=np.random.default_rng(18),
    )
    return p.sample_source_root_base(
        profile,
        modes=("standardize", "center"),
        calibration_rows=32,
        calibration_rng=np.random.default_rng(19),
    )


@pytest.mark.parametrize(
    "family",
    ["dense", "individual-sparse", "block-sparse", "skip", "residual", "irregular", "gated"],
)
def test_every_architecture_retains_sampled_root_and_statistics(family: str):
    network = p.sample_network(
        p.NetworkSamplingRequest(2, (3, 1), families=(family,)),
        named_rng=np.random.default_rng(2),
        raw_rng=np.random.default_rng(3),
    )
    source = p.NetworkRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2),
        network,
        root(),
    )
    recipe = p.sample_network_innovations(
        source,
        p.NetworkInnovationRequest(
            p.NetworkScalarPrior("uniform", 0.1, 0.3),
            importance=p.NetworkScalarPrior("uniform", 0.2, 1.5),
        ),
        statistics_rng=np.random.default_rng(4),
    )
    assert recipe.nodes[0].version == "functional-causal-node-v2"
    assert recipe.nodes[0].root == source.root
    decoded = p.recipe_from_dict(asdict(recipe))
    assert decoded == recipe
    before = p.generate_task(recipe, seed=13, task_id=2**54 + 3)
    after = p.generate_task(decoded, seed=13, task_id=2**54 + 3)
    assert torch.equal(before.inputs, after.inputs)
    assert torch.equal(before.targets, after.targets)
    bad = asdict(recipe)
    bad["source"]["root"]["calibration"]["locations"] = (999.0, 999.0)
    with pytest.raises(ValueError, match="complete replay"):
        p.recipe_from_dict(bad)


def test_reference_uses_observed_root_outputs_once():
    function = p.FrozenNetworkFunction(
        2, (p.NetworkBlock((0,), (p.NetworkAffine(((2.0, -1.0),), (0.3,)),)),)
    )
    source = p.NetworkRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2),
        function,
        root(),
    )
    recipe = p.with_network_innovations(
        source,
        p.NetworkInnovationPlan((p.NetworkInnovationStage((0.2,), (0.4,)),), importance=(3.0, 0.5)),
    )
    x = torch.tensor([[0.2, -0.8], [-1.0, 0.9]], dtype=torch.float64)
    z = torch.tensor([[1.7], [-0.6]], dtype=torch.float64)
    expected = x.numpy() @ np.array([6.0, -0.5]) + 0.5 + 0.4 * z.numpy()[:, 0]
    actual = p.evaluate_network_innovation_reference(recipe, x, innovations={2: z})
    np.testing.assert_allclose(actual.numpy()[:, 0], expected, rtol=1e-14, atol=1e-14)
    assert recipe.plan is not None
    no_root = p.with_network_innovations(replace(source, root=None), recipe.plan)
    assert torch.equal(
        actual, p.evaluate_network_innovation_reference(no_root, x, innovations={2: z})
    )
