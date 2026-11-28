"""Kumo's broad same-input products retain source eligibility, fits and replay."""

import json
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from r0.modules import priors


def prior() -> priors.JointFunctionPrior:
    return priors.JointFunctionPrior(
        functions="kumo-prod", frequencies=4, matrix=priors.JointMatrixPrior(families=("gaussian",))
    )


def test_nonlinear_mlp_times_cluster_has_independent_numpy_oracle() -> None:
    data = np.array([[-2.0, 1.0], [0.0, -1.0], [1.0, 2.0], [3.0, 0.0]])
    activation = priors.calibrate_source_activation(
        priors.CausalActivation("relu"),
        data,
        rng=np.random.default_rng(4),
        scale=1.2,
        calibration_id="independent-function-fixture",
    )
    matrix = np.array([[1.0, 2.0], [-1.0, 0.5]])
    mlp = priors.FrozenMLPFunction(
        (priors.FrozenLinearFunction(((1.0, 2.0), (-1.0, 0.5))),),
        (activation,),
        start_activation=True,
    )
    cluster = priors.FrozenEMFunction(
        ((-1.0, 0.0), (2.0, 1.0)), (1.0, 2.0), priors.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0)))
    )
    product = priors.FrozenKumoProductFunction(mlp, cluster)
    x = np.array([[-1.5, 0.25], [0.5, 1.5], [3.0, -2.0]])
    shifted = activation.scale * (
        (x - activation.input_mean) * activation.input_inverse_std + activation.shift
    )
    neural = (
        (np.maximum(shifted, 0) - activation.output_mean) * activation.output_inverse_std
    ) @ matrix.T
    distance = np.sqrt(((x[:, None, :] - np.array(cluster.centers)[None, :, :]) ** 2).sum(2))
    logits = (
        -np.log(np.array(cluster.scales))[None, :] - distance / np.array(cluster.scales)[None, :]
    )
    weights = np.exp(logits - logits.max(1, keepdims=True))
    weights /= weights.sum(1, keepdims=True)
    np.testing.assert_allclose(
        priors.evaluate_frozen_function(product, torch.tensor(x)).numpy(), neural * weights
    )
    assert priors.frozen_function_from_dict(json.loads(json.dumps(asdict(product)))) == product
    with pytest.raises(ValueError, match="cheap"):
        priors.FrozenProductFunction(mlp, cluster)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonrecursive"):
        priors.FrozenKumoProductFunction(product, cluster)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("seed", "left", "right"),
    [
        (72, "mlp", "em"),
        (12, "em", "mlp"),
        (114, "mlp", "mlp"),
        (28, "em", "em"),
        (0, "tree", "gp"),
        (6, "disc", "quad"),
        (61, "gp", "lin"),
    ],
)
def test_sampled_product_retains_closed_catalog_and_shared_original_fit(
    seed: int, left: str, right: str
) -> None:
    context = priors.JointSamplingContext(np.random.default_rng(seed), mode="local")
    plan = priors.sample_joint_function(
        context, "f", input_width=3, output_width=2, calibration_rows=8, prior=prior()
    )
    assert tuple(c.family for c in plan.children) == (left, right)
    assert plan.prior.product_children == ("mlp", "tree", "disc", "lin", "quad", "gp", "em")
    decisions = context.freeze()
    population = np.random.default_rng(999).normal(size=(8, 3))
    realized = priors.materialize_joint_function(
        plan,
        decisions,
        rng=np.random.default_rng(10),
        fit_rng=np.random.default_rng(11),
        calibration=population,
        calibration_id="independent-ancestor-fixture",
    )
    assert isinstance(realized, priors.JointProductFunctionRealization)
    assert isinstance(realized.function, priors.FrozenKumoProductFunction)
    assert realized.version == "joint-kumo-product-function-realization-v1"
    fitted = [
        c for c in (realized.left, realized.right) if getattr(c, "calibration", None) is not None
    ]
    for child in fitted:
        assert getattr(child, "calibration", None) == realized.calibration
    if fitted:
        assert realized.calibration is not None
        np.testing.assert_array_equal(realized.calibration.sample, population)
    restored = priors.joint_function_realization_from_dict(
        json.loads(json.dumps(asdict(realized))), decisions=decisions
    )
    assert restored == realized


def test_historical_catalog_cannot_be_relabelled_as_broad_product() -> None:
    legacy = priors.JointFunctionPrior(functions="prod")
    assert legacy.product_children == ("gp", "tree", "disc", "lin", "quad")
    assert legacy.request(cheap=True).name == "cheap_random_function_fct_type"
    assert prior().request(cheap=True).name == "kumo_product_function_fct_type_v1"
    assert (
        priors.JointFunctionPrior(functions="lin,kumo-prod").family("function-slot-1-prod-v1")
        == "prod"
    )
    context = priors.JointSamplingContext(np.random.default_rng(72), mode="local")
    plan = priors.sample_joint_function(
        context, "f", input_width=3, output_width=2, calibration_rows=8, prior=prior()
    )
    with pytest.raises(ValueError, match="product child"):
        replace(plan, prior=legacy)


def test_broad_product_admission_preserves_both_streams_on_rejection() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(72), mode="local")
    plan = priors.sample_joint_function(
        context, "f", input_width=3, output_width=2, calibration_rows=8, prior=prior()
    )
    raw, fit = np.random.default_rng(10), np.random.default_rng(11)
    before = json.dumps((raw.bit_generator.state, fit.bit_generator.state))
    with pytest.raises(ValueError, match="limit"):
        priors.materialize_joint_function(
            plan,
            context.freeze(),
            rng=raw,
            fit_rng=fit,
            calibration=np.ones((8, 3)),
            calibration_id="admission-rejection",
            limits=priors.CausalLimits(parameters=100),
        )
    assert json.dumps((raw.bit_generator.state, fit.bit_generator.state)) == before
