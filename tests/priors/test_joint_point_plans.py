"""Point subplans retain source names and shared outer weight preferences."""

import copy
import json
from dataclasses import asdict

import numpy as np
from numpy.typing import NDArray

from r0.modules import priors


def test_joint_mixture_points_bind_count_and_shared_weight_batches() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(28), mode="global")
    plan = priors.sample_joint_points(context, "cloud", count=14, dimension=3)
    record = context.freeze()
    assert plan.law == "gaussian-mixture"
    assert plan.mixture_weights.width == plan.components
    assert plan.factor_weights.batch == plan.components
    assert plan.factor_weights.width == 3
    assert record.events[0].role == "components"
    assert record.numerical.draws[0].request.name == "random_gmp_n_centers"
    assert record.categorical.draws == ()
    assert [p.request.name for p in record.numerical.preferences] == [
        "random_gmp_n_centers",
        "simple_random_weights_decay_rate",
        "simple_random_weights_std_scale",
    ]
    assert set(plan.factor_weights.scales) == set(plan.mixture_weights.scales)
    plan.validate(record)
    assert (
        priors.joint_point_plan_from_dict(json.loads(json.dumps(asdict(plan))), decisions=record)
        == plan
    )


def test_joint_point_realization_preserves_source_raw_order_and_geometry() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(28), mode="global")
    plan = priors.sample_joint_points(
        context, "cloud", count=7, dimension=2, vector_family="uniform"
    )
    rng = np.random.default_rng(17)
    control = copy.deepcopy(rng)
    cloud = priors.materialize_joint_points(plan, context.freeze(), rng=rng)
    c = plan.components
    weights = control.random((1, c))[0]
    assignments = control.choice(c, size=7, p=weights / weights.sum())
    centers = control.normal(size=(c, 2))
    factors = control.normal(size=(c, 2, 2))
    factor_weights = control.random((c, 2))
    mean, coefficient = control.normal(size=2)
    scale_draws = control.normal(size=c)
    innovations = control.normal(size=(7, 2))
    expected: list[NDArray[np.float64]] = []
    for component, noise in zip(assignments, innovations, strict=True):
        expected.append(
            centers[component]
            + np.exp(mean + coefficient * scale_draws[component])
            * (factors[component] @ (factor_weights[component] * noise))
        )
    np.testing.assert_allclose(cloud.materialize(), expected, rtol=1e-12, atol=1e-12)
    assert cloud.assignments == tuple(assignments)
    np.testing.assert_array_equal(rng.random(8), control.random(8))
