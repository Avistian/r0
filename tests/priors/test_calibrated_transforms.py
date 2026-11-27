"""Independent calibration fixes source normalization and range-warp functions."""

import json
from dataclasses import asdict, replace
from decimal import Decimal, localcontext

import numpy as np
import pytest
import torch

from r0.modules import priors


def test_scalar_l2_calibration_keeps_channels_and_task_rows_independent() -> None:
    transform = priors.calibrate_l2_normalization(
        np.array([[3.0, 4.0], [-3.0, -4.0]]), calibration_id="independent-3-4-norm"
    )
    assert transform.factor == pytest.approx(1 / 5.00000001, rel=1e-15)
    controls = priors.LinearRecipe(
        rows=16, context_rows=8, features=1, signal_features=1, active_features=1
    )
    recipe = priors.CausalRecipe(
        task=controls,
        nodes=(
            priors.ActivatedCausalNode(
                bias=(6.0, 8.0),
                noise_scale=(0.0, 0.0),
                output_activation=transform,
            ),
        ),
        observed=((0, 1),),
        target=((0, 0),),
        version="causal-v2",
    )
    restored = priors.recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
    assert restored == recipe
    for rows in (16, 32):
        task = priors.generate_task(
            replace(recipe, task=replace(controls, rows=rows, context_rows=rows // 2)),
            seed=28,
            task_id=0,
        )
        torch.testing.assert_close(
            task.scores,
            torch.full((rows, 1), 1.1999999976, dtype=torch.float64),
            rtol=1e-15,
            atol=1e-15,
        )
        assert priors.explain_generation(task, [0, rows - 1])["target_max_error"] < 1e-14


def test_kumaraswamy_retains_source_epsilon_domain_clamp_and_frozen_range() -> None:
    warp = priors.calibrate_kumaraswamy(
        np.array([[0.0, 0.0, 7.0], [1.0, 1e-30, 7.0]]),
        a=2.0,
        b=3.0,
        calibration_id="independent-worked-ranges",
    )
    value = torch.tensor([[0.5, 1e-30, 7.0], [2.0, 2e-30, 8.0]], dtype=torch.float64)
    # Source epsilon means the tiny range's maximum maps to u=1/2, not one.
    torch.testing.assert_close(
        warp.apply(value),
        torch.tensor([[0.578125, 0.578125, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64),
        rtol=1e-14,
        atol=1e-14,
    )
    controls = priors.LinearRecipe(
        rows=16, context_rows=8, features=1, signal_features=1, active_features=1
    )
    recipe = priors.CausalRecipe(
        task=controls,
        nodes=(
            priors.ActivatedCausalNode(
                bias=(0.5, 1e-30, 8.0),
                noise_scale=(0.0, 0.0, 0.0),
                output_activation=warp,
            ),
        ),
        observed=((0, 1),),
        target=((0, 0),),
        version="causal-v2",
    )
    restored = priors.recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
    assert restored == recipe
    task = priors.generate_task(restored, seed=28, task_id=0)
    assert "kumaraswamy_domain_clamp" in json.loads(task.mechanism_json)["flags"]
    assert priors.explain_generation(task, [0, 15])["target_max_error"] < 1e-14


def test_kumaraswamy_preserves_representable_tails_across_extreme_ranges() -> None:
    smallest = float.fromhex("0x0.0000000000001p-1022")
    warp = priors.calibrate_kumaraswamy(
        np.array([[0.0], [1e308]]), a=0.2, b=1.0, calibration_id="independent-wide-range"
    )
    with localcontext() as context:
        context.prec = 100
        expected = float(
            (Decimal.from_float(smallest) / Decimal.from_float(1e308)) ** Decimal.from_float(0.2)
        )
    actual = float(warp.apply(torch.tensor([[smallest]], dtype=torch.float64))[0, 0])
    assert actual == pytest.approx(expected, rel=1e-12, abs=0)
    endpoint = priors.calibrate_kumaraswamy(
        np.array([[0.0], [1.0]]), a=1.0, b=0.2, calibration_id="independent-endpoint"
    )
    # Epsilon is tiny before the power, but its fifth root is observable.
    assert float(endpoint.apply(torch.ones((1, 1), dtype=torch.float64))[0, 0]) == (
        pytest.approx(0.999999, abs=1e-14)
    )


def test_authored_positive_shapes_preserve_subnormal_exponent_effects() -> None:
    smallest = float.fromhex("0x0.0000000000001p-1022")
    warp = priors.calibrate_kumaraswamy(
        np.array([[0.0], [1.0]]),
        a=smallest,
        b=1e-308,
        calibration_id="independent-authored-shape-extension",
    )
    with localcontext() as context:
        context.prec = 100
        expected = float(
            -Decimal.from_float(1e-308)
            * (Decimal.from_float(smallest) * (-Decimal("0.5").ln())).ln()
        )
    actual = float(warp.apply(torch.tensor([[0.5]], dtype=torch.float64))[0, 0])
    assert actual == pytest.approx(expected, rel=1e-12, abs=0)


def test_l2_avoids_squared_norm_overflow_and_admits_fitted_metadata() -> None:
    transform = priors.calibrate_l2_normalization(
        np.full((2, 2), 1e308),
        calibration_id="independent-large-l2",
        limits=priors.CausalLimits(parameters=7),
    )
    torch.testing.assert_close(
        transform.apply(torch.full((1, 2), 1e308, dtype=torch.float64)),
        torch.full((1, 2), 2**-0.5, dtype=torch.float64),
        rtol=1e-14,
        atol=0,
    )
    with pytest.raises(ValueError, match="parameter limit"):
        priors.calibrate_l2_normalization(
            np.ones((2, 2)),
            calibration_id="independent-budget",
            limits=priors.CausalLimits(parameters=6),
        )
    with pytest.raises(ValueError, match="factor"):
        replace(transform, factor=0.0)


def test_bounded_warp_certifies_noise_before_output_but_not_noise_after() -> None:
    transform = priors.calibrate_kumaraswamy(
        np.array([[-2.0], [2.0]]),
        a=0.7,
        b=1.3,
        calibration_id="independent-bounded-warp",
    )
    node = priors.StochasticCausalNode(
        exogenous=priors.ExogenousDistribution(
            inputs=priors.InputDistribution(
                distribution="student_t",
                df=0.5,
            )
        ),
        output_activation=transform,
    )
    recipe = priors.CausalRecipe(
        task=priors.LinearRecipe(
            rows=16, context_rows=8, features=1, signal_features=1, active_features=1
        ),
        nodes=(node, priors.CausalNode()),
        observed=((1, 0),),
        target=((0, 0),),
        normalize=True,
        calibration_rows=32,
        version="causal-v2",
    )
    task = priors.generate_task(recipe, seed=28, task_id=0)
    assert json.loads(task.mechanism_json)["moment_contract"]["target_orders"] == ["all"]
    with pytest.raises(ValueError, match="second moments"):
        replace(recipe, nodes=(replace(node, noise_position="after_output"), recipe.nodes[1]))
