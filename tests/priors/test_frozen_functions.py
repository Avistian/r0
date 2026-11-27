"""Frozen function families expose source equations without emission-time fitting."""

import json
import math
from dataclasses import asdict

import numpy as np
import pytest
import torch

from r0.modules import priors


def test_frozen_linear_and_quadratic_keep_source_orientation_and_constant_term() -> None:
    linear = priors.FrozenLinearFunction(matrix=((1.0, 2.0), (3.0, 4.0)))
    result = priors.evaluate_frozen_function(
        linear, torch.tensor([[2.0, -1.0]], dtype=torch.float64)
    )
    torch.testing.assert_close(result, torch.tensor([[0.0, 2.0]], dtype=torch.float64))
    quadratic = priors.FrozenQuadraticFunction(
        input_width=1,
        indices=(0,),
        matrices=(((1.0, 3.0), (5.0, 7.0)),),
    )
    result = priors.evaluate_frozen_function(
        quadratic, torch.tensor([[2.0], [0.0]], dtype=torch.float64)
    )
    torch.testing.assert_close(result, torch.tensor([[27.0], [7.0]], dtype=torch.float64))


def test_frozen_oblivious_trees_use_strict_split_bits_and_average_leaves() -> None:
    function = priors.FrozenTreeFunction(
        input_width=2,
        features=((0, 1), (0, 1)),
        thresholds=((1.0, 2.0), (1.0, 2.0)),
        leaves=(((10.0,), (20.0,), (30.0,), (40.0,)), ((1.0,), (2.0,), (6.0,), (8.0,))),
    )
    values = torch.tensor([[1.0, 3.0], [2.0, 1.0]], dtype=torch.float64)
    result = priors.evaluate_frozen_function(function, values)
    torch.testing.assert_close(result, torch.tensor([[18.0], [11.0]], dtype=torch.float64))


def test_frozen_discretization_keeps_center_order_and_source_quasi_distance() -> None:
    function = priors.FrozenDiscretizationFunction(
        centers=((-1.0,), (1.0,)),
        linear=priors.FrozenLinearFunction(((2.0,),)),
        p=0.5,
    )
    values = torch.tensor([[0.0], [0.75]], dtype=torch.float64)
    result = priors.evaluate_frozen_function(function, values)
    torch.testing.assert_close(result, torch.tensor([[-2.0], [2.0]], dtype=torch.float64))
    reversed_function = priors.FrozenDiscretizationFunction(
        centers=((1.0,), (-1.0,)),
        linear=function.linear,
        p=0.5,
    )
    result = priors.evaluate_frozen_function(reversed_function, values[:1])
    torch.testing.assert_close(result, torch.tensor([[2.0]], dtype=torch.float64))


def test_frozen_fourier_function_keeps_frequencies_phases_and_output_coefficients() -> None:
    function = priors.FrozenRFFFunction(
        frequencies=((math.pi / 2,),),
        phases=(0.0,),
        coefficients=((2.0,),),
    )
    values = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float64)
    result = priors.evaluate_frozen_function(function, values)
    torch.testing.assert_close(result, torch.tensor([[2.0], [0.0], [-2.0]], dtype=torch.float64))


def test_frozen_em_preserves_source_scale_constant_without_dimension_multiplier() -> None:
    function = priors.FrozenEMFunction(
        centers=((0.0, 0.0), (0.0, 0.0)),
        scales=(1.0, 2.0),
        p=2.0,
        q=1.0,
        linear=priors.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))),
    )
    result = priors.evaluate_frozen_function(function, torch.zeros((1, 2), dtype=torch.float64))
    torch.testing.assert_close(result, torch.tensor([[2 / 3, 1 / 3]], dtype=torch.float64))


def test_frozen_product_evaluates_both_cheap_children_on_the_same_inputs() -> None:
    left = priors.FrozenDiscretizationFunction(
        centers=((-1.0,), (1.0,)),
        linear=priors.FrozenLinearFunction(((2.0,),)),
    )
    function = priors.FrozenProductFunction(left=left, right=priors.FrozenLinearFunction(((3.0,),)))
    result = priors.evaluate_frozen_function(function, torch.tensor([[0.75]], dtype=torch.float64))
    torch.testing.assert_close(result, torch.tensor([[4.5]], dtype=torch.float64))


def test_frozen_mlp_keeps_start_and_end_activation_placement() -> None:
    activation = priors.calibrate_source_activation(
        priors.CausalActivation("relu"),
        np.array([[-2.0], [2.0]]),
        rng=np.random.default_rng(17),
        scale=1.0,
        calibration_id="independent-function-fit",
    )
    assert activation.shift_rows == (1,)
    linear = priors.FrozenLinearFunction(((-1.0,),))
    start = priors.FrozenMLPFunction(
        linears=(linear,), activations=(activation,), start_activation=True
    )
    end = priors.FrozenMLPFunction(
        linears=(linear,), activations=(activation,), end_activation=True
    )
    values = torch.tensor([[3.0]], dtype=torch.float64)
    before = priors.evaluate_frozen_function(start, values)
    after = priors.evaluate_frozen_function(end, values)
    torch.testing.assert_close(before, torch.tensor([[-10000 / 2.0001]], dtype=torch.float64))
    torch.testing.assert_close(after, torch.zeros((1, 1), dtype=torch.float64))


def test_frozen_function_json_roundtrips_all_families_and_rejects_nonarray_records() -> None:
    linear = priors.FrozenLinearFunction(((2.0,),))
    activation = priors.calibrate_source_activation(
        priors.CausalActivation("tanh"),
        np.array([[-2.0], [2.0]]),
        rng=np.random.default_rng(17),
        scale=1.0,
        calibration_id="independent-roundtrip-fit",
    )
    functions = (
        linear,
        priors.FrozenQuadraticFunction(1, (0,), (((1.0, 3.0), (5.0, 7.0)),)),
        priors.FrozenTreeFunction(1, ((0,),), ((1.0,),), (((2.0,), (3.0,)),)),
        priors.FrozenDiscretizationFunction(((-1.0,), (1.0,)), linear, 0.5),
        priors.FrozenRFFFunction(((1.0,),), (0.0,), ((2.0,),)),
        priors.FrozenEMFunction(
            ((0.0,), (1.0,)), (1.0, 2.0), priors.FrozenLinearFunction(((1.0, 2.0),))
        ),
        priors.FrozenProductFunction(linear, linear),
        priors.FrozenMLPFunction((linear,), (activation,), end_activation=True),
    )
    values = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float64)
    original_values = values.clone()
    rng_before = torch.random.get_rng_state().clone()
    for function in functions:
        data = json.loads(json.dumps(asdict(function)))
        restored = priors.frozen_function_from_dict(data)
        assert restored == function
        complete = priors.evaluate_frozen_function(restored, values)
        torch.testing.assert_close(
            complete,
            priors.evaluate_frozen_function(function, values),
        )
        chunks = torch.cat(
            [priors.evaluate_frozen_function(restored, values[j : j + 1]) for j in range(3)]
        )
        torch.testing.assert_close(chunks, complete)
        complete.fill_(12345)
        torch.testing.assert_close(priors.evaluate_frozen_function(restored, values), chunks)
        assert asdict(restored) == asdict(function)
        with pytest.raises(ValueError, match="limit"):
            priors.frozen_function_from_dict(data, limits=priors.CausalLimits(parameters=1))
    assert torch.equal(values, original_values)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    with pytest.raises(ValueError, match="array"):
        priors.frozen_function_from_dict(
            {"version": "frozen-linear-function-v1", "matrix": {"2": [1.0]}}
        )


def test_discretization_does_not_turn_distinct_large_distances_into_logarithmic_ties() -> None:
    near = 1e10
    far = math.nextafter(near, math.inf)
    function = priors.FrozenDiscretizationFunction(
        centers=((far,), (near,)),
        linear=priors.FrozenLinearFunction(((1e-10,),)),
    )
    result = priors.evaluate_frozen_function(function, torch.zeros((1, 1), dtype=torch.float64))
    assert result.item() == 1.0


def test_frozen_function_admission_honors_exact_and_one_below_caller_limits() -> None:
    function = priors.FrozenLinearFunction(((1.0, 2.0), (3.0, 4.0)))
    values = torch.tensor([[2.0, -1.0]], dtype=torch.float64)
    # Four coefficients plus the shared geometry allowance: 12 parameters,
    # 56 float64 working channels for one row plus 128 bytes per parameter.
    exact = priors.CausalLimits(parameters=12, working_bytes=1984)
    result = priors.evaluate_frozen_function(function, values, limits=exact)
    torch.testing.assert_close(result, torch.tensor([[0.0, 2.0]], dtype=torch.float64))
    for limits in (
        priors.CausalLimits(parameters=11),
        priors.CausalLimits(working_bytes=1983),
        priors.CausalLimits(nodes=1),
        priors.CausalLimits(node_channels=1),
        priors.CausalLimits(total_channels=3),
    ):
        with pytest.raises(ValueError, match="limit"):
            priors.evaluate_frozen_function(function, values, limits=limits)
    with pytest.raises(ValueError, match="limit"):
        priors.evaluate_frozen_function(function, values.repeat(2, 1), limits=exact)
